"""Bounded exact branch-and-bound over shared, possibly multi-root DAGs.

One e-node is chosen per demanded e-class, and paid for once across all uses.
Cycles are infeasible. A complete-assignment evaluator may impose actual backend
resource constraints and increase the nonnegative additive lower bound. This is
global within the supplied finite one-representative-per-class space, not an
unrestricted scheduling, cloning, routing or native-latency optimality claim.
"""
from dataclasses import dataclass, replace
import math
from .contracts import identity
from .semantics import expression_id, infer


@dataclass(frozen=True)
class Node:
    id: str
    eclass: str
    children: tuple
    cost: object  # nonnegative finite scalar, or None (uncovered)


def extract_dags(nodes, roots, *, max_states=100000, keep=8, evaluate=None, evaluator_costs_complete=False,
                 visit_complete=None, max_complete=None, option_priority=None):
    if type(max_states) is not int or max_states<1 or type(keep) is not int or keep<1:
        raise ValueError('positive extraction budgets required')
    if not roots:
        raise ValueError('at least one root required')
    if type(evaluator_costs_complete) is not bool:
        raise ValueError('explicit evaluator cost coverage flag required')
    if max_complete is not None and (type(max_complete) is not int or max_complete < 1):
        raise ValueError('positive complete-assignment budget required')
    if option_priority is not None and not callable(option_priority):
        raise ValueError('callable traversal priority required')
    nodes=tuple(nodes)
    by_class = {}; by_id = {}
    for n in nodes:
        if n.id in by_id or (n.cost is not None and
            (type(n.cost) not in (int,float) or not math.isfinite(n.cost) or n.cost<0)):
            raise ValueError('unique nodes and finite nonnegative costs required')
        by_class.setdefault(n.eclass,[]).append(n); by_id[n.id]=n
    if any(c not in by_class for n in nodes for c in n.children) or any(r not in by_class for r in roots):
        raise ValueError('missing e-class implementation')
    reachable = set(); pending = list(roots)
    while pending:
        c = pending.pop()
        if c in reachable: continue
        reachable.add(c); pending.extend(k for n in by_class[c] for k in n.children)
    unknown_possible = any(n.cost is None for c in reachable for n in by_class[c])
    for options in by_class.values():
        options.sort(key=lambda n:((option_priority(n),) if option_priority else ())+
                     (n.cost is None,0 if n.cost is None else n.cost,len(n.children),n.id))
    best = []; unknown = []; states = 0; complete = 0; cycles = 0; refused = 0; pruned = 0
    exhausted = True; evaluator_unknown_seen=False; visitor_budget_reached=False
    # A custom evaluator can discover missing coverage only at a complete DAG.
    # Unless its caller guarantees complete coverage, cost pruning could hide an
    # unknown candidate behind an already known incumbent.
    can_prune=not unknown_possible and (evaluate is None or evaluator_costs_complete)

    def has_cycle(chosen):
        visited=set(); active=set()
        for root in chosen:
            stack=[(root,False)]
            while stack:
                c,exiting=stack.pop()
                if c in visited or c not in chosen:continue
                if exiting:active.remove(c);visited.add(c);continue
                if c in active:return True
                active.add(c);stack.append((c,True))
                stack.extend((k,False) for k in reversed(chosen[c].children))
        return False

    # Explicit stack avoids Python recursion limits for large dependency DAGs.
    stack=[({},set(roots),0,False)]
    while stack:
        if states>=max_states:
            exhausted=False;break
        chosen,demand,lower,uncovered=stack.pop();states+=1
        if can_prune and len(best)>=keep and lower>best[-1]['objective'][0]:
            pruned+=1;continue
        missing=demand-chosen.keys()
        if not missing:
            if max_complete is not None and complete >= max_complete:
                stack.append((chosen,demand,lower,uncovered));exhausted=False;break
            complete+=1
            selected={c:n.id for c,n in sorted(chosen.items())}
            if visit_complete is not None and visit_complete(dict(selected)) is False:
                # A visitor may explicitly refuse admission at its declared
                # budget. The unconsumed assignment remains unexplored.
                complete-=1;stack.append((chosen,demand,lower,uncovered))
                exhausted=False;visitor_budget_reached=True;break
            metrics = evaluate(selected) if evaluate else {'cost':None if uncovered else lower}
            if metrics is None:
                refused+=1;continue
            cost=metrics.get('cost')
            if cost is None and not uncovered:
                if evaluator_costs_complete:raise ValueError('evaluator violated complete-cost coverage contract')
                evaluator_unknown_seen=True
            if uncovered:
                cost=None
            if cost is not None and (type(cost) not in (int,float) or not math.isfinite(cost) or cost<lower):
                raise ValueError('evaluator violates the certified nonnegative lower bound')
            objective=(cost,)+tuple(metrics.get('tie_break',())) if cost is not None else None
            if objective is not None and any(type(v) not in (int,float) or not math.isfinite(v) for v in objective):
                raise ValueError('finite numeric tie breakers required')
            entry={'id':identity(selected),'selected':selected,'objective':objective,
                   'cost_status':'unknown' if cost is None else 'known','metrics':{**metrics,'cost':cost}}
            target=unknown if cost is None else best;target.append(entry)
            target.sort(key=lambda e:(e['id'],) if e['objective'] is None else (*e['objective'],e['id']))
            del target[keep:]
            continue
        # Fail-first expands the tightest demanded class, with deterministic order.
        c=min(missing,key=lambda x:(len(by_class[x]),x))
        for n in reversed(by_class[c]):
            new={**chosen,c:n}
            if has_cycle(new):cycles+=1;continue
            stack.append((new,demand|set(n.children),lower+(n.cost or 0),uncovered or n.cost is None))
    report={'states':states,'complete_assignments':complete,'cycle_refusals':cycles,
        'resource_refusals':refused,'bound_prunes':pruned,'search_exhausted':exhausted,
        'unknown_cost_reachable':unknown_possible or evaluator_unknown_seen,'unexplored_states':len(stack),
        'evaluator_costs_complete':evaluate is None or evaluator_costs_complete,
        'complete_budget_reached':not exhausted and max_complete is not None and complete>=max_complete,
        'visitor_budget_reached':visitor_budget_reached,
        'optimality':('exact_in_known_cost_subspace' if unknown_possible or evaluator_unknown_seen else 'exact_in_supplied_space')
            if exhausted else 'best_found_budget_limited',
        'scope':'global shared DAG, one representative per e-class; supplied roots, costs and evaluator only'}
    return best,unknown,report


def graph_nodes(graph, cost=None):
    def mac_cost(e):
        if e.op!='matmul':return 0
        a,b=(infer(c,graph.spec) for c in e.args)
        return a.rows*a.cols*b.cols
    cost=cost or mac_cost
    return [Node(tid,root,tuple(graph.find(expression_id(c)) for c in graph.terms[tid].args),
                 cost(graph.terms[tid])) for root,terms in graph.classes().items() for tid in terms]


def materialize(graph, selected, roots):
    cache={};active=set()
    def visit(c):
        c=graph.find(c)
        if c in cache:return cache[c]
        if c in active:raise ValueError('cyclic extraction')
        if c not in selected or graph.find(selected[c])!=c:
            raise ValueError('missing or mismatched selected e-node')
        active.add(c);e=graph.terms[selected[c]]
        result=replace(e,args=tuple(visit(expression_id(a)) for a in e.args))
        active.remove(c);cache[c]=result;return result
    return [visit(r) for r in roots]


def extract_regions(graph, roots, *, max_states=10000, keep=8, arena_cap=None, refinement=None):
    """Use actual lowered CSE/rematerialization and allocation, never fake cycles.

    Each root is a separate output region. Cross-root sharing is supported by
    extract_dags, but this ADF adapter currently binds exactly one tensor root.
    """
    from .ir_backend import lower_program
    if len(roots)!=1:raise ValueError('ADF region backend currently requires one output tensor')
    roots=[graph.find(r) for r in roots]
    def evaluate(selected):
        try:
            expr=materialize(graph,selected,roots)[0]
            execution,steps=refinement(expr) if refinement is not None else (None,[])
            _, reports=lower_program(expr,graph.spec,1,execution)
            report=reports[0]
            if arena_cap is not None and report['arena_bytes']>arena_cap:return None
            return {'cost':report['scheduled_macs'],
                    'tie_break':(report['arena_bytes'],report['scheduled_operators']),
                    'arena_bytes':report['arena_bytes'],'operators':report['scheduled_operators'],
                    'cost_unit':'integer_MACs_not_time','latency_ns':None,
                    'evaluation_stage':'after_registered_refinement' if refinement else 'before_execution_refinement',
                    'refinement_rules':[s['rule'] for s in steps]}
        except ValueError:
            return None
    best,unknown,report=extract_dags(graph_nodes(graph),roots,max_states=max_states,keep=keep,evaluate=evaluate,
                                   evaluator_costs_complete=True)
    report['objective']='lexicographic (actual scheduled MACs, allocated scratch bytes, operator count)'
    report['memory_scope']='serial region scratch; plan validation separately checks ADF buffers, stack and target cap'
    report['evaluation_stage']='after_registered_refinement' if refinement else 'before_execution_refinement'
    return best,unknown,report
