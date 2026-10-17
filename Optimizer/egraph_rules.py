"""Finite-pattern e-matching, registered equalities and budgeted saturation.

Patterns inspect alternative e-nodes, not just one whole-expression rewrite
trace. Variables denote e-classes. Arbitrary Python guards are applied to finite
witnesses, so the report deliberately does not claim unrestricted saturation.
"""
from dataclasses import dataclass, replace
from .builtin_rules import default_registry, chunks, partition_check
from .registry import Rule, Decision
from .semantics import Expr, expression_id, infer, partition_output


@dataclass(frozen=True)
class Pattern:
    op: str
    children: tuple = ()


def default_equality_registry():
    r = default_registry()
    r.register(Rule('tensor.partition_columns_shared',1,'M01-shared-producer-output-partition',
        ('https://arxiv.org/pdf/2302.07594v1',),
        'ordered disjoint output columns reconstruct the tensor; pure producers may share',
        lambda e,s: ({'chunk':c} for c in chunks(infer(e,s).cols,8)) if e.op=='quantize' else (),
        lambda e,s,p: partition_check(e,s,p,1,8) if e.op=='quantize' else Decision('DENY','requires Q'),
        lambda e,s,p: partition_output(e,1,p['chunk'],s,False)))
    r.register(Rule('tensor.slice_identity',1,'tensor-index-identity',(),
        'a full-domain slice is the identity; nested same-axis slices compose their offsets',
        lambda e,s: ({},) if e.op=='slice' and (e.begin==0 and e.end==
            (infer(e.args[0],s).rows,infer(e.args[0],s).cols)[e.axis]
            or e.args[0].op=='slice' and e.args[0].axis==e.axis) else (),
        lambda e,s,p: Decision('ALLOW','validated slice bounds') if e.op=='slice' and not p
            and (e.begin==0 and e.end==(infer(e.args[0],s).rows,infer(e.args[0],s).cols)[e.axis]
                 or e.args[0].op=='slice' and e.args[0].axis==e.axis)
            else Decision('DENY','not an identity/composable slice'),
        lambda e,s,p: e.args[0] if e.begin==0 and e.end==
            (infer(e.args[0],s).rows,infer(e.args[0],s).cols)[e.axis]
            else replace(e,args=e.args[0].args,begin=e.args[0].begin+e.begin,end=e.args[0].begin+e.end)))
    return r


def match_pattern(graph, root, pattern, bindings=None, snapshot=None):
    """Return (finite expression, e-class bindings) for every matched e-node."""
    witnesses,classes = snapshot if snapshot is not None else (graph.witnesses(),graph.classes())
    def match(eclass, p, env):
        eclass = graph.find(eclass)
        if isinstance(p,str):
            if p in env and env[p] != eclass:
                return
            if eclass in witnesses:
                yield witnesses[eclass], {**env,p:eclass}
            return
        for tid in classes[eclass]:
            e = graph.terms[tid]
            if e.op != p.op or p.children and len(e.args)!=len(p.children):
                continue
            patterns = p.children or tuple('_arg'+str(i) for i in range(len(e.args)))
            def children(index, args, current):
                if index == len(e.args):
                    yield replace(e,args=tuple(args)),current; return
                c = graph.find(expression_id(e.args[index]))
                for value, nxt in match(c,patterns[index],current):
                    yield from children(index+1,args+[value],nxt)
            yield from children(0,[],env)
    yield from match(root,pattern,bindings or {})


def default_patterns():
    return {
        'tensor.partition_rows': (Pattern('quantize'),),
        'tensor.partition_k_int32': (Pattern('matmul'),),
        'tensor.partition_outputs_rematerialize': (Pattern('quantize'),),
        'tensor.partition_columns_shared': (Pattern('quantize'),),
        'tensor.slice_identity': (Pattern('slice'),Pattern('slice',(Pattern('slice'),))),
        'tensor.propagate_slice': tuple(Pattern('slice',(Pattern(op),))
            for op in ('matmul','quantize','sum','concat','rematerialize'))}


def saturate(graph, *, patterns=None, max_rounds=3, max_terms=1000, max_applications=80):
    if any(type(v) is not int or v<1 for v in (max_rounds,max_terms,max_applications)):
        raise ValueError('positive saturation budgets required')
    patterns = patterns or default_patterns()
    if set(patterns)-set(graph.registry._rules):
        raise ValueError('patterns refer to unregistered rules')
    seen = set(); refused = []; rounds = 0; applications = 0; stop = 'round_budget'
    for round_index in range(max_rounds):
        rounds += 1; before = (len(graph.terms),graph.merges)
        # Snapshot matches per round: newly added classes enter the next round.
        tasks = [];snapshot=(graph.witnesses(),graph.classes())
        for root in snapshot[1]:
            for rid, pats in patterns.items():
                rule = graph.registry._rules[rid]
                for pattern in pats:
                    for expr, _ in match_pattern(graph,root,pattern,snapshot=snapshot):
                        for p in rule.match(expr,graph.spec):
                            key = (rid,expression_id(expr),tuple(sorted(p.items())))
                            if key not in seen:
                                seen.add(key); tasks.append((rid,expr,p))
        for rid, expr, params in tasks:
            if applications >= max_applications or len(graph.terms)>=max_terms:
                stop = 'application_budget' if applications>=max_applications else 'term_budget'
                break
            # Validate before mutation and enforce the term budget including RHS.
            try:
                from .registry import Match, walk
                rhs,_ = graph.registry.apply(expr,graph.spec,rid,Match((),tuple(sorted(params.items()))))
                added = {expression_id(n) for e in (expr,rhs) for _,n in walk(e)}-graph.terms.keys()
                if len(graph.terms)+len(added)>max_terms:
                    stop='term_budget'; break
                lhs=graph.add(expr); graph.apply(lhs,rid,params); applications+=1
            except ValueError as e:
                refused.append({'rule':rid,'reason':str(e)})
        else:
            if before == (len(graph.terms),graph.merges):
                stop='no_new_matches_in_witness_policy'; break
            continue
        break
    return {'rounds':rounds,'applications':applications,'terms':len(graph.terms),
        'eclasses':len(graph.members),'congruence_merges':graph.merges,'stop_reason':stop,
        'budget_reached':stop.endswith('_budget'),'refused':refused,
        'scope':'finite pattern e-matching with least-size variable witnesses; not unrestricted equality saturation'}
