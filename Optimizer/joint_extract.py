"""Bounded joint semantic/deployment/execution selection using phase latency.

All complete DAG choices encountered under the state budget reach deployment
evaluation; no MAC/scratch top-k is applied before latency. The result is a
conditional ranking of the supplied finite space, not native global optimality.
"""
from .contracts import identity,validate_contract
from .egraph import EGraph,check_certificate
from .egraph_rules import default_equality_registry,saturate
from .global_extract import extract_dags,graph_nodes,materialize
from .semantics import build,expression_id
from .ir_planner import bind_expression
from .execution_ir import build_execution
from .cost_refinement import refine_for_cost
from .phase_costs import finite_score


def rank_plans(plans,evaluations,*,metric='first_output'):
    """Unknown costs never become infinity, zero or certified dominated points."""
    known=[];unknown=[];comparison=None;scenario=None
    for plan in plans:
        result=evaluations[plan['id']];score=finite_score(result,metric)
        if score is None:unknown.append(plan['id']);continue
        if result.get('plan_id')!=plan['id']:raise ValueError('phase estimate belongs to another candidate')
        key=result['comparison_key'];member=result['scenario_id']
        if comparison is None:comparison,scenario=key,member
        if (key,member)!=(comparison,scenario):raise ValueError('mixed workloads or cost-family members cannot be ranked together')
        known.append({'plan_id':plan['id'],'conditional_objective_ns':score,'planned_cores':plan['planned_compute_tiles']})
    known.sort(key=lambda r:(r['conditional_objective_ns'],r['planned_cores'],r['plan_id']))
    frontier=[r for r in known if not any(q['conditional_objective_ns']<=r['conditional_objective_ns']
        and q['planned_cores']<=r['planned_cores'] and (q['conditional_objective_ns']<r['conditional_objective_ns']
        or q['planned_cores']<r['planned_cores']) for q in known)]
    return {'metric':metric,'ranked':known,'conditional_pareto':frontier,'unknown_plan_ids':unknown,
        'comparison_key':comparison,'scenario_id':scenario,'native_frontier':None,
        'scope':'conditional point-model ranking and planned cores; no deletion of unknown candidates'}


def enumerate_joint_plans(contract,*,evaluator=None,registry=None,max_rounds=2,max_terms=600,
                          max_applications=32,max_states=10000,max_plans=256,max_executions=8,
                          execution_depth=2,metric='first_output'):
    for n in (max_plans,max_executions,max_states):
        if type(n) is not int or n<1:raise ValueError('positive joint search budgets required')
    spec,target=validate_contract(contract);registry=registry or default_equality_registry()
    graph=EGraph(spec,contract['id'],registry);original=build(spec)['materialized_layers'].expression;root=graph.add(original)
    saturation=saturate(graph,max_rounds=max_rounds,max_terms=max_terms,max_applications=max_applications)
    expressions={expression_id(original):original}
    def collect(selected):
        expr=materialize(graph,selected,[root])[0];expressions[expression_id(expr)]=expr
    # Zero is a universal time lower bound, not a MAC-to-time conversion. A
    # visitor consumes EVERY complete selection, not just extract_dags' top-k.
    _,_,enumeration=extract_dags(graph_nodes(graph,cost=lambda e:0),[graph.find(root)],max_states=max_states,
                                keep=1,evaluate=lambda selected:{'cost':0},visit_complete=collect)
    plans={};evaluations={};refused=[];reports=[];plan_budget_hit=False;reference_id=None
    for eid,expr in expressions.items():
        if len(plans)>=max_plans:plan_budget_hit=True;break
        certificate=graph.certificate([root],[expr]);_,checked=check_certificate(certificate,spec,contract['id'],registry)
        for groups in range(1,min(8,target.max_compute_tiles,spec.m//4)+1):
            if spec.m%(4*groups):continue
            for gather in (('flat','tree') if groups>2 else ('flat',)):
                if len(plans)>=max_plans:plan_budget_hit=True;break
                context={'spec':spec,'target':target,'groups':groups}
                try:
                    seed=build_execution(checked[0],spec,groups,gather)
                    variants,report=registry.execution.explore(seed,context,max_executions,execution_depth)
                    refined,trace=refine_for_cost(checked[0],spec,target,registry,groups,gather)
                    if not any(v['state']==refined for v in variants):variants.append({'state':refined,'steps':trace})
                    # Expose deferred acquire after the full fusion/storage
                    # projection, which depth-limited BFS can otherwise miss.
                    for rid,params in registry.execution.proposals(refined,context):
                        if rid!='schedule.defer_gather_acquire':continue
                        state,step=registry.execution.apply(refined,context,rid,params)
                        if not any(v['state']==state for v in variants):variants.append({'state':state,'steps':trace+[step]})
                    reports.append({'expression_id':eid,'groups':groups,'gather':gather,'exploration':report})
                    for variant in variants:
                        if len(plans)>=max_plans:plan_budget_hit=True;break
                        try:plan=bind_expression(contract,checked[0],groups,gather,registry,variant['steps'],certificate=certificate)
                        except ValueError as error:refused.append({'expression_id':eid,'reason':str(error)});continue
                        if plan['id'] in plans:continue
                        plans[plan['id']]=plan
                        if reference_id is None:reference_id=plan['id']
                        result=evaluator(plan) if evaluator is not None else {'status':'needs_evaluator','conditional_latency_ns':None,'latency_ns':None,'native_admitted':False}
                        evaluations[plan['id']]=result
                except ValueError as error:
                    # Input/evaluator errors must not silently disappear as
                    # unprofitable candidates; only generation is handled above.
                    raise ValueError('joint binding/evaluation failed: '+str(error)) from error
    values=list(plans.values());ranking=rank_plans(values,evaluations,metric=metric)
    ranked_ids=[r['plan_id'] for r in ranking['ranked']]
    ordered=list(dict.fromkeys(([reference_id] if reference_id else [])+ranked_ids+ranking['unknown_plan_ids']))
    report={'schema':'aie-optimizer-joint-extraction-v1','saturation':saturation,'dag_enumeration':enumeration,
        'semantic_expressions_seen':len(expressions),'plans_evaluated':len(values),'max_plans':max_plans,
        'plan_budget_reached':plan_budget_hit,'execution_explorations':reports,'refused':refused,
        'reference_plan_id':reference_id,'ranking':ranking,'evaluations':evaluations,
        'structural_proxy_prunes':0,'native_optimality':False,'latency_ns':None,
        'scope':'joint latency comparison over visited semantic DAGs and bounded core/execution choices; native timing remains unadmitted'}
    report['id']=identity(report)
    return [plans[pid] for pid in ordered],report


def choose_joint(plans,attempted,report):
    """Reference first, then modeled rank with one unknown slot in every four."""
    used={p['id'] for p in attempted};pending={p['id']:p for p in plans if p['id'] not in used}
    if not pending:return None
    reference=report['reference_plan_id']
    if reference in pending:return pending[reference]
    unknown=[pid for pid in report['ranking']['unknown_plan_ids'] if pid in pending]
    known=[r['plan_id'] for r in report['ranking']['ranked'] if r['plan_id'] in pending]
    if unknown and (not known or len(attempted)%4==3):return pending[unknown[0]]
    return pending[known[0]] if known else next(iter(pending.values()))
