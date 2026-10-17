"""Congruence-certified extraction connected to the standalone ADF generator."""
from .contracts import validate_contract
from .egraph import EGraph,check_certificate
from .egraph_rules import default_equality_registry,saturate
from .global_extract import extract_regions,materialize
from .semantics import build,expression_id
from .ir_planner import bind_expression
from .execution_ir import build_execution
from .cost_refinement import refine_for_cost


def make_egraph_plan(contract,certificate,groups=1,gather='flat',registry=None,execution_steps=()):
    spec,_=validate_contract(contract);registry=registry or default_equality_registry()
    _,expressions=check_certificate(certificate,spec,contract['id'],registry)
    seed=expression_id(build(spec)['materialized_layers'].expression)
    if certificate['roots'] != [seed] or len(expressions)!=1:
        raise ValueError('certificate is not rooted at the contract program')
    return bind_expression(contract,expressions[0],groups,gather,registry,execution_steps,certificate=certificate)


def validate_egraph_plan(contract,plan,registry=None):
    expected=make_egraph_plan(contract,plan['equality_certificate'],plan['row_groups'],
        plan['gather'],registry,plan.get('execution_steps',()))
    if plan!=expected:raise ValueError('e-graph plan failed independent certificate/lowering replay')


def enumerate_egraph_plans(contract,*,registry=None,max_rounds=2,max_terms=600,max_applications=32,
                           max_states=10000,keep=6,max_executions=8,execution_depth=2):
    spec,target=validate_contract(contract);registry=registry or default_equality_registry()
    graph=EGraph(spec,contract['id'],registry)
    root=graph.add(build(spec)['materialized_layers'].expression)
    saturation=saturate(graph,max_rounds=max_rounds,max_terms=max_terms,max_applications=max_applications)
    projection=lambda e:refine_for_cost(e,spec,target,registry)
    candidates,unknown,extraction=extract_regions(graph,[root],max_states=max_states,keep=keep,
                                                  arena_cap=target.tile_memory_bytes,refinement=projection)
    # Materialize all selections BEFORE certificate additions can change canonical IDs.
    expressions=[materialize(graph,c['selected'],[root])[0] for c in candidates]
    original=build(spec)['materialized_layers'].expression
    # A structural proxy must never erase the reference before time coverage is
    # established. This is a validation allocation policy, not a ranking change.
    baseline_retained=original not in expressions
    if baseline_retained:
        expressions.append(original)
        candidates.append({'id':'reference_'+expression_id(original),'cost_status':'reference_retained',
                           'metrics':{'latency_ns':None,'reason':'retain semantic reference outside structural top-k'}})
    plans=[];refused=[];execution_reports=[]
    for candidate,expr in zip(candidates,expressions):
        certificate=graph.certificate([root],[expr])
        # One expensive independent replay here; subsequent bindings use this checked expression.
        _,checked=check_certificate(certificate,spec,contract['id'],registry)
        for groups in range(1,min(8,target.max_compute_tiles,spec.m//4)+1):
            if spec.m%(4*groups):continue
            for gather in (('flat','tree') if groups>2 else ('flat',)):
                context={'spec':spec,'target':target,'groups':groups}
                try:
                    seed=build_execution(checked[0],spec,groups,gather)
                    variants,er=registry.execution.explore(seed,context,max_executions,execution_depth)
                    refined,trace=refine_for_cost(checked[0],spec,target,registry,groups,gather)
                    # The exact projection used for costing must be deployable,
                    # even when a separate execution-exploration budget is small.
                    if not any(v['state']==refined for v in variants):
                        variants.append({'state':refined,'steps':trace,'depth':len(trace)})
                    er['cost_projection_retained']=True
                    execution_reports.append({'extraction_id':candidate['id'],'groups':groups,
                                               'gather':gather,'report':er})
                    for v in variants:
                        try:
                            plans.append(bind_expression(contract,checked[0],groups,gather,registry,
                                                         v['steps'],certificate=certificate))
                        except ValueError as e:refused.append(str(e))
                except ValueError as e:refused.append(str(e))
    unique={p['id']:p for p in plans}
    return list(unique.values()),{'saturation':saturation,'extraction':extraction,
        'structural_candidates':candidates,'unknown_candidates':unknown,'lowering_refused':refused,
        'execution_explorations':execution_reports,'plans':len(unique),
        'reference_retained_outside_top_k':baseline_retained,
        'latency_ns':None,'latency_prediction_status':'unsupported_AIE_ML',
        'scope':'structural global extraction followed by bounded deployment/execution expansion; not a joint native-latency optimum'}
