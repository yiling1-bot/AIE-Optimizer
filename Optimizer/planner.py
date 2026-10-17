"""Compose semantic rewrites and bind row regions to distinct physical slots."""
from dataclasses import asdict
from .contracts import identity, validate_contract
from .semantics import build, partition_output, rewrite_k_partition, logical_stats, expression_id


MODES = ('materialized', 'full_reuse', 'row_streaming', 'hidden_accumulation', 'recompute')


def expression_for(spec, mode, groups):
    expr = build(spec)['materialized_layers'].expression
    chunk = spec.m//groups
    if mode == 'row_streaming': chunk = 4
    if mode == 'hidden_accumulation':
        expr = rewrite_k_partition(expr, 'L1', 8, spec)
    if mode == 'recompute':
        expr = partition_output(expr, 1, 8, spec, rematerialize=True)
    return partition_output(expr, 0, chunk, spec)


def gather_regions(groups, strategy='flat'):
    if strategy not in ('flat','tree'):raise ValueError('unsupported output gather')
    children={g:[] for g in range(groups)};regions={g:[g,g+1] for g in range(groups)}
    if strategy=='flat':
        children[groups-1]=[(g,g,g+1) for g in range(groups-1)]
        regions[groups-1]=[0,groups]
    else:
        def build(lo,hi):
            if hi-lo==1:return lo
            mid=(lo+hi)//2;left=build(lo,mid);right=build(mid,hi)
            children[right].insert(0,(left,lo,mid));regions[right]=[lo,hi]
            return right
        build(0,groups)
    return children,regions


def make_plan(contract, mode, groups, gather='flat'):
    spec, target = validate_contract(contract)
    if mode not in MODES or type(groups) is not int or groups < 1:
        raise ValueError('unsupported mode or row group count')
    if spec.m % (4*groups) or any(d % 8 for d in (spec.k, spec.hidden, spec.n)):
        raise ValueError('backend requires complete 4x8 microtiles and equal row regions')
    if groups > min(target.columns, 8):
        raise ValueError('initial gather/backend supports up to eight row groups')
    gather_regions(groups,gather)
    slots_per_group = 2 if mode == 'materialized' else 1
    if slots_per_group*groups > target.max_compute_tiles:
        raise ValueError('compute tile cap exceeded')
    group_rows = spec.m//groups
    slots=[]
    for g in range(groups):
        if mode == 'materialized':
            slots.append({'id':f'l0_{g}', 'tile':[g,2], 'group':g, 'stages':['L0']})
            slots.append({'id':f'l1_{g}', 'tile':[g,3], 'group':g, 'stages':['L1']})
        else:
            slots.append({'id':f'fused_{g}', 'tile':[g,2], 'group':g, 'stages':['L0','L1']})
    expr=expression_for(spec, mode, groups)
    # Conservative declared buffers/scratch, not a compiler memory-map claim.
    full_inputs=spec.m*spec.k + spec.k*spec.hidden + spec.hidden*spec.n
    scratch=(group_rows*spec.hidden + 5*group_rows*spec.n + spec.m*spec.n + 8192)
    declared_upper=2*(full_inputs+group_rows*spec.hidden+spec.m*spec.n)+scratch
    if declared_upper > target.tile_memory_bytes:
        raise ValueError('conservative local storage cap exceeded')
    plan={'schema':'aie-optimizer-plan-v1','contract_id':contract['id'], 'mode':mode,
          'row_groups':groups,'group_rows':group_rows,'slots':slots,
          'planned_compute_tiles':len({tuple(s['tile']) for s in slots}),
          'compiled_compute_tiles':None,'native_legality':'unverified',
          'expression_id':expression_id(expr),'logical_stats':logical_stats(expr,spec),
          'derivation':['source_semantics',f'partition_rows:{group_rows}',
                        'dedicated_layer_slots' if mode=='materialized' else 'reuse_layer_slots',mode],
          'resources':{'declared_per_tile_memory_upper_bytes':declared_upper,
                       'source_bytes_per_frame':full_inputs,'input_plios':3,'output_plios':1,
                       'compiler_memory_bytes':None,'compiler_dma_endpoints':None},
          'scope':'row partition and whole-region implementation choices; no arbitrary mapping or cascade backend yet'}
    if gather!='flat':
        plan['gather']=gather
        plan['derivation'].append('ordered_output_tree')
    plan['id']=identity(plan)
    return plan


def enumerate_plans(contract):
    spec,_=validate_contract(contract)
    plans=[]; refused=[]
    for groups in range(1, min(spec.m//4,8)+1):
        if spec.m % (groups*4): continue
        for mode in MODES:
            for gather in (('flat','tree') if groups>2 else ('flat',)):
                try: plans.append(make_plan(contract,mode,groups,gather))
                except ValueError as e: refused.append({'mode':mode,'row_groups':groups,'gather':gather,'reason':str(e)})
    return plans,refused


def validate_plan(contract, plan, registry=None):
    if plan.get('schema') == 'aie-optimizer-egraph-plan-v3':
        from .egraph_planner import validate_egraph_plan
        return validate_egraph_plan(contract,plan,registry)
    if plan.get('schema') == 'aie-optimizer-ir-plan-v2':
        from .ir_planner import validate_ir_plan
        return validate_ir_plan(contract,plan,registry)
    if make_plan(contract,plan['mode'],plan['row_groups'],plan.get('gather','flat')) != plan:
        raise ValueError('plan is not a reproducible supported extraction')


def analyzer_request(contract, plan, registry=None):
    validate_plan(contract,plan,registry)
    request={'schema':'aie-optimizer-evaluation-request-v1','contract_id':contract['id'],
        'candidate_id':plan['id'],'architecture':contract['target']['architecture'],
        'status':'unsupported','latency_ns':None,
        'reason':'existing Model.Timing native event frontend is AIE1; AIE-ML window/multicast/gather coverage is not established',
        'required_evidence':['compiled mapping','kernel schedule profiles','buffer/lock/DMA phases',
                             'common external input service boundary'],
        'simulation_timing_used_as_prediction':False}
    if 'execution' in plan:
        from .stage_costs import stage_request
        request['stage_cost_request']=stage_request(contract,plan)
    return request
