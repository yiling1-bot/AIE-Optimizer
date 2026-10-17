"""Lowered stage inventory and explicit, fail-closed compute-cost coverage.

Counts describe emitted operations, not cycles or observed memory transactions.
Measured end-to-end latency is deliberately not accepted as a kernel cost. A
complete graph estimate still requires input/DMA/lock/gather phase coverage.
"""
from math import isfinite
from .contracts import identity,validate_contract
from .execution_ir import validate_execution


def operator_features(op):
    out=op['out'];args=op['args'];kind=op['kind'];elements=out['rows']*out['cols']
    element_bytes=4 if out['dtype']=='int32' else 1
    f={'kind':kind,'macs':0,'mmul_microtiles':0,'qmm_calls':0,
       'explicit_copy_bytes':0,'output_elements':elements,'load_vector_bytes':None,
       'accumulator_grid':op.get('accumulator_grid',[1,1]),'loop_order':op.get('loop_order','row_column')}
    if kind in ('matmul','fused_qmm'):
        reduction=sum(a['cols'] for a in args[::2]) if kind=='fused_qmm' else args[0]['cols']
        f['macs']=elements*reduction;f['mmul_microtiles']=(out['rows']//4)*(out['cols']//8)*(reduction//8)
        if kind=='fused_qmm':
            r,c=f['accumulator_grid'];groups=(out['rows']//(4*r))*(out['cols']//(8*c))
            f['qmm_calls']=1;f['load_vector_bytes']=groups*(reduction//8)*(32*r+64*c)
    elif kind=='concat':f['explicit_copy_bytes']=0 if op.get('concat_forwarded') else elements*element_bytes
    elif kind=='sum':f['explicit_copy_bytes']=elements*element_bytes
    return f


def stage_request(contract,plan):
    spec,target=validate_contract(contract)
    if plan.get('id')!=identity({k:v for k,v in plan.items() if k!='id'}):raise ValueError('plan identity mismatch')
    if plan['contract_id']!=contract['id']:raise ValueError('plan/contract mismatch')
    if 'execution' not in plan:raise ValueError('operator execution IR required for stage inventory')
    validate_execution(plan['execution'],{'spec':spec,'target':target,'groups':plan['row_groups']})
    stages=[]
    # Full plan identity prevents unsound reuse across source, layout, placement,
    # buffering or compiler-generation changes. Broader transfer needs a separate
    # validated coverage predicate; shape equality alone is insufficient.
    context={'architecture':target.architecture,'toolchain':target.toolchain,
             'implementation_fingerprint':plan['implementation_fingerprint'],'plan_id':plan['id']}
    for g,dag in enumerate(plan['execution']['regions']):
        kernel=next(k['id'] for k in plan['execution']['kernels']
            if any(a['op']=='compute' and a['region']==g for a in k['actions']))
        for index,op in enumerate(dag['operators']):
            if op.get('concat_forwarded'):continue  # validated storage identity emits no instructions
            key={'context':context,'kernel':kernel,'region':g,'operator_index':index,'operator':op}
            stages.append({'id':identity(key),'key':key,'features':operator_features(op),
                           'cycles':None,'coverage':'unmodeled'})
    copies=[{'kernel':k['id'],'bytes':sum(a['size'] for a in k['actions'] if a['op']=='copy')}
            for k in plan['execution']['kernels']]
    request={'schema':'aie-stage-cost-request-v1','contract_id':contract['id'],'context':context,'stages':stages,
        'wrapper_copy_bytes':copies,'external_input_bytes_per_frame':spec.m*spec.k+spec.k*spec.hidden+spec.hidden*spec.n,
        'external_output_bytes_per_frame':spec.m*spec.n,
        'interkernel_payload_bytes':sum(p['size'] for k in plan['execution']['kernels'] for p in k['inputs'] if p['source'].startswith('kernel:')),
        'kernel_count':len(plan['execution']['kernels']),
        'missing_phase_models':['PLIO/DMA source service','buffer/lock phases','wrapper/gather copies','contention and startup'],
        'latency_ns':None,'counts_are_time_predictions':False}
    request['id']=identity(request);return request


def resolve_compute_costs(request,catalog):
    if request.get('schema')!='aie-stage-cost-request-v1' or request.get('id')!=identity({k:v for k,v in request.items() if k!='id'}):
        raise ValueError('invalid stage request identity')
    if catalog.get('schema')!='aie-compute-cost-catalog-v1' or not isinstance(catalog.get('entries'),list):
        raise ValueError('invalid compute cost catalog')
    entries={}
    for entry in catalog['entries']:
        required={'stage_id','key','cycles','scope','evidence_kind','evidence_ref','evidence_sha256'}
        if set(entry)!=required or entry['stage_id']!=identity(entry['key']) or entry['stage_id'] in entries:
            raise ValueError('noncanonical or duplicate stage cost')
        if (type(entry['cycles']) not in (int,float) or not isfinite(entry['cycles']) or entry['cycles']<0
            or entry['scope']!='isolated_compute' or entry['evidence_kind'] not in ('compiled_cycle_model','isolated_kernel_measurement')
            or not entry['evidence_ref'] or not isinstance(entry['evidence_sha256'],str)
            or len(entry['evidence_sha256'])!=64 or any(c not in '0123456789abcdef' for c in entry['evidence_sha256'])):
            raise ValueError('positive documented isolated-compute evidence required; end-to-end timing is not a kernel cost')
        entries[entry['stage_id']]=entry
    covered=[];missing=[];totals={}
    for stage in request['stages']:
        entry=entries.get(stage['id'])
        if entry is None:missing.append(stage['id']);continue
        if entry['key']!=stage['key']:raise ValueError('stage context differs despite matching ID')
        covered.append(stage['id']);kernel=stage['key']['kernel']
        totals[kernel]=totals.get(kernel,0)+entry['cycles']
    return {'schema':'aie-compute-cost-coverage-v1','request_id':request['id'],
        'covered_stages':covered,'missing_stages':missing,
        'status':'compute_only_covered' if not missing else 'uncovered',
        'known_serial_compute_cycles_by_kernel':totals,
        'compute_cycles_complete':not missing,'latency_ns':None,
        'missing_phase_models':request['missing_phase_models'],
        'limitation':'catalog provenance is declared, not authenticated; validate calibration evidence before admitting entries; no graph latency prediction'}
