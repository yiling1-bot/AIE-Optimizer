"""Bind Optimizer ports to real AIE-ML resources without inventing slot use."""
from pathlib import Path
from hashlib import sha256
import json
from .contracts import identity
from .compiled_profile import read_compiled_profile
from . import analyzer_bridge
from Timing.compiled_resource_graph import build_compiled_resource_graph,file_ref


def inspect_phase_inventory(artifact):
    root=Path(artifact);profile=read_compiled_profile(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));plan=manifest['plan']
    paths={'compiler_report':'Work/reports/compiler_report.json','router_soln':'Work/temp/router_soln.json',
        'driver_control':'Work/ps/c_rts/aie_control.cpp','driver_config':'Work/ps/c_rts/aie_control_config.json',
        'simulator_config':'Work/config/scsim_config.json'}
    request={'schema':'aie-compiled-resource-inputs-v1','provenance_kind':'actual_compiler_artifacts',
        'artifacts':{role:file_ref(root/name) for role,name in paths.items()}}
    graph=build_compiled_resource_graph(request)
    if graph['architecture']!='AIE-ML':raise ValueError('AIE-ML compiler binding required')
    raw=json.loads((root/paths['compiler_report']).read_text(encoding='utf-8'))
    actors={a['function']:a for a in graph['actors']};ports={p['id']:p for p in graph['ports']};bound={}
    if set(actors)!={k['id'] for k in plan['execution']['kernels']}:raise ValueError('compiled actor inventory differs from plan')
    for kernel in plan['execution']['kernels']:
        actor=actors[kernel['id']]
        if actor['actual_compute_tile']!=kernel['tile'] or len(actor['input_ports'])!=len(kernel['inputs']) or len(actor['output_ports'])!=1:
            raise ValueError('compiled placement or port arity differs from plan')
        for ordinal,(name,size,pid) in enumerate([(p['name'],p['size'],pid) for p,pid in zip(kernel['inputs'],actor['input_ports'])]+
                                               [('y',kernel['output_size'],actor['output_ports'][0])]):
            port=ports[pid]
            if port['bytes']!=size or port['parameter_index']!=ordinal:raise ValueError('compiled port extent/order differs')
            buffers=raw['mapping']['portInstanceMapping'].get(pid,{}).get('bufferInfo',[])
            bound[kernel['id']+'.'+name]={'compiler_port':pid,'bytes':size,'raw_buffers':buffers,
                'driver_DMA_slots':port['buffer_slots'],'channel':port.get('channel_resource_id'),
                'actual_slot_uses':None,'initial_free_time_ps':None}
    aliases=[]
    for connection in graph['connections']:
        if connection['source_actor'] is None or connection['target_actor'] is None:continue
        src=connection['src_port_id'];dst=connection['dst_port_id']
        source=raw['mapping']['portInstanceMapping'].get(src,{}).get('bufferInfo',[])
        dest=raw['mapping']['portInstanceMapping'].get(dst,{}).get('bufferInfo',[])
        def physical(rows):return [{k:r[k] for k in ('tile','column','row','offset','size','lock')} for r in rows]
        if source and dest and physical(source)==physical(dest) and not connection['route_ids']:
            aliases.append({'connection':connection['id'],'source_port':src,'destination_port':dst,
                'classification':'compiler_shared_buffer_alias','physical_slots':physical(source),
                'active_DMA_copy':False,'dynamic_lock_protocol_verified':False})
    report={'schema':'aie-optimizer-compiled-phase-inventory-v1','plan_id':plan['id'],'source_digest':manifest['source_digest'],
        'compiled_profile_id':profile['id'],'compiler_binding_id':graph['compile_identity'],
        'compiler_inputs':request,'ports':bound,'shared_aliases':aliases,'resources':graph['counts'],
        'requested_actions':{k['id']:k['actions'] for k in plan['execution']['kernels']},
        'actual_compiled_action_order':None,'native_admitted':False,'timing_prediction':None,
        'gaps':['compiled AIE-ML wrapper order and per-call physical slot sequence',
                'counter lock/BD readiness and multicast phase service costs',
                'core region binding and independent native ranking holdout'],
        'scope':'static physical inventory/aliases verified; requested execution actions are not promoted to actual dynamic execution'}
    report['id']=identity(report);return report
