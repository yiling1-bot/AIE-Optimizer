"""Audit observed AIE-ML function cycles; never relabel them isolated costs."""
from pathlib import Path
from hashlib import sha256
import json,xml.etree.ElementTree as ET
from .compiled_profile import read_compiled_profile
from .runner import parse_output
from .contracts import identity


def audit_function_profiles(artifact,simulation):
    root=Path(artifact);sim=Path(simulation)
    compiled=read_compiled_profile(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=sum(json.loads((root/'golden.json').read_text(encoding='utf-8')),[])
    n=manifest['contract']['spec']['m']*manifest['contract']['spec']['n']
    values,complete=parse_output(sim/'out_y.txt',n,manifest['frames'])
    if values!=expected:raise ValueError('profile run numerical mismatch')
    original=json.loads((root/'native_result.json').read_text(encoding='utf-8'))
    if original.get('source_digest')!=manifest['source_digest'] or original.get('status')!='verified':
        raise ValueError('verified source-matched original run required')
    reports=[];pins={};programs={}
    for binding in manifest['kernel_bindings']:
        col,row=binding['tile'];path=sim/f'profile_funct_{col}_{row}.xml';raw=path.read_bytes()
        binary=root/'Work/aie'/f'{col}_{row}'/'Release'/f'{col}_{row}'
        linked=binary.read_bytes()
        if not linked.startswith(b'\x7fELF'):raise ValueError('actual linked ELF evidence required')
        programs[str(binary.relative_to(root))]=sha256(linked).hexdigest()
        document=ET.fromstring(raw);functions=[]
        for f in document.findall('function'):
            record={'name':f.findtext('function_name'),'symbol':f.findtext('mangled_function_name'),
                    'calls':int(f.findtext('calls')),'low_pc':int(f.findtext('low_pc')),'high_pc':int(f.findtext('high_pc'))}
            for tag in ('function_time','function_and_descendants_time'):
                record[tag]={c.tag:int(c.text) for c in f.find(tag)}
                times=record[tag]
                if min(times.values())<0 or times['minimum_cycle_count']>times['maximum_cycle_count']:
                    raise ValueError('invalid function profile cycle range')
            functions.append(record)
        own=sum(f['function_time']['total_cycle_count'] for f in functions)
        total=int(document.findtext('report_cycle_count'))
        if own!=total:raise ValueError('exclusive function cycles do not cover report total')
        kernel=[f for f in functions if f['name']==binding['function']]
        if len(kernel)!=1 or kernel[0]['calls']!=manifest['frames']:raise ValueError('kernel invocation count differs from workload')
        reports.append({'kernel':binding['function'],'tile':[col,row],'report_cycle_count':total,'functions':functions,
                        'profile_cycles_are_isolated_compute':False})
        pins[path.name]=sha256(raw).hexdigest()
    result={'schema':'aie-optimizer-observed-function-profile-v1','source_digest':manifest['source_digest'],
        'compiled_profile_id':compiled['id'],'profile_sha256':pins,'linked_program_sha256':programs,
        'frames':manifest['frames'],'exact_values':len(values),'mismatches':0,'frame_completion_ns':complete,
        'matches_original_timing':complete==original['frame_completion_ns'],
        'observed_functions':reports,'compute_cost_admitted':False,'latency_prediction':None,
        'scope':'observed in-context function/inclusive cycles; not target-free prediction or calibrated uncontended service',
        'limitations':['function self and descendant cycles must not be added twice',
                       'min/max variation may contain context effects; stable min/max does not prove stall freedom',
                       'per-call detail absent when simulator execution tracing is disabled']}
    result['id']=identity(result);return result
