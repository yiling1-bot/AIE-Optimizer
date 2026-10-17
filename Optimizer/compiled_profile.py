"""Extract compiler-only evidence from current or frozen generated artifacts.

This verifies source closure and compiler diagnostics without replaying a newer
generator against an older certificate. It is not a substitute for native_review
or a cycle estimator. Runtime output files are never used to build this profile.
"""
from pathlib import Path
from hashlib import sha256
import json,re
from .contracts import identity,validate_contract
from .stage_costs import stage_request


def read_compiled_profile(directory):
    root=Path(directory).resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='aie-optimizer-artifact-v1':raise ValueError('generated artifact manifest required')
    hashes=manifest['files']
    if identity(hashes)!=manifest['source_digest']:raise ValueError('source digest mismatch')
    for name,digest in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('artifact source hash mismatch: '+name)
    spec,target=validate_contract(manifest['contract']);plan=manifest['plan']
    if identity({k:v for k,v in plan.items() if k!='id'})!=plan['id']:raise ValueError('plan identity mismatch')
    config_path=root/'Work/config/scsim_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))['SimulationConfig']
    if config.get('aiearch') not in ('aie2','aie-ml') or target.architecture!='AIE-ML':raise ValueError('AIE-ML architecture evidence required')
    from .runner import mapping_report,validate_plios
    mapping_report(root/'Work/reports/compiler_report.json',manifest['kernel_bindings'])
    validate_plios(config,manifest)
    request=stage_request(manifest['contract'],plan);profiles=[];evidence={}
    def read(path):
        raw=path.read_bytes();evidence[path.relative_to(root).as_posix()]=sha256(raw).hexdigest()
        return raw.decode('utf-8',errors='replace')
    read(config_path);read(root/'Work/reports/compiler_report.json')
    for binding in manifest['kernel_bindings']:
        col,row=binding['tile'];tile=f'{col}_{row}'
        folder=root/'Work/aie'/tile/'Release';mp=folder/(tile+'.map');ct=folder/(tile+'.calltree')
        mapping=read(mp);calls=read(ct)
        if '__AIE_ARCH__=20' not in calls:raise ValueError('compiler architecture marker missing')
        block=re.search(r"Memory map for memory 'PM':(.*?)(?=Memory map for memory|\Z)",mapping,re.S)
        used=re.search(r'Used\s*=\s*(\d+)',block.group(1)) if block else None
        stack=re.search(r'Maximum stack size\s*:\s*(\d+)',calls)
        if not used or not stack:raise ValueError('compiler memory/stack report is not recognized')
        functions=[]
        for match in re.finditer(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$',calls,re.M):
            values=list(map(int,match.groups()[:6]));functions.append({'symbol':match[7],
                'self_stack_bytes':values[0],'stack_with_descendants_bytes':values[1],
                'call_depth':values[3],'code_bytes':values[4]})
        profiles.append({'kernel':binding['function'],'tile':[col,row],'program_bytes':int(used[1]),
            'maximum_stack_bytes':int(stack[1]),'functions':functions,
            'spill_bytes':None,'compute_cycles':None})
    result={'schema':'aie-compiled-profile-v1','source_digest':manifest['source_digest'],'plan_id':plan['id'],
        'architecture':'AIE-ML','aie_hz':config['aie_freq'],'profiles':profiles,'stage_request':request,
        'compiler_evidence_hashes':evidence,'runtime_timing_used':False,'latency_ns':None,
        'coverage':'compiler memory/call structure and emitted work only; instruction/cycle and lock/DMA phase models missing',
        'semantic_replay':'not performed by this inspector; use the artifact generation snapshot for certificate/native review'}
    result['id']=identity(result);return result
