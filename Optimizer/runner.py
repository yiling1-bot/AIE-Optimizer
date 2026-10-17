"""Bounded execution and evidence checking for generated artifacts only."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from .contracts import identity
from .planner import validate_plan
from .resources import compiler_resources


def check_artifact(root,registry=None):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != 'aie-optimizer-artifact-v1': raise ValueError('artifact schema')
    validate_plan(manifest['contract'],manifest['plan'],registry)
    if identity(manifest['files']) != manifest['source_digest']: raise ValueError('digest mismatch')
    for name,expected in manifest['files'].items():
        p=(root/name).resolve()
        if root not in p.parents or not p.is_file(): raise ValueError('unsafe or missing artifact file')
        if sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError('artifact changed: '+name)
    return root,manifest


def command(root,args,timeout,label):
    started=time.monotonic()
    try:
        with (root/(label+'.log')).open('w') as log:
            process=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=(os.name!='nt'))
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name!='nt': os.killpg(process.pid,signal.SIGKILL)
                else: process.kill()
                process.wait();raise
        return {'exit_code':code,'wall_seconds':time.monotonic()-started,'command':args}
    except subprocess.TimeoutExpired:
        return {'exit_code':None,'timeout':True,'wall_seconds':time.monotonic()-started,'command':args}


def cpu_check(root,compiler='g++',registry=None):
    root,m=check_artifact(root,registry)
    exe=root/('cpu_check.exe' if __import__('os').name=='nt' else 'cpu_check')
    compile_result=command(root,[compiler,'-std=c++17','-O2',str(root/'cpu_check.cpp'),'-o',str(exe)],120,'cpu_compile')
    run=command(root,[str(exe)],30,'cpu_run') if compile_result['exit_code']==0 else None
    ok=run is not None and run['exit_code']==0 and (root/'cpu_run.log').read_text().strip()=='0'
    result={'schema':'aie-optimizer-cpu-check-v1','candidate_id':m['plan']['id'],
            'status':'passed' if ok else 'failed','compile':compile_result,'run':run,
            'frames':m['frames'],'source_digest':m['source_digest'],
            'scope':'generated packed compute code with scalar MMUL primitive; no ADF scheduling emulation'}
    (root/'cpu_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def parse_output(path,frame_elements,frames):
    values=[]; times=[]; current=None
    units={'ps':.001,'ns':1.,'us':1000.,'ms':1000000.}
    for raw in Path(path).read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith('#') or line=='TLAST': continue
        match=re.fullmatch(r'T\s+([0-9.eE+-]+)\s+(ps|ns|us|ms)',line)
        if match:
            value=float(match[1])*units[match[2]]
            if not __import__('math').isfinite(value) or value<0 or (current is not None and value<current):
                raise ValueError('invalid output timestamp')
            current=value; continue
        if current is None or not all(re.fullmatch(r'[-+]?\d+',v) for v in line.split()):
            raise ValueError('invalid output data or missing timestamp')
        data=list(map(int,line.split()))
        if any(not -128<=v<=127 for v in data): raise ValueError('output outside int8')
        values.extend(data);times.extend([current]*len(data))
    if len(values)!=frame_elements*frames: raise ValueError('incomplete or extra output')
    complete=[times[(i+1)*frame_elements-1] for i in range(frames)]
    return values,complete


def mapping_report(path,bindings):
    report=json.loads(Path(path).read_text()); actual={}
    for instance_id,instance in report['blockInstances'].items():
        block=report['blockTypes'][instance['blockType']]
        if block.get('type')!='function' or block.get('function')=='ps_main': continue
        name=block['function']; core=report['mapping']['blockInstanceMapping'][instance_id]['coreInfo']
        if name in actual: raise ValueError('duplicate kernel function mapping')
        actual[name]=[core['column'],core['row']]
    expected={b['function']:b['tile'] for b in bindings}
    if actual!=expected: raise ValueError('compiled mapping differs: '+json.dumps(actual))
    return actual


def tool_identity(version):
    version_match=re.search(r'v\+\+ v(\d{4}\.\d+)',version)
    build_match=re.search(r'SW Build ([^\r\n]+)',version)
    if not version_match or not build_match: raise ValueError('unrecognized compiler version')
    return {'version':version_match[1],'build':build_match[1].strip()}


def compile_diagnosis(log_text):
    if 'UDM Placer Did NOT Finish Successfully' in log_text or 'Global placement failed' in log_text:
        code='placement_failure_observed'
    elif 'ERROR:' in log_text: code='compiler_error'
    else: code='unclassified_compile_failure'
    return {'code':code,'compiler_exception_observed':'EXCEPTION:' in log_text,
            'scope':'observed tool failure; does not prove mathematical or hardware infeasibility'}


def comparison_key(manifest,result):
    return identity({'spec':manifest['contract']['spec'],'interface':manifest['contract']['interface'],
        'aie_hz':result['aie_hz'],'plio_hz':manifest['contract']['target']['plio_hz'],
        'architecture':result['compiled_architecture'],'tool':tool_identity(result['tool_version']),
        'platform':result['platform_sha256'],
        'fixtures':{k:v for k,v in manifest['files'].items() if k.startswith('data/')},
        'frames':manifest['frames'],'metric':result['metric']})


def validate_plios(config,manifest):
    blocks={b['name']:b for b in config['pl_ip_block'] if b.get('axi_stream')}
    if set(blocks)!={'X','W0','W1','Y'}: raise ValueError('unexpected PLIO inventory')
    hz=manifest['contract']['target']['plio_hz']
    for name,b in blocks.items():
        direction='me_to_pl' if name=='Y' else 'pl_to_me'
        filename='out_y.txt' if name=='Y' else 'data/'+name+'.txt'
        if (b['pl_freq']!=hz or b['arguments'][0]!=filename or b['arguments'][-1]!='int8'
                or any(s['direction']!=direction or s['bus_width']!=128 for s in b['axi_stream'])):
            raise ValueError('compiled PLIO service contract mismatch')


def review_native(root,registry=None):
    """Revalidate archived measurements without rewriting the execution record."""
    root,m=check_artifact(root,registry)
    record=root/'native_result.json';result=json.loads(record.read_text())
    if (result.get('status')!='verified' or result.get('source_digest')!=m['source_digest']
            or result.get('compile',{}).get('exit_code')!=0 or result.get('simulate',{}).get('exit_code')!=0):
        raise ValueError('successful native execution with matching source required')
    mapping=mapping_report(root/'Work/reports/compiler_report.json',m['kernel_bindings'])
    config=json.loads((root/'Work/config/scsim_config.json').read_text())['SimulationConfig']
    validate_plios(config,m)
    if config['aie_freq']!=result['aie_hz'] or config['aiearch']!=result['compiled_architecture']:
        raise ValueError('clock or architecture changed after execution')
    output=root/'aiesimulator_output/out_y.txt'
    if sha256(output.read_bytes()).hexdigest()!=result['output_sha256']:
        raise ValueError('output changed after execution')
    values,complete=parse_output(output,m['contract']['spec']['m']*m['contract']['spec']['n'],m['frames'])
    if values!=sum(json.loads((root/'golden.json').read_text()),[]): raise ValueError('numerical mismatch')
    if complete!=result['frame_completion_ns']: raise ValueError('timing changed after execution')
    if len({tuple(x) for x in mapping.values()})!=result['actual_compute_tiles']:
        raise ValueError('core count changed after execution')
    result['comparison_key']=comparison_key(m,result)
    result['resources']=compiler_resources(json.loads((root/'Work/reports/compiler_report.json').read_text()))
    result['review_implementation_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    result['review_evidence']={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in
        [record,root/'manifest.json',root/'Work/reports/compiler_report.json',root/'Work/config/scsim_config.json']}
    (root/'native_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def native_run(root,platform,compile_timeout=420,simulate_timeout=180,total_timeout=None,registry=None):
    started=time.monotonic()
    root,m=check_artifact(root,registry)
    if (root/'native_result.json').exists(): raise FileExistsError('native result already exists')
    target=m['contract']['target']
    result={'schema':'aie-optimizer-native-result-v1','candidate_id':m['plan']['id'],
            'source_digest':m['source_digest'],'status':'compile_failed','evidence_kind':'native_simulation',
            'numerical':'unknown','mapping':'unknown','actual_compute_tiles':None,'latency_ns':None,
            'planned_compute_tiles':m['plan']['planned_compute_tiles'],'mode':m['plan'].get('mode','operator_lowering'),
            'row_groups':m['plan']['row_groups'], 'frames':m['frames'],
            'gather':m['plan'].get('gather','flat'),
            'metric':'simulation_origin_to_first_complete_Y_including_startup',
            'analyzer_prediction':None}
    # Hash actual platform content; comparison keys also include tool version,
    # fixtures, service contract and clocks read from compiler configuration.
    platform=Path(platform).resolve()
    if not platform.is_file(): raise ValueError('platform file missing')
    result['platform_sha256']=sha256(platform.read_bytes()).hexdigest()
    version=subprocess.run(['v++','--version'],capture_output=True,text=True,timeout=30)
    if version.returncode: raise ValueError('cannot identify compiler version')
    result['tool_version']=version.stdout.strip()
    if '2024.2' not in result['tool_version']: raise ValueError('unreviewed Vitis toolchain')
    compile_args=['v++','-c','--mode','aie','--target','hw','--platform',str(platform),
        '--include',str(root),'--work_dir',str(root/'Work'),'-o',str(root/'libadf.a'),
        '--aie.xlopt=0','--aie.stacksize=8192','--aie.keep-intermediate-results',
        '--freqhz='+str(target['plio_hz']),str(root/'TopGraph.cpp')]
    remaining=lambda limit: min(limit,max(.1,total_timeout-(time.monotonic()-started))) if total_timeout else limit
    result['compile']=command(root,compile_args,remaining(compile_timeout),'native_compile')
    try:
        if result['compile']['exit_code']!=0:
            result['diagnosis']=compile_diagnosis((root/'native_compile.log').read_text(errors='replace'))
            return result
        result['status']='mapping_failed'
        actual=mapping_report(root/'Work/reports/compiler_report.json',m['kernel_bindings'])
        result['actual_compute_tiles']=len({tuple(x) for x in actual.values()})
        result['mapping']='EXACT'; result['mapped_functions']=actual
        config=json.loads((root/'Work/config/scsim_config.json').read_text())['SimulationConfig']
        if config.get('aiearch') not in ('aie2','aie-ml'): raise ValueError('compiled target is not AIE-ML')
        validate_plios(config,m)
        result['compiled_architecture']=config['aiearch']; result['aie_hz']=config['aie_freq']
        result['status']='simulation_failed'
        if total_timeout and time.monotonic()-started>=total_timeout:
            result['status']='budget_exhausted';return result
        result['simulate']=command(root,['aiesimulator','--pkg-dir',str(root/'Work')],remaining(simulate_timeout),'native_simulate')
        if result['simulate']['exit_code']!=0: return result
        output=root/'aiesimulator_output/out_y.txt'
        expected=sum(json.loads((root/'golden.json').read_text()),[])
        count=m['contract']['spec']['m']*m['contract']['spec']['n']
        values,complete=parse_output(output,count,m['frames'])
        result['mismatches']=sum(a!=b for a,b in zip(values,expected))
        result['status']='numerical_failed'
        if result['mismatches']: return result
        check_artifact(root,registry)
        result['status']='verified';result['numerical']='EXACT'
        result['latency_ns']=complete[0];result['frame_completion_ns']=complete
        result['finite_frame_intervals_ns']=[b-a for a,b in zip(complete,complete[1:])]
        result['output_sha256']=sha256(output.read_bytes()).hexdigest()
        result['comparison_key']=comparison_key(m,result)
        result['timing_note']='includes simulator/graph startup; not comparable to historical X-FIFO-to-Y results'
        return result
    except (ValueError,KeyError,OSError) as e:
        result['error']=str(e);return result
    finally:
        (root/'native_result.json').write_text(json.dumps(result,indent=2)+'\n')
