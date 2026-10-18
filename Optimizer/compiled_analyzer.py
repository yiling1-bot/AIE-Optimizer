"""Isolated access to a pinned, external AIE1 Analyzer implementation.

This evaluates compiled candidates; it does not relabel an AIE-ML plan as AIE1
or prove equivalence between unrelated compiled programs. Comparisons initially
cover one compiled program with exact inputs, or explicitly permitted dense
packet interleavings preserving each destination's complete input sequence.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

from .contracts import identity

SNAPSHOT = 'aie-optimizer-analyzer-snapshot-v1'
CANDIDATE = 'aie-optimizer-compiled-candidate-v1'
POLICIES = ('exact_inputs', 'dense_packet_interleaving')


def seal(value):
    return {**value, 'id': identity(value)}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value): raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def save_new(path, value, *, compact=False):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=None if compact else 2,
                  separators=(',',':') if compact else None, allow_nan=False)
        stream.write('\n')
    return file_ref(path)


def file_ref(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def read_ref(ref):
    if (type(ref) is not dict or set(ref) != {'path', 'bytes', 'sha256'} or
            type(ref['path']) is not str or not Path(ref['path']).is_absolute() or
            type(ref['bytes']) is not int or not 0 <= ref['bytes'] <= 512 * 1024**2 or
            type(ref['sha256']) is not str or len(ref['sha256']) != 64):
        raise ValueError('absolute path, byte count and SHA256 reference required')
    raw = Path(ref['path']).read_bytes()
    if len(raw) != ref['bytes'] or sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('pinned artifact changed: ' + ref['path'])
    return raw


def validate_snapshot(snapshot):
    if (snapshot.get('schema') != SNAPSHOT or
            snapshot.get('id') != identity({k:v for k,v in snapshot.items() if k != 'id'}) or
            snapshot.get('supported_request_schema') != 'aie-generic-analyzer-request-v1' or
            snapshot.get('architecture') != 'AIE1' or not snapshot.get('files')):
        raise ValueError('recognized sealed AIE1 Analyzer snapshot required')
    if not Path(snapshot['root']).is_absolute(): raise ValueError('absolute Analyzer snapshot root required')
    root = Path(snapshot['root']).resolve()
    for name, digest in snapshot['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root / 'Model') or path.suffix != '.py':
            raise ValueError('unsafe Analyzer source path')
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Analyzer snapshot source changed: ' + name)
    actual = {p.relative_to(root).as_posix() for p in (root/'Model').rglob('*.py')}
    if actual != set(snapshot['files']): raise ValueError('Analyzer snapshot module inventory changed')
    return snapshot


def adapter_identity():
    names = ('compiled_analyzer.py','compiled_analyzer_worker.py','contracts.py')
    return {name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def capture_analyzer(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source/'Model'):
        raise ValueError('fresh output outside the source Model tree required')
    files = {p.relative_to(source).as_posix(): p.read_bytes() for p in sorted((source/'Model').rglob('*.py'))}
    if 'Model/Timing/generic_analyzer.py' not in files:
        raise ValueError('source does not provide Model.Timing.generic_analyzer')
    destination.mkdir(parents=True)
    for name, raw in files.items():
        path = destination/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    if set(files) != {p.relative_to(source).as_posix() for p in (source/'Model').rglob('*.py')}:
        raise ValueError('source module inventory changed during capture')
    for name, raw in files.items():
        if (source/name).read_bytes() != raw: raise ValueError('Analyzer source changed during capture')
    snapshot = seal({'schema': SNAPSHOT, 'root': str(destination), 'source_root': str(source),
        'files': {name:sha256(raw).hexdigest() for name,raw in files.items()},
        'architecture': 'AIE1', 'supported_request_schema': 'aie-generic-analyzer-request-v1',
        'scope': 'frozen Python implementation; executable/cost evidence remains independently pinned by each request'})
    validate_snapshot(snapshot); save_new(destination/'snapshot.json', snapshot)
    return snapshot


def make_candidate(request, *, plan_id=None, label='', policy='exact_inputs'):
    if policy not in POLICIES: raise ValueError('unknown input comparison policy')
    ref = file_ref(request) if not isinstance(request, dict) else request
    value = strict_json(read_ref(ref))
    if (value.get('schema') != 'aie-generic-analyzer-request-v1' or
            value.get('mode') not in ('costs', 'predict')):
        raise ValueError('normal AIE1 costs/predict request required')
    fields = {'schema','mode','compiler_inputs','program_binding','cost_registry'}
    if value['mode'] == 'predict': fields |= {'source_csv','model_selection','budgets'}
    if set(value) != fields: raise ValueError('exact normal request fields required; observations forbidden')
    if plan_id is None: plan_id = 'compiled:' + identity(ref)
    if type(plan_id) is not str or not plan_id or type(label) is not str:
        raise ValueError('nonempty plan ID and string label required')
    return seal({'schema': CANDIDATE, 'plan_id': plan_id, 'label': label,
                 'request_ref': ref, 'architecture': 'AIE1', 'comparison_policy': policy})


def validate_candidate(candidate):
    if (set(candidate) != {'schema','plan_id','label','request_ref','architecture','comparison_policy','id'} or
            candidate.get('schema') != CANDIDATE or candidate.get('architecture') != 'AIE1' or
            candidate.get('id') != identity({k:v for k,v in candidate.items() if k != 'id'})):
        raise ValueError('sealed compiled candidate required')
    expected = make_candidate(candidate['request_ref'], plan_id=candidate['plan_id'],
                              label=candidate['label'], policy=candidate['comparison_policy'])
    if expected != candidate: raise ValueError('compiled candidate replay differs')
    return candidate


def normalize(candidate, raw, facts, backend_id):
    """Keep conditional model time separate from measured/native time."""
    graph = raw.get('graph', {}); resolution = raw.get('component_costs', {})
    components = resolution.get('components', []); hardware = raw.get('hardware_component_costs', [])
    result = {'schema': 'aie-optimizer-analyzer-evaluation-v1', 'candidate_id': candidate['id'],
        'plan_id': candidate['plan_id'], 'architecture': graph.get('architecture'),
        'backend_id': backend_id, 'status': raw.get('status', 'invalid_input'),
        'conditional_latency_ns': None, 'conditional_frame_completion_ns': [], 'latency_ns': None,
        'native_admitted': False, 'native_accuracy_admitted': False, 'ranking_accuracy_admitted': False,
        'actual_compute_tiles': graph.get('counts', {}).get('compute_tiles'),
        'comparison_key': facts.get('comparison_key'),
        'scenario_id': identity({'backend':backend_id,'cost_scenario':facts['scenario_id']}) if facts.get('scenario_id') else None,
        'comparison_scope': 'same compiled program; cross-compile semantic binding not yet admitted',
        'coverage': {'components': len(components), 'statuses': dict(Counter(c['status'] for c in components)),
                     'TRAIN_available': sum(bool(c.get('TRAIN_cost_ids')) for c in hardware),
                     'needs_measurement': [c['component_id'] for c in hardware if not c.get('TRAIN_cost_ids')]},
        'unsupported_resources': raw.get('unsupported_resources', []),
        'reason': raw.get('reason') or raw.get('prediction',{}).get('reason'),
        'analyzer_timing_prediction': raw.get('timing_prediction'),
        'analyzer_status': raw.get('status'), 'input_refs': raw.get('input_refs', [])}
    if raw.get('Simulator_timestamps_used_as_inputs') is not False:
        raise ValueError('Analyzer did not confirm observation-free prediction')
    if graph.get('architecture') != 'AIE1':
        return {**result, 'status':'unsupported', 'reason':'AIE1 evaluator cannot admit another architecture'}
    if raw.get('status') == 'reused': return {**result, 'status':'costs_resolved'}
    if raw.get('status') != 'conditional_prediction': return result
    if raw.get('software_contract_prediction_only') is not False:
        return {**result, 'status':'software_conditional_prediction', 'reason':'software fixture is not native TRAIN coverage'}
    if not hardware or not all(c.get('TRAIN_cost_ids') and c.get('executable_hardware_binding_admitted') is True for c in hardware):
        raise ValueError('complete TRAIN costs and executable binding required for native-context estimates')
    timing = raw.get('timing_prediction'); prediction = raw.get('prediction', {})
    if (not timing or timing.get('metric') != 'global_origin0_to_last_valid_SDK_FILE' or
            type(timing.get('value_ps')) is not int or timing['value_ps'] < 0):
        raise ValueError('recognized integer-picosecond Analyzer metric required')
    frames = prediction.get('frames', []); by_call = {}; seen = set(); ports = set()
    for route in graph.get('routes', []):
        if route.get('PLIO') and any(p['id'] == route['window_port_id'] and p['direction'] == 'output'
                                     for p in graph.get('ports', [])):
            ports.add(route['window_port_id'])
    for row in frames:
        call, value = row['call_ordinal'], row['FILE_last_ps']
        if type(call) is not int or call < 0 or type(value) is not int or value < 0:
            raise ValueError('invalid frame completion record')
        key = (row['port_id'], call)
        if key in seen or row['port_id'] not in ports: raise ValueError('duplicate or unknown frame output port')
        seen.add(key); by_call[call] = max(by_call.get(call, 0), value)
    if (not ports or not by_call or sorted(by_call) != list(range(len(by_call))) or
            seen != {(p,c) for p in ports for c in by_call} or max(by_call.values()) != timing['value_ps']):
        raise ValueError('complete contiguous frame records must match the Analyzer endpoint')
    complete = [by_call[i]/1000 for i in range(len(by_call))]
    if complete != sorted(complete) or not result['comparison_key'] or not result['scenario_id']:
        raise ValueError('ordered frame times and explicit comparison/scenario required')
    return {**result, 'status':'conditional_estimate', 'conditional_latency_ns':complete[0],
            'conditional_frame_completion_ns':complete,
            'scope':'first complete frame and complete finite batch; actual compiled AIE1 model, no accuracy promotion'}


def prepare_compiled_candidate(snapshot, extraction, preparation, output, *, plan_id=None,
                               policy='exact_inputs', timeout=300):
    """Automate normal extraction/preparation over existing compiler artifacts."""
    snapshot = deepcopy(validate_snapshot(snapshot)); adapter = adapter_identity()
    if type(timeout) not in (int,float) or not 0 < timeout <= 3600: raise ValueError('finite bounded timeout required')
    extraction_ref = file_ref(extraction); preparation_ref = file_ref(preparation)
    template = strict_json(read_ref(preparation_ref)); specification = strict_json(read_ref(extraction_ref))
    if template.get('metadata_ref') is not None or template.get('compiler_inputs') != specification.get('compiler_inputs'):
        raise ValueError('preparation template needs null metadata and the identical compiler input reference')
    output = Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    job = {'schema':'aie-optimizer-analyzer-job-v1','operation':'prepare','snapshot':snapshot,'output':str(output),
           'extraction_ref':extraction_ref,'preparation_ref':preparation_ref}
    job_ref = save_new(output/'job.json',job); worker = Path(__file__).with_name('compiled_analyzer_worker.py')
    worker_sha = sha256(worker.read_bytes()).hexdigest()
    with (output/'worker.log').open('x',encoding='utf-8') as log:
        try:
            run = subprocess.run([sys.executable,'-I',str(worker),job_ref['path']],stdout=log,stderr=subprocess.STDOUT,
                timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except subprocess.TimeoutExpired as error: raise ValueError('compiled preparation worker timeout; see ' + str(output/'worker.log')) from error
    validate_snapshot(snapshot);read_ref(job_ref);read_ref(extraction_ref);read_ref(preparation_ref)
    if adapter_identity() != adapter or sha256(worker.read_bytes()).hexdigest() != worker_sha:
        raise ValueError('adapter source changed during preparation')
    if run.returncode or not (output/'preparation.json').is_file():
        raise ValueError('compiled preparation failed; see ' + str(output/'worker.log'))
    receipt = strict_json((output/'preparation.json').read_bytes())
    ref = receipt['prepared'].get('predict_request_ref') or receipt['prepared']['costs_request_ref']
    candidate = make_candidate(ref,plan_id=plan_id,label='prepared compiled candidate',policy=policy)
    save_new(output/'candidate.json',candidate)
    report = seal({'schema':'aie-optimizer-compiled-preparation-v1','backend_id':snapshot['id'],
        'candidate':candidate,'receipt':receipt,'adapter_identity':adapter,
        'new_compilations':0,'new_Simulator_runs':0,'new_cost_fits':0})
    save_new(output/'prepared-candidate.json',report)
    return report


def _invalid(candidate, reason):
    return {'schema':'aie-optimizer-analyzer-evaluation-v1','candidate_id':candidate['id'],
            'plan_id':candidate['plan_id'],'status':'invalid_input','reason':reason,
            'conditional_latency_ns':None,'latency_ns':None,'native_admitted':False}


def evaluate_candidates(candidates, snapshot, output, *, timeout=300, reuse_prepared=True):
    snapshot = deepcopy(validate_snapshot(snapshot)); candidates = deepcopy(list(candidates)); adapter = adapter_identity()
    if not 1 <= len(candidates) <= 32 or type(timeout) not in (int,float) or not 0 < timeout <= 3600:
        raise ValueError('1..32 candidates and finite timeout <=3600 seconds required')
    for candidate in candidates: validate_candidate(candidate)
    if len({c['id'] for c in candidates}) != len(candidates): raise ValueError('duplicate candidate identity')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    job = {'schema':'aie-optimizer-analyzer-job-v1', 'snapshot':snapshot, 'candidates':candidates,
           'output':str(output), 'reuse_prepared':bool(reuse_prepared)}
    job_ref = save_new(output/'job.json', job); worker = Path(__file__).with_name('compiled_analyzer_worker.py')
    worker_sha = sha256(worker.read_bytes()).hexdigest(); started = time.perf_counter(); timed_out = False
    with (output/'worker.log').open('x',encoding='utf-8') as log:
        try:
            run = subprocess.run([sys.executable,'-I',str(worker),job_ref['path']], stdout=log, stderr=subprocess.STDOUT,
                                 timeout=timeout, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            exit_code = run.returncode
        except subprocess.TimeoutExpired:
            timed_out = True; exit_code = None
    validate_snapshot(snapshot)
    if adapter_identity() != adapter or sha256(worker.read_bytes()).hexdigest() != worker_sha:
        raise ValueError('adapter source changed during evaluation')
    read_ref(job_ref); evaluations = []
    for i,candidate in enumerate(candidates):
        receipt = output/f'query-{i:03d}.json'
        if receipt.exists():
            record = strict_json(receipt.read_bytes())
            if record.get('candidate_id') != candidate['id']: raise ValueError('worker receipt candidate mismatch')
            if record['status'] == 'completed':
                try:
                    raw = strict_json(read_ref(record['result_ref']))
                    for ref in raw.get('input_refs', []): read_ref(ref)
                    result = normalize(candidate, raw, record['comparison_facts'], snapshot['id'])
                    result.update(raw_result_ref=record['result_ref'], seconds=record['seconds'],
                                  query_reused=record.get('query_reused',False), prepared_reused=record.get('prepared_reused',False))
                except (ValueError,TypeError,KeyError,OSError) as error:
                    result = _invalid(candidate,str(error))
            else:
                result = _invalid(candidate,record['reason'])
        else:
            result = {'schema':'aie-optimizer-analyzer-evaluation-v1','candidate_id':candidate['id'],
                'plan_id':candidate['plan_id'],'status':'refused','reason':'worker timeout' if timed_out else 'worker did not complete',
                'conditional_latency_ns':None,'latency_ns':None,'native_admitted':False}
        evaluations.append(result)
    report = seal({'schema':'aie-optimizer-analyzer-batch-v1','backend_id':snapshot['id'],
        'evaluations':evaluations,'worker_exit_code':exit_code,'timed_out':timed_out,
        'seconds':time.perf_counter()-started,'worker_sha256':worker_sha,'adapter_identity':adapter,
        'native_accuracy_admitted':False,'new_Simulator_runs':0,'new_cost_fits':0})
    save_new(output/'evaluations.json',report)
    return report


class AnalyzerEvaluator:
    """Plan callback for a registry of already bound compiled candidates."""
    def __init__(self, snapshot, bindings, output, *, architecture='AIE1', timeout=300):
        self.snapshot, self.bindings, self.output = snapshot, bindings, Path(output)
        self.architecture, self.timeout = architecture, timeout
        self.output.mkdir(parents=True,exist_ok=False); self.calls = 0

    def __call__(self, plan):
        base = {'plan_id':plan['id'],'conditional_latency_ns':None,'latency_ns':None,'native_admitted':False}
        if self.architecture != 'AIE1': return {**base,'status':'unsupported','reason':'AIE1 compiled evaluator target mismatch'}
        candidate = self.bindings.get(plan['id'])
        if candidate is None: return {**base,'status':'needs_analyzer_binding'}
        if candidate['plan_id'] != plan['id']: raise ValueError('compiled binding belongs to another plan')
        destination = self.output/f'evaluation-{self.calls:03d}'; self.calls += 1
        result = evaluate_candidates([candidate],self.snapshot,destination,timeout=self.timeout)['evaluations'][0]
        if result.get('actual_compute_tiles') is not None and result['actual_compute_tiles'] != plan['planned_compute_tiles']:
            raise ValueError('compiled core count differs from requested plan')
        return result
