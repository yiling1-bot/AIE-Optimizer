"""TRAIN-only bounded GEMM-family model with explicit noncompute residual.

Compute coefficients use inclusive cycles from the same unmodified ELF.
Transport is nominal PLIO service; the remaining measured TRAIN delay is an
aggregate synchronization/startup residual, not separately identified DMA or
lock service. A held-out prediction never reads candidate native outcomes.
"""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from .contracts import identity
from .program_ir import seal, transfer_elements, live_nodes
from .program_planner import validate_plan, implementation_identity
from .program_backend import check_artifact
from .program_profile import inspect_profile
from .runner import tool_identity
from .analyzer_bridge import ClockDomain, DeclaredEvent, CausalRule, CausalParent, ClockedCausalGraph, diagnose_clocked_graph


def geometry(contract, plan):
    validate_plan(contract, plan)
    p = plan['program']; nodes = p['nodes']; products = [n for n in nodes if n['op'] == 'matmul']
    if len(plan['kernels']) != 1 or len(products) != 1 or len(p['inputs']) != 2 or len(p['outputs']) != 1:
        raise ValueError('cost family covers one-core single-GEMM signatures only')
    mm = products[0]; values = {v['id']: v for v in p['inputs'] + nodes}; a, b, init = mm['operands']
    if {a, b} != {v['id'] for v in p['inputs']} or p['outputs'][0]['value'] != mm['id']:
        raise ValueError('GEMM must cover the complete function')
    zeros = set()
    def zero(v):
        node = values[v]; zeros.add(v)
        if node.get('op') == 'constant': return set(node['attributes']['values']) == {0}
        return node.get('op') == 'splat' and zero(node['operands'][0])
    if (not zero(init) or {n['id'] for n in live_nodes(p)} != zeros | {mm['id']} or
            not zeros <= set(plan['kernels'][0]['fused']) or values[a]['type']['bits'] != 8 or
            values[b]['type']['bits'] != 8 or mm['type']['bits'] != 32 or plan['kernels'][0]['tile'] != [0, 2]):
        raise ValueError('zero-init fused i8 GEMM -> i32 at tile (0,2) required')
    m, n = mm['type']['shape']; k = values[a]['type']['shape'][1]
    return {'m': m, 'k': k, 'n': n, 'implementation': plan['lowerings'][mm['id']],
            'version': plan['implementation_versions'][plan['lowerings'][mm['id']]],
            'target': contract['target'], 'input_types': [v['type'] for v in p['inputs']], 'output_type': mm['type']}


def nominal_service(g):
    hz = g['target']['plio_hz']
    if 10**12 % hz: raise ValueError('integer picosecond source clock required')
    period = 10**12 // hz
    def service(t): return transfer_elements(t) * (t['bits'] // 8) // 16 * period
    return max(service(t) for t in g['input_types']), service(g['output_type'])


def fit_line(points):
    if len(points) != 2 or points[0][0] == points[1][0]: raise ValueError('two distinct predeclared endpoints required')
    (x0, y0), (x1, y1) = sorted(points); slope = Fraction(y1 - y0, x1 - x0); intercept = Fraction(y0) - slope*x0
    return {'slope': [slope.numerator, slope.denominator], 'intercept': [intercept.numerator, intercept.denominator]}


def line_value(line, x):
    value = Fraction(*line['slope']) * x + Fraction(*line['intercept'])
    if value < 0: raise ValueError('model predicts negative active/residual time')
    return (value.numerator + value.denominator - 1) // value.denominator


def _pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def fit_model(design, training):
    if design.get('id') != identity({k:v for k,v in design.items() if k != 'id'}): raise ValueError('frozen design identity mismatch')
    if design['implementation'] != implementation_identity(): raise ValueError('design generator changed; start a new design')
    cases = {c['name']: c for c in design['cases']}; expected = {n for n,c in cases.items() if c['role'] == 'TRAIN'}
    if set(training) != expected: raise ValueError('TRAIN set must be exact; HO or APP outcomes cannot enter model fitting')
    observations = []; pins = {}; groups = {}; contexts = set()
    target = None; m = n = None; period = None; compiled_context = None
    for name, directory in sorted(training.items()):
        root, manifest = check_artifact(directory); case = cases[name]
        if manifest['contract'] != case['contract'] or manifest['plan'] != case['plan'] or manifest['frames'] != 1:
            raise ValueError('TRAIN artifact differs from preregistered single-frame plan')
        result = json.loads((root/'native_result.json').read_text()); profile = inspect_profile(root, root/'profile_output')
        if result['status'] != 'verified' or len(profile['regions']) != 1: raise ValueError('verified native region required')
        region = profile['regions'][0]; cycles = region['inclusive_cycles']
        if region['calls'] != 1 or len(set(cycles.values())) != 1: raise ValueError('one exact elapsed region cycle observation required')
        g = geometry(manifest['contract'], manifest['plan']); tin, tout = nominal_service(g)
        hz = result['aie_hz']
        if hz <= 0 or 10**12 % int(hz): raise ValueError('integer picosecond target clock required')
        core = 10**12 // int(hz); compute = cycles['total_cycle_count'] * core
        total = round(result['latency_ns'] * 1000); residual = total - compute - tin - tout
        if residual < 0: raise ValueError('nominal transport/compute decomposition exceeds observed TRAIN time')
        context = identity({'target': g['target'], 'm': g['m'], 'n': g['n'], 'period': core,
                            'tool': tool_identity(result['tool_version']), 'platform': result['platform_sha256']})
        contexts.add(context); target, m, n, period = g['target'], g['m'], g['n'], core
        compiled_context = {'tool_identity': tool_identity(result['tool_version']), 'platform_sha256': result['platform_sha256'],
                            'aie_hz': result['aie_hz'], 'actual_compute_tiles': 1}
        groups.setdefault((g['implementation'], g['version']), []).append((g['k'], compute, residual))
        observations.append({'name': name, 'role': 'TRAIN', 'plan_id': manifest['plan']['id'],
            'geometry': g, 'compute_ps': compute, 'nominal_input_ps': tin, 'nominal_output_ps': tout,
            'aggregate_noncompute_residual_ps': residual, 'observed_total_ps': total,
            'compiled_elf_sha256': region['elf_sha256'], 'profile_id': profile['id']})
        for rel in manifest['files']:
            p = root/rel; pins[str(p.resolve())] = _pin(p)
        for rel in ['manifest.json', 'native_result.json', 'Work/config/scsim_config.json',
                    'Work/reports/compiler_report.json', 'profile_output/profile_funct_0_2.xml', 'Work/aie/0_2/Release/0_2']:
            p = root/rel; pins[str(p.resolve())] = _pin(p)
    if len(contexts) != 1: raise ValueError('TRAIN compilation/platform/geometry contexts differ')
    models = []
    endpoints = set()
    for (impl, version), rows in sorted(groups.items()):
        if len(rows) != 2: raise ValueError('each implementation needs both frozen K endpoints')
        endpoints.add(tuple(sorted(r[0] for r in rows)))
        models.append({'implementation': impl, 'version': version,
            'compute': fit_line([(k,c) for k,c,r in rows]),
            'residual': fit_line([(k,r) for k,c,r in rows])})
    if len(endpoints) != 1: raise ValueError('all implementations require the same endpoint domain')
    low, high = endpoints.pop()
    return seal({'schema': 'aie-optimizer-gemm-cost-model-v1', 'origin': 'TRAIN_calibrated_conditional_family',
        'design_id': design['id'], 'implementation': implementation_identity(),
        'compiled_context': compiled_context,
        'adapter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'domain': {'m': m, 'n': n, 'k_min': low, 'k_max': high, 'k_step': 8,
                   'target': target, 'core_period_ps': period, 'single_frame': True, 'single_core': True},
        'models': models, 'training': observations, 'evidence': list(pins.values()),
        'phase_coverage': {'compute': 'inclusive input-ready compiled-region cycles, fitted between TRAIN endpoints',
            'transport': 'nominal independent typed PLIO service; not measured DMA arbitration',
            'synchronization': 'aggregate TRAIN noncompute residual; startup/lock/release not separately identified'},
        'native_accuracy_admitted': False,
        'limitations': ['bounded one-core GEMM family only', 'interpolation is an assumption until held-out validation',
                       'no independent lock/DMA calibration', 'no cross-frame, multi-core or board admission']})


def validate_model(model):
    if (model.get('schema') != 'aie-optimizer-gemm-cost-model-v1' or
        model.get('id') != identity({k:v for k,v in model.items() if k != 'id'}) or
        model['implementation'] != implementation_identity() or
        model['adapter_sha256'] != sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('cost model identity/implementation changed')
    if (model.get('origin') != 'TRAIN_calibrated_conditional_family' or not model['training'] or
            not model['models'] or not model['evidence'] or model.get('native_accuracy_admitted') is not False):
        raise ValueError('nonempty pinned TRAIN evidence and explicit accuracy boundary required')
    if any(r['role'] != 'TRAIN' for r in model['training']): raise ValueError('non-TRAIN observation in model')
    for ref in model['evidence']:
        raw = Path(ref['path']).read_bytes()
        if len(raw) != ref['bytes'] or sha256(raw).hexdigest() != ref['sha256']: raise ValueError('TRAIN evidence changed')
    return model


def predict(model, contract, plan, *, frames=1):
    validate_model(model)
    result = {'schema': 'aie-optimizer-gemm-cost-prediction-v1', 'plan_id': plan['id'],
        'model_id': model['id'], 'status': 'needs_measurement', 'conditional_latency_ns': None,
        'latency_ns': None, 'native_admitted': False, 'phase_coverage': model['phase_coverage'], 'execution_frames': frames}
    if type(frames) is not int or frames != 1:
        return {**result, 'reason': 'TRAIN family covers single-frame execution only'}
    try: g = geometry(contract, plan)
    except ValueError as exc: return {**result, 'reason': str(exc)}
    d = model['domain']
    if (g['m'] != d['m'] or g['n'] != d['n'] or g['target'] != d['target'] or
        not d['k_min'] <= g['k'] <= d['k_max'] or g['k'] % d['k_step']):
        return {**result, 'reason': 'geometry or target outside declared TRAIN family'}
    match = [m for m in model['models'] if m['implementation'] == g['implementation'] and m['version'] == g['version']]
    if len(match) != 1: return {**result, 'reason': 'implementation has no TRAIN endpoint coverage'}
    compute = line_value(match[0]['compute'], g['k']); residual = line_value(match[0]['residual'], g['k'])
    tin, tout = nominal_service(g); core = ClockDomain(d['core_period_ps']); plio = ClockDomain(10**12//d['target']['plio_hz'])
    graph = ClockedCausalGraph((DeclaredEvent('request', 0),), (
        CausalRule('inputs_ready', plio, (CausalParent('request', 0, 'input'),), 'nominal independent input delivery'),
        CausalRule('residual_done', ClockDomain(1), (CausalParent('inputs_ready', 0, 'residual'),), 'aggregate noncompute temporal allocation'),
        CausalRule('compute_done', core, (CausalParent('residual_done', 0, 'compute'),), 'input-ready elapsed region'),
        CausalRule('outputs_done', plio, (CausalParent('compute_done', 0, 'output'),), 'nominal typed output delivery')),
        ('input','residual','compute','output'), ('model:'+model['id'], 'plan:'+plan['id']))
    costs = {'input': tin, 'residual': residual, 'compute': compute, 'output': tout}
    timing = diagnose_clocked_graph(graph, costs)
    return {**result, 'status': 'conditional_estimate', 'conditional_latency_ns': timing['times_ps']['outputs_done']/1000,
        'components_ps': costs, 'timing': timing, 'scenario_id': model['id'],
        'expected_compiled_context': model.get('compiled_context'),
        'comparison_key': identity({'contract': contract['id'], 'model': model['id'], 'metric': 'first_all_outputs_including_startup'}),
        'scope': 'TRAIN-only bounded-family interpolation with aggregate noncompute residual; no candidate observations read'}


class KernelModelEvaluator:
    def __init__(self, contract, model, frames=1): self.contract, self.model, self.frames = contract, model, frames
    def __call__(self, plan): return predict(self.model, self.contract, plan, frames=self.frames)


def freeze_predictions(design, model):
    validate_model(model)
    if design['id'] != model['design_id']: raise ValueError('model and held-out design differ')
    rows = {c['name']: predict(model,c['contract'],c['plan'],frames=c.get('frames',1)) for c in design['cases'] if c['role'] == 'HO'}
    if not rows or any(r['status'] != 'conditional_estimate' for r in rows.values()): raise ValueError('held-out predictions incomplete')
    return seal({'schema': 'aie-optimizer-prediction-lock-v1', 'design_id': design['id'], 'model_id': model['id'],
        'predictions': rows, 'outcomes_read': False, 'policy': 'freeze before HO execution; no HO fitting'})


def validate_ranking(lock, outcomes):
    if lock.get('id') != identity({k:v for k,v in lock.items() if k != 'id'}): raise ValueError('prediction lock changed')
    if set(outcomes) != set(lock['predictions']): raise ValueError('held-out outcome set differs')
    rows = []
    for name, p in lock['predictions'].items():
        actual = outcomes[name]
        if actual['status'] != 'verified' or actual['candidate_id'] != p['plan_id']: raise ValueError('verified held-out candidate mismatch')
        if 'execution_frames' in p and actual.get('frames') != p['execution_frames']:
            raise ValueError('held-out execution frame count differs from prediction scenario')
        if p.get('expected_compiled_context'):
            context = {**actual, 'tool_identity': tool_identity(actual['tool_version'])}
            if any(context.get(k) != v for k,v in p['expected_compiled_context'].items()):
                raise ValueError('held-out compiler/platform/clock context differs from TRAIN')
        obs, est = actual['latency_ns'], p['conditional_latency_ns']
        rows.append({'name': name, 'predicted_ns': est, 'observed_ns': obs,
                     'absolute_relative_error': abs(obs-est)/obs})
    order = sorted(rows,key=lambda r:r['predicted_ns']); best = min(r['observed_ns'] for r in rows)
    concordant = ties = pairs = 0
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            pairs += 1; x = a['predicted_ns']-b['predicted_ns']; y = a['observed_ns']-b['observed_ns']
            if y == 0 or x == 0: ties += 1
            elif x*y > 0: concordant += 1
    return seal({'schema': 'aie-optimizer-heldout-ranking-v1', 'prediction_lock_id': lock['id'], 'cases': rows,
        'selected': order[0]['name'], 'selected_observed_ns': order[0]['observed_ns'], 'best_observed_ns': best,
        'selection_regret': (order[0]['observed_ns']-best)/best,
        'strict_concordant_pairs': concordant, 'tied_pairs': ties, 'total_pairs': pairs,
        'all_pairs_correct': concordant+ties == pairs,
        'scope': 'one held-out K shape, three implementation candidates; no population-wide ranking reliability claim'})
