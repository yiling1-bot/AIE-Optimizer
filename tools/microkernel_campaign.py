"""Frozen synthetic design and bounded native/profile batches for phase 12.

TRAIN and HO roles are explicit. HO execution requires a pinned prediction
file created before its outcomes are read. No model fitting runs on the server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Optimizer.contracts import Target, identity
from Optimizer.importing.ir import GraphBuilder, tensor_type
from Optimizer.program_ir import make_program, program_from_builder, program_from_import, seal
from Optimizer.program_planner import make_plan, implementation_identity
from Optimizer.program_backend import generate
from Optimizer.program_native import native_run
from Optimizer.program_profile import capture_profile


def make_design(output):
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    cases = []
    implementations = ['scalar.matmul.ijk', 'aie_ml.mmul.i8.grid1x1', 'aie_ml.mmul.i8.grid1x2']
    for k, role in ((8, 'TRAIN'), (32, 'TRAIN'), (16, 'HO')):
        b = GraphBuilder(); a = b.input('samples', tensor_type([8, k], 8), 0)
        w = b.input('weights', tensor_type([k, 16], 8), 1)
        init = b.add('constant', [], tensor_type([8, 16], 32), {'values': [0]}, {})
        out = b.add('matmul', [a, w, init], tensor_type([8, 16], 32), {}, {})
        contract = make_program(program_from_builder(b, [out]), Target(max_compute_tiles=1))
        for index, impl in enumerate(implementations):
            plan = make_plan(contract, fuse=True, lowerings={out: impl})
            cases.append({'name': f'gemm_k{k}_i{index}', 'role': role, 'frames': 1, 'profile': True,
                          'contract': contract, 'plan': plan})
    from Optimizer.importing.mlir_frontend import import_mlir
    contract = program_from_import(import_mlir(ROOT / 'tests/fixtures/optimizer_program/three_layer.mlir'), Target(max_compute_tiles=1))
    for index, impl in enumerate(implementations[:2]):
        choices = {n['id']: impl for n in contract['program']['nodes'] if n['op'] == 'matmul'}
        cases.append({'name': f'three_layer_i{index}', 'role': 'APP', 'frames': 4, 'profile': False,
                      'contract': contract, 'plan': make_plan(contract, fuse=True, lowerings=choices)})
    rng = random.Random(20261018)
    training = [c for c in cases if c['role'] == 'TRAIN']; rng.shuffle(training)
    applications = [c for c in cases if c['role'] == 'APP']; rng.shuffle(applications)
    holdout = [c for c in cases if c['role'] == 'HO']; rng.shuffle(holdout)
    batches = [training[:4], training[4:] + applications, holdout]
    design = seal({'schema': 'aie-optimizer-microkernel-design-v1', 'seed': 20261018,
        'implementation': implementation_identity(), 'cases': cases,
        'batches': [[c['name'] for c in group] for group in batches],
        'question': 'same-core microkernel benefit and bounded GEMM-family held-out ranking',
        'experimental_unit': 'one compiled candidate run; frames are correctness/throughput observations, not replicates',
        'controls': 'AIE-ML 1.25GHz, PLIO128 200MHz, same source/fixtures per shape, core (0,2), Vitis2024.2',
        'model_domain': 'one-core zero-init GEMM M8,N16,K8..32 step8; no multi-kernel generalization',
        'prediction_policy': 'fit TRAIN only; freeze predictions before HO; no HO-driven coefficient changes',
        'replication': 'one build/run per candidate; one profile replay of the same ELF, not an independent replicate'})
    (root / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    for i, group in enumerate(batches): (root / f'batch{i}.json').write_text(json.dumps(group, indent=2) + '\n')
    return design


def validate_batch_predictions(cases, lock):
    if (lock.get('schema') != 'aie-optimizer-prediction-lock-v1' or lock.get('outcomes_read') is not False or
            lock.get('id') != identity({k:v for k,v in lock.items() if k != 'id'})):
        raise ValueError('sealed outcome-free prediction lock required')
    if set(lock['predictions']) != {c['name'] for c in cases}: raise ValueError('prediction set differs from held-out cases')
    for case in cases:
        p = lock['predictions'][case['name']]; time_ns = p.get('conditional_latency_ns')
        if (case['role'] != 'HO' or p.get('plan_id') != case['plan']['id'] or
                p.get('model_id') != lock['model_id'] or p.get('execution_frames') != case['frames'] or
                p.get('status') != 'conditional_estimate' or p.get('native_admitted') is not False or
                not isinstance(time_ns, (int, float)) or isinstance(time_ns, bool) or
                not isfinite(time_ns) or time_ns <= 0):
            raise ValueError('held-out plan/model/scenario differs from frozen prediction')


def run_batch(cases, destination, platform, jobs=2, prediction_lock=None):
    if not 1 <= len(cases) <= 4 or jobs not in (1, 2): raise ValueError('bounded batch requires <=4 cases and <=2 workers')
    if any(c['role'] == 'HO' for c in cases):
        if prediction_lock is None: raise ValueError('held-out execution requires frozen prediction evidence')
        validate_batch_predictions(cases, json.loads(Path(prediction_lock).read_text()))
    root = Path(destination); root.mkdir(parents=True, exist_ok=False)
    def run(case):
        start = time.time(); path = root / case['name']
        print(json.dumps({'start': case['name'], 'role': case['role']}), flush=True)
        generate(case['contract'], case['plan'], path, frames=case['frames'])
        native = native_run(path, platform, compile_timeout=420, simulate_timeout=180)
        profile = capture_profile(path, 180) if native['status'] == 'verified' and case['profile'] else None
        result = {'name': case['name'], 'role': case['role'], 'started_unix': start, 'finished_unix': time.time(),
                  'native': native, 'profile': profile}
        print(json.dumps({'finish': case['name'], 'status': native['status'], 'latency_ns': native['latency_ns'],
                          'profile': profile['status'] if profile else None}), flush=True)
        return result
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run, c): c for c in cases}
        for f in as_completed(futures):
            try: result = f.result()
            except Exception as exc: result = {'name': futures[f]['name'], 'role': futures[f]['role'], 'error': str(exc)}
            results.append(result); (root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('design'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('cases', type=Path); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--platform', required=True); p.add_argument('--jobs', type=int, default=2); p.add_argument('--prediction-lock', type=Path)
    args = parser.parse_args()
    if args.command == 'design':
        result = make_design(args.output); print(json.dumps({'design_id': result['id'], 'cases': len(result['cases'])})); return 0
    rows = run_batch(json.loads(args.cases.read_text()), args.output, args.platform, args.jobs, args.prediction_lock)
    return int(any('error' in r or r['native']['status'] != 'verified' or
                   (r['profile'] and r['profile']['status'] != 'verified') for r in rows))


if __name__ == '__main__': raise SystemExit(main())
