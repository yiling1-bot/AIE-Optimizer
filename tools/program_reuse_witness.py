"""Run the same public program pipeline on unrelated synthetic source graphs.

Selection covers a baseline, fusion, a semantic rewrite, and multi-core plans;
it is a correctness/coverage witness, not a performance or ranking experiment.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Optimizer.contracts import Target, identity
from Optimizer.importing.mlir_frontend import import_mlir
from Optimizer.importing.cpp_frontend import import_cpp
from Optimizer.program_ir import program_from_import
from Optimizer.program_planner import enumerate_plans
from Optimizer.program_backend import generate, cpu_check


def run(destination, frames=4, max_plans=24):
    out = Path(destination).resolve(); out.mkdir(parents=True, exist_ok=False)
    temp = out / 'temporary'; temp.mkdir(); tempfile.tempdir = str(temp)
    fixtures = ROOT / 'tests/fixtures/optimizer_program'
    cases = [('three_layer', fixtures / 'three_layer.mlir', None),
             ('fir', fixtures / 'fir.mlir', None),
             ('branched', ROOT / 'tests/fixtures/optimizer_import/branched.mlir', None),
             ('cpp_residual', fixtures / 'residual.cpp', 'residual')]
    records = []
    for name, path, entry in cases:
        artifact = import_cpp(path, entry, project_root=path.parent) if entry else import_mlir(path)
        contract = program_from_import(artifact, Target(max_compute_tiles=3))
        plans, report = enumerate_plans(contract, max_plans=max_plans, max_variants=4)
        case = out / name; case.mkdir()
        def save(filename, value): (case / filename).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        save('source_import.json', artifact); save('contract.json', contract); save('space.json', {'plans': plans, 'report': report})
        selected = []
        selectors = [('reference', lambda p: not p['rewrite_trace'] and not p['fuse'] and p['planned_compute_tiles'] == 1),
                     ('fused', lambda p: not p['rewrite_trace'] and p['fuse'] and p['matmul_order'] == 'ijk' and p['planned_compute_tiles'] == 1),
                     ('loop_order', lambda p: not p['rewrite_trace'] and p['matmul_order'] == 'ikj' and p['planned_compute_tiles'] == 1),
                     ('rewrite', lambda p: bool(p['rewrite_trace']) and p['planned_compute_tiles'] == 1 and p['fuse']),
                     ('two_core', lambda p: not p['rewrite_trace'] and p['planned_compute_tiles'] == 2),
                     ('three_core', lambda p: not p['rewrite_trace'] and p['planned_compute_tiles'] == 3)]
        for label, predicate in selectors:
            plan = next((p for p in plans if predicate(p)), None)
            if plan is None: continue
            generated = case / label; generate(contract, plan, generated, frames=frames)
            result = cpu_check(generated)
            selected.append({'label': label, 'plan_id': plan['id'], 'cores': plan['planned_compute_tiles'],
                             'trace': plan['rewrite_trace'], 'cpu': result})
            print(json.dumps({'application': name, 'label': label, 'status': result['status']}), flush=True)
            if result['status'] != 'passed': raise RuntimeError('generated CPU check failed: ' + str(generated))
        records.append({'application': name, 'import_id': artifact['id'], 'contract_id': contract['id'],
                        'legacy_backend_status': artifact['backend_admission']['status'],
                        'plans_retained': len(plans), 'report': report, 'selected': selected})
    result = {'schema': 'aie-optimizer-program-reuse-witness-v1', 'status': 'passed',
        'applications': records, 'generated_projects': sum(len(r['selected']) for r in records),
        'frames': frames, 'source_seed': 20261018,
        'output_checks': sum(p['cpu']['outputs_checked'] for r in records for p in r['selected']),
        'native_performance_claim': False, 'scope': 'same importer adapter, rule engine, planner, generator and checker; synthetic software witnesses'}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=4); parser.add_argument('--max-plans', type=int, default=24)
    args = parser.parse_args(); result = run(args.output, args.frames, args.max_plans)
    print(json.dumps({k: v for k, v in result.items() if k != 'applications'}))
