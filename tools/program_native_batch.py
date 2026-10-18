"""Run bounded native checks for an explicit list of synthetic program plans."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Optimizer.program_backend import generate
from Optimizer.program_native import native_run


def main():
    p = argparse.ArgumentParser(); p.add_argument('cases', type=Path); p.add_argument('--platform', required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text()); results = []
    if not 1 <= len(cases) <= 4: raise ValueError('bounded native batch requires 1..4 cases')
    for case in cases:
        if set(case) != {'name', 'contract', 'plan'} or not case['name'].replace('_', '').isalnum():
            raise ValueError('explicit safe case record required')
        root = args.output / case['name']
        generate(case['contract'], case['plan'], root, frames=4)
        result = native_run(root, args.platform, compile_timeout=420, simulate_timeout=180)
        results.append({'case': case['name'], 'result': result})
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'case': case['name'], 'status': result['status'], 'latency_ns': result['latency_ns']}), flush=True)
    return int(any(r['result']['status'] != 'verified' for r in results))


if __name__ == '__main__': raise SystemExit(main())
