"""Select an enumerated plan for generation; the index is not a latency rank."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('space', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    plans = json.loads(args.space.read_text(encoding='utf-8'))['plans']
    if not 0 <= args.index < len(plans):
        parser.error(f'index must be in [0, {len(plans)})')
    plan = plans[args.index]
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'plan_id': plan['id'], 'index': args.index,
                      'status': 'selected_for_demonstration'}))


if __name__ == '__main__':
    main()
