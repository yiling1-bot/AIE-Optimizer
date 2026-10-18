"""Bounded native validation for typed, multi-output program artifacts."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
import time

from .contracts import identity
from .program_backend import check_artifact
from .program_ir import elements, transfer_elements
from .runner import command, compile_diagnosis, mapping_report, tool_identity


def parse_typed_output(path, typ, frames):
    data, times, current = [], [], None
    units = {'ps': .001, 'ns': 1., 'us': 1000., 'ms': 1_000_000.}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or line == 'TLAST': continue
        match = re.fullmatch(r'T\s+([0-9.eE+-]+)\s+(ps|ns|us|ms)', line)
        if match:
            value = float(match[1]) * units[match[2]]
            if not isfinite(value) or value < 0 or (current is not None and value < current):
                raise ValueError('invalid native timestamp')
            current = value; continue
        fields = line.split()
        if current is None or any(re.fullmatch(r'[-+]?\d+', v) is None for v in fields):
            raise ValueError('native data without valid timestamp')
        vals = list(map(int, fields)); bits = typ['bits']
        if any(not -(1 << (bits - 1)) <= v < (1 << (bits - 1)) for v in vals):
            raise ValueError('native output outside declared signed type')
        data.extend(vals); times.extend([current] * len(vals))
    count = transfer_elements(typ)
    if len(data) != count * frames: raise ValueError('incomplete/extra typed native output')
    return data, [times[(f + 1) * count - 1] for f in range(frames)]


def validate_plios(config, manifest):
    program = manifest['plan']['program']; expected = {}
    for direction, values in (('input', program['inputs']), ('output', program['outputs'])):
        for i, v in enumerate(values):
            name = ('I' if direction == 'input' else 'O') + str(i)
            expected[name] = (direction, v['type'], f'data/I{i}.txt' if direction == 'input' else f'out_{i}.txt')
    blocks = {b['name']: b for b in config['pl_ip_block'] if b.get('axi_stream')}
    if set(blocks) != set(expected): raise ValueError('typed program PLIO signature changed')
    for name, (direction, typ, filename) in expected.items():
        b = blocks[name]; flow = 'pl_to_me' if direction == 'input' else 'me_to_pl'
        if (b['pl_freq'] != manifest['contract']['target']['plio_hz'] or b['arguments'][0] != filename or
                b['arguments'][-1] != f'int{typ["bits"]}' or
                any(s['direction'] != flow or s['bus_width'] != 128 for s in b['axi_stream'])):
            raise ValueError('typed program compiled PLIO service mismatch')


def native_run(root, platform, compile_timeout=420, simulate_timeout=180):
    root, manifest = check_artifact(root)
    if (root / 'native_result.json').exists(): raise FileExistsError('native program result already exists')
    platform = Path(platform).resolve()
    if not platform.is_file(): raise ValueError('platform file missing')
    version = subprocess.run(['v++', '--version'], capture_output=True, text=True, timeout=30)
    if version.returncode or tool_identity(version.stdout)['version'] != '2024.2':
        raise ValueError('Vitis 2024.2 required')
    result = {'schema': 'aie-optimizer-program-native-result-v1', 'status': 'compile_failed',
        'candidate_id': manifest['plan']['id'], 'source_digest': manifest['source_digest'],
        'evidence_kind': 'native_simulation', 'tool_version': version.stdout.strip(),
        'platform_sha256': sha256(platform.read_bytes()).hexdigest(), 'latency_ns': None,
        'analyzer_prediction': None, 'actual_compute_tiles': None, 'frames': manifest['frames'],
        'metric': 'simulation_origin_to_first_all_complete_outputs_including_startup'}
    args = ['v++', '-c', '--mode', 'aie', '--target', 'hw', '--platform', str(platform),
        '--include', str(root), '--work_dir', str(root / 'Work'), '-o', str(root / 'libadf.a'),
        '--aie.xlopt=0', '--aie.keep-intermediate-results',
        '--freqhz=' + str(manifest['contract']['target']['plio_hz']), str(root / 'TopGraph.cpp')]
    result['compile'] = command(root, args, compile_timeout, 'native_compile')
    try:
        if result['compile']['exit_code'] != 0:
            result['diagnosis'] = compile_diagnosis((root / 'native_compile.log').read_text(errors='replace'))
            return result
        result['status'] = 'mapping_failed'
        actual = mapping_report(root / 'Work/reports/compiler_report.json', manifest['kernel_bindings'])
        config = json.loads((root / 'Work/config/scsim_config.json').read_text())['SimulationConfig']
        validate_plios(config, manifest)
        if config.get('aiearch') not in ('aie2', 'aie-ml'): raise ValueError('target is not AIE-ML')
        result.update(mapped_functions=actual, actual_compute_tiles=len({tuple(v) for v in actual.values()}),
                      aie_hz=config['aie_freq'], compiled_architecture=config['aiearch'], status='simulation_failed')
        result['simulate'] = command(root, ['aiesimulator', '--pkg-dir', str(root / 'Work')], simulate_timeout, 'native_simulate')
        if result['simulate']['exit_code'] != 0: return result
        result['status'] = 'numerical_failed'; golden = json.loads((root / 'golden.json').read_text())
        completions, evidence = [], []
        for i, out in enumerate(manifest['plan']['program']['outputs']):
            path = root / f'aiesimulator_output/out_{i}.txt'
            values, complete = parse_typed_output(path, out['type'], manifest['frames'])
            if values != [v for frame in golden for v in frame[i]]: raise ValueError('native output mismatch')
            completions.append(complete)
            evidence.append({'output': i, 'sha256': sha256(path.read_bytes()).hexdigest(), 'frame_completion_ns': complete})
        check_artifact(root)
        complete = [max(v[f] for v in completions) for f in range(manifest['frames'])]
        result.update(status='verified', numerical='EXACT', outputs=evidence, latency_ns=complete[0],
            frame_completion_ns=complete, finite_frame_intervals_ns=[b - a for a, b in zip(complete, complete[1:])],
            logical_outputs_checked=sum(elements(o['type']) for o in manifest['plan']['program']['outputs']) * manifest['frames'])
        result['comparison_key'] = identity({'contract': manifest['contract']['id'], 'frames': manifest['frames'],
            'fixtures': {k: v for k, v in manifest['files'].items() if k.startswith('data/')},
            'tool': tool_identity(result['tool_version']), 'platform': result['platform_sha256'],
            'aie_hz': result['aie_hz'], 'metric': result['metric']})
        return result
    except (ValueError, KeyError, OSError) as exc:
        result['error'] = str(exc); return result
    finally:
        (root / 'native_result.json').write_text(json.dumps(result, indent=2) + '\n')
