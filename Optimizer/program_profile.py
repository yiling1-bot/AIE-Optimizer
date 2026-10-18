"""Pinned in-context region cycle evidence from the actual unmodified ELF.

Only single-frame, single-core records are admitted to the first cost family.
Elapsed function cycles include the compiled region's own memory effects; they
are not relabelled as stall-free instruction counts or independent DMA costs.
"""
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .program_backend import check_artifact
from .program_native import parse_typed_output
from .runner import command
from .program_ir import seal


def inspect_profile(root, directory):
    root, manifest = check_artifact(root); directory = Path(directory)
    native = json.loads((root / 'native_result.json').read_text())
    if native['status'] != 'verified' or native['source_digest'] != manifest['source_digest']:
        raise ValueError('verified source-matched native run required')
    golden = json.loads((root / 'golden.json').read_text()); completions = []
    for i, output in enumerate(manifest['plan']['program']['outputs']):
        values, times = parse_typed_output(directory / f'out_{i}.txt', output['type'], manifest['frames'])
        if values != [v for frame in golden for v in frame[i]]: raise ValueError('profile run output mismatch')
        completions.append(times)
    complete = [max(t[f] for t in completions) for f in range(manifest['frames'])]
    if complete != native['frame_completion_ns']: raise ValueError('profiling changed target output timing')
    regions = []; pins = {}
    for kernel in manifest['plan']['kernels']:
        col, row = kernel['tile']; xml = directory / f'profile_funct_{col}_{row}.xml'
        raw = xml.read_bytes(); doc = ET.fromstring(raw)
        matching = [f for f in doc.findall('function') if f'compute_{kernel["id"]}' in (f.findtext('function_name') or '')]
        if len(matching) != 1: raise ValueError('unique outlined compute region required')
        f = matching[0]; calls = int(f.findtext('calls'))
        if calls != manifest['frames']: raise ValueError('region call count differs from frames')
        times = {v.tag: int(v.text) for v in f.find('function_and_descendants_time')}
        if times['minimum_cycle_count'] < 0 or times['maximum_cycle_count'] < times['minimum_cycle_count']:
            raise ValueError('invalid inclusive cycle interval')
        elf = root / f'Work/aie/{col}_{row}/Release/{col}_{row}'
        binary = elf.read_bytes()
        if not binary.startswith(b'\x7fELF'): raise ValueError('linked ELF required')
        pins[xml.name] = sha256(raw).hexdigest()
        regions.append({'kernel': kernel['id'], 'function': f.findtext('function_name'), 'calls': calls,
            'inclusive_cycles': times, 'elf_sha256': sha256(binary).hexdigest(),
            'elapsed_cycles_are_stall_free': False})
    return seal({'schema': 'aie-optimizer-program-profile-v1', 'source_digest': manifest['source_digest'],
        'plan_id': manifest['plan']['id'], 'frames': manifest['frames'], 'regions': regions,
        'profile_sha256': pins, 'matches_native_timing': True,
        'scope': 'observed inclusive elapsed region cycles in the pinned compiled context; no self/child double count'})


def capture_profile(root, timeout=180):
    root = Path(root).resolve(); output = root / 'profile_output'
    if output.exists(): raise FileExistsError('profile output already exists')
    run = command(root, ['aiesimulator', '--pkg-dir', str(root / 'Work'), '--input-dir', str(root),
                         '--output-dir', str(output), '--profile'], timeout, 'profile_simulate')
    result = {'run': run, 'status': 'failed'}
    if run['exit_code'] == 0:
        try: result.update(status='verified', evidence=inspect_profile(root, output))
        except (ValueError, OSError, KeyError) as exc: result['error'] = str(exc)
    (root / 'profile_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
