"""Operator-compositional ADF/C++ lowering for complete static tensor graphs.

This portable scalar baseline supports the importer DAG, not a two-layer
template. It preserves modular arithmetic explicitly without signed C++ UB.
Operator implementation choices are guarded and replayed through a registry.
"""
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import random

from .contracts import identity
from .importing.ir import bounds, evaluate
from .program_ir import elements, transfer_elements
from .program_planner import validate_plan


def _ctype(typ): return 'int' + str(typ['bits']) + '_t'


def _coords(index, shape):
    return [f'(({index})/{prod(shape[d+1:])}%{size})' for d, size in enumerate(shape)]


def _offset(coords, shape):
    return '(' + '+'.join(f'({v})*{prod(shape[d+1:])}' for d, v in enumerate(coords)) + ')' if shape else '0'


def emit_compute(plan, implementation_registry=None):
    from .program_kernels import default_implementations
    implementations = implementation_registry or default_implementations()
    program = plan['program']; values = {v['id']: v for v in program['inputs'] + program['nodes']}
    native = any(implementations.entries[name].uses_aie_api for name in plan['lowerings'].values())
    header = ['#pragma once', '#include <cstdint>']
    if native: header += ['#ifndef AIE_OPT_HOST', '#include <aie_api/aie.hpp>', '#endif']
    header += ['namespace opt_program {',
        'inline int64_t wrap_u(uint64_t v,unsigned bits) {',
        ' uint64_t mask=(uint64_t(1)<<bits)-1; v &= mask;',
        ' return (v & (uint64_t(1)<<(bits-1))) ? int64_t(v)-int64_t(uint64_t(1)<<bits) : int64_t(v); }',
        'inline int64_t wrap(int64_t v,unsigned bits) {return wrap_u(uint64_t(v),bits);}',
        'inline int64_t maximum(int64_t a,int64_t b) {return a>b?a:b;}',
        'inline int64_t minimum(int64_t a,int64_t b) {return a<b?a:b;}']
    for node in program['nodes']:
        if node['op'] == 'constant' and len(node['attributes']['values']) > 1:
            header.append(f'static constexpr {_ctype(node["type"])} constant_{node["id"]}[]={{' +
                          ','.join(str(v) for v in node['attributes']['values']) + '};')
    bodies = {}
    for kernel in plan['kernels']:
        refs = {v: f'in{i}' for i, v in enumerate(kernel['inputs'])}
        refs.update({v: f'out{i}' for i, v in enumerate(kernel['outputs']) if v not in refs})
        for v in kernel['materialized']:
            if v not in refs: refs[v] = 'tmp_' + v
        params = [f'const {_ctype(values[v]["type"])}* in{i}' for i, v in enumerate(kernel['inputs'])]
        params += [f'{_ctype(values[v]["type"])}* out{i}' for i, v in enumerate(kernel['outputs'])]
        def read(v, index, depth=0):
            if depth > 256: raise ValueError('fusion recursion budget exceeded')
            if v in refs: return f'{refs[v]}[{index}]'
            if v not in kernel['fused']: raise ValueError('missing materialization: ' + v)
            return scalar(values[v], index, depth + 1)
        def scalar(node, index, depth=0):
            op, args, attrs, typ = node['op'], node['operands'], node['attributes'], node['type']
            bits, shape = typ['bits'], typ['shape']
            get = lambda v, ix=index: read(v, ix, depth + 1)
            if op == 'constant':
                data = attrs['values']
                return str(data[0]) if len(data) == 1 else f'constant_{node["id"]}[{index}]'
            if op == 'splat': return get(args[0], '0')
            if op in ('extend_signed', 'truncate'): return f'wrap(int64_t({get(args[0])}),{bits})'
            if op in ('add', 'mul'):
                sign = '+' if op == 'add' else '*'
                return f'wrap(int64_t({get(args[0])}){sign}int64_t({get(args[1])}),{bits})'
            if op in ('min_signed', 'max_signed'):
                fn = 'minimum' if op == 'min_signed' else 'maximum'
                return f'{fn}(int64_t({get(args[0])}),int64_t({get(args[1])}))'
            if op == 'transpose':
                coords = _coords(index, shape); src = values[args[0]]['type']['shape']; original = ['0'] * len(src)
                for d, p in enumerate(attrs['permutation']): original[p] = coords[d]
                return get(args[0], _offset(original, src))
            if op == 'slice':
                coords = [f'{o}+({c})*{s}' for o, c, s in zip(attrs['offsets'], _coords(index, shape), attrs['strides'])]
                return get(args[0], _offset(coords, values[args[0]]['type']['shape']))
            if op == 'concat':
                coords = _coords(index, shape); axis = attrs['axis']; begin = 0; terms = []
                for arg in args:
                    src = values[arg]['type']['shape']; ix = list(coords); ix[axis] = f'({coords[axis]})-{begin}'
                    begin += src[axis]
                    terms.append((f'({coords[axis]})<{begin}', get(arg, _offset(ix, src))))
                expression = terms[-1][1]
                for condition, term in reversed(terms[:-1]): expression = f'({condition}?({term}):({expression}))'
                return expression
            raise ValueError('no scalar operator lowering: ' + op)
        lines = [f'__attribute__((noinline)) inline void compute_{kernel["id"]}(' + ','.join(params) + ') {']
        for v in kernel['materialized']:
            if v not in kernel['outputs']:
                lines.append(f' alignas(32) {_ctype(values[v]["type"])} {refs[v]}[{elements(values[v]["type"])}];')
        for i, v in enumerate(kernel['outputs']):
            typ = values[v]['type']
            lines.append(f' for(unsigned p={elements(typ)};p<{transfer_elements(typ)};++p) out{i}[p]=0;')
        for v in kernel['nodes']:
            if v in kernel['fused']: continue
            node = values[v]; typ = node['type']; dest = refs[v]; bits = typ['bits']
            implementation = implementations.entries[plan['lowerings'][v]]
            lines += implementation.emit({'node': node, 'values': values, 'destination': dest,
                'read': read, 'scalar': scalar, 'elements': elements(typ), 'implementation': implementation})
        for i, v in enumerate(kernel['outputs']):
            if v in kernel['inputs']:
                lines.append(f' for(unsigned p=0;p<{elements(values[v]["type"])};++p) out{i}[p]={refs[v]}[p];')
        lines.append('}')
        bodies[kernel['id']] = '\n'.join(lines) + '\n'
    return '\n'.join(header) + '\n' + ''.join(bodies.values()) + '}\n', bodies


def _fixtures(contract, frames, seed, compiler):
    program = contract['program']; rng = random.Random(seed); batches = []
    for f in range(frames):
        inputs = {}
        for arg in program['inputs']:
            lo, hi = bounds(arg['type']['bits'])
            pool = [lo, hi, -1, 0, 1] if f % 2 else [-3, -1, 0, 1, 2, 4]
            inputs[arg['id']] = [rng.choice(pool) for _ in range(elements(arg['type']))]
        batches.append(inputs)
    expected = [evaluate(program, b) for b in batches]
    evidence = {'kind': 'declared_tensor_interpreter', 'independent_source_checked': False}
    if contract['source']['kind'] == 'verified_source_import':
        artifact = contract['source']['artifact']
        source_batches = [{old: batch[new] for new, old in contract['source']['input_binding'].items()}
                          for batch in batches]
        if artifact['schema'] == 'aie-optimizer-mlir-import-v1':
            from .importing.check import source_evaluate
            original = [source_evaluate(artifact, b) for b in source_batches]
            evidence = {'kind': 'original_mlir_scalar_regions', 'independent_source_checked': True}
        else:
            from .importing.cpp_check import original_cpp_outputs
            original, details = original_cpp_outputs(artifact, source_batches, compiler)
            evidence = {'kind': 'original_cpp_host_execution', 'independent_source_checked': True, **details}
        if original != expected: raise ValueError('original source differs from complete program on generated inputs')
    return batches, expected, {**evidence, 'input_id': identity(batches), 'output_id': identity(expected)}


def generate(contract, plan, destination, *, frames=4, seed=20261018, compiler='g++', rewrite_registry=None, implementation_registry=None):
    validate_plan(contract, plan, rewrite_registry, implementation_registry)
    if type(frames) is not int or not 1 <= frames <= 16: raise ValueError('frames must be 1..16')
    if type(seed) is not int: raise ValueError('literal seed required')
    batches, expected, oracle = _fixtures(contract, frames, seed, compiler)
    if [evaluate(plan['program'], b) for b in batches] != expected:
        raise ValueError('rewritten whole program differs from original')
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=False)
    def write(name, text):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')
    program = plan['program']; values = {v['id']: v for v in program['inputs'] + program['nodes']}
    code, bodies = emit_compute(plan, implementation_registry); write('compute.hpp', code)
    golden = []
    for frame in expected:
        golden.append([out + [0] * (transfer_elements(o['type']) - len(out)) for out, o in zip(frame, program['outputs'])])
    write('fixtures.json', json.dumps({'inputs': batches, 'logical_outputs': expected}, indent=2) + '\n')
    write('golden.json', json.dumps(golden) + '\n'); write('source_check.json', json.dumps(oracle, indent=2) + '\n')
    for i, arg in enumerate(program['inputs']):
        typ = arg['type']; data = []
        for batch in batches:
            v = batch[arg['id']]; data.extend(v + [0] * (transfer_elements(typ) - len(v)))
        line = 128 // typ['bits']
        write(f'data/I{i}.txt', '\n'.join(' '.join(map(str, data[j:j+line])) for j in range(0, len(data), line)) + '\n')
    cpu = ['#include <cstdio>', '#include "compute.hpp"', 'int main() { unsigned long errors=0;']
    for f, batch in enumerate(batches):
        cpu.append('{'); host_refs = {}
        for i, arg in enumerate(program['inputs']):
            data = batch[arg['id']]; typ = arg['type']; name = f'input_{i}'
            cpu.append(f'alignas(32) {_ctype(typ)} {name}[{transfer_elements(typ)}]={{' + ','.join(map(str, data)) + '};')
            host_refs[arg['id']] = name
        for kernel in plan['kernels']:
            actual = [host_refs[v] for v in kernel['inputs']]
            for i, v in enumerate(kernel['outputs']):
                name = f'{kernel["id"]}_out{i}'; typ = values[v]['type']
                cpu.append(f'alignas(32) {_ctype(typ)} {name}[{transfer_elements(typ)}]={{}};')
                actual.append(name); host_refs[v] = name
            cpu.append(f'opt_program::compute_{kernel["id"]}(' + ','.join(actual) + ');')
        for i, out in enumerate(program['outputs']):
            data = golden[f][i]; typ = out['type']
            cpu.append(f'const {_ctype(typ)} expected_{i}[]={{' + ','.join(map(str, data)) + '};')
            cpu.append(f'for(unsigned p=0;p<{len(data)};++p) errors+=({host_refs[out["value"]]}[p]!=expected_{i}[p]);')
        cpu.append('}')
    cpu += ['std::printf("%lu\\n",errors); return errors?1:0;', '}']
    write('cpu_check.cpp', '\n'.join(cpu) + '\n')
    signatures, wrappers, graph = [], [], ['#pragma once', '#include <adf.h>', '#include "kernels.hpp"',
        'class OptGraph:public adf::graph { public:', f'adf::kernel k[{len(plan["kernels"])}];',
        *[f'adf::input_plio i{i};' for i in range(len(program['inputs']))],
        *[f'adf::output_plio o{i};' for i in range(len(program['outputs']))], 'OptGraph() {']
    producers = {}
    for i, arg in enumerate(program['inputs']):
        graph.append(f'i{i}=adf::input_plio::create("I{i}",adf::plio_128_bits,"data/I{i}.txt");')
        producers[arg['id']] = f'i{i}.out[0]'
    for index, kernel in enumerate(plan['kernels']):
        params = [f'adf::input_buffer<int{values[v]["type"]["bits"]}>& in{i}' for i, v in enumerate(kernel['inputs'])]
        params += [f'adf::output_buffer<int{values[v]["type"]["bits"]}>& out{i}' for i, v in enumerate(kernel['outputs'])]
        sig = f'void program_{kernel["id"]}(' + ','.join(params) + ')'
        signatures.append(sig + ';')
        actual = [f'in{i}.data()' for i in range(len(kernel['inputs']))] + [f'out{i}.data()' for i in range(len(kernel['outputs']))]
        wrappers.append(sig + '{opt_program::compute_' + kernel['id'] + '(' + ','.join(actual) + ');}\n')
        graph += [f'k[{index}]=adf::kernel::create(program_{kernel["id"]});',
                  f'adf::source(k[{index}])="kernels.cpp"; adf::runtime<adf::ratio>(k[{index}])=1.0;',
                  f'adf::stack_size(k[{index}])={kernel["stack_bytes"]};',
                  f'adf::location<adf::kernel>(k[{index}])=adf::tile({kernel["tile"][0]},{kernel["tile"][1]});']
        for i, v in enumerate(kernel['inputs']):
            graph += [f'adf::connect<>({producers[v]},k[{index}].in[{i}]);',
                      f'adf::dimensions(k[{index}].in[{i}])={{{transfer_elements(values[v]["type"])}}};']
        for i, v in enumerate(kernel['outputs']):
            graph.append(f'adf::dimensions(k[{index}].out[{i}])={{{transfer_elements(values[v]["type"])}}};')
            producers[v] = f'k[{index}].out[{i}]'
    for i, out in enumerate(program['outputs']):
        graph += [f'o{i}=adf::output_plio::create("O{i}",adf::plio_128_bits,"out_{i}.txt");',
                  f'adf::connect<>({producers[out["value"]]},o{i}.in[0]);']
    graph += ['}', '};']
    write('kernels.hpp', '#pragma once\n#include <adf.h>\n' + '\n'.join(signatures) + '\n')
    write('kernels.cpp', '#include "kernels.hpp"\n#include "compute.hpp"\n' + ''.join(wrappers))
    write('graph.hpp', '\n'.join(graph) + '\n')
    write('TopGraph.cpp', '#include "graph.hpp"\nOptGraph graph;\nint main(){graph.init();graph.run(' + str(frames) + ');graph.end();return 0;}\n')
    from .program_costs import cost_request
    write('analyzer_request.json', json.dumps(cost_request(contract, plan, rewrite_registry=rewrite_registry,
          implementation_registry=implementation_registry), indent=2) + '\n')
    if contract['source']['kind'] == 'verified_source_import':
        write('source_import.json', json.dumps(contract['source']['artifact'], indent=2) + '\n')
    hashes = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
    manifest = {'schema': 'aie-optimizer-program-artifact-v1', 'contract': contract, 'plan': plan,
        'frames': frames, 'seed': seed, 'files': hashes, 'source_digest': identity(hashes),
        'kernel_bindings': [{'function': 'program_' + k['id'], 'index': i, 'tile': k['tile']}
                            for i, k in enumerate(plan['kernels'])],
        'numerical': 'unverified_generated_code', 'native': 'unverified',
        'oracle': oracle, 'lowering': 'registered_operator_implementations'}
    write('manifest.json', json.dumps(manifest, indent=2) + '\n')
    return manifest


def check_artifact(root, rewrite_registry=None, implementation_registry=None):
    root = Path(root).resolve(); manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'aie-optimizer-program-artifact-v1': raise ValueError('program artifact schema')
    validate_plan(manifest['contract'], manifest['plan'], rewrite_registry, implementation_registry)
    if identity(manifest['files']) != manifest['source_digest']: raise ValueError('program digest mismatch')
    for name, expected in manifest['files'].items():
        p = (root / name).resolve()
        if root not in p.parents or not p.is_file() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('unsafe, missing or changed program artifact file: ' + name)
    if (root / 'compute.hpp').read_text(encoding='utf-8') != emit_compute(manifest['plan'], implementation_registry)[0]:
        raise ValueError('generated compute differs from operator lowering replay')
    return root, manifest


def cpu_check(root, compiler='g++', rewrite_registry=None, implementation_registry=None):
    from .runner import command
    root, m = check_artifact(root, rewrite_registry, implementation_registry)
    exe = root / ('cpu_check.exe' if __import__('os').name == 'nt' else 'cpu_check')
    build = command(root, [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-ftrapv', '-DAIE_OPT_HOST',
                           str(root / 'cpu_check.cpp'), '-o', str(exe)], 120, 'cpu_compile')
    run = command(root, [str(exe)], 30, 'cpu_run') if build['exit_code'] == 0 else None
    passed = run is not None and run['exit_code'] == 0 and (root / 'cpu_run.log').read_text().strip() == '0'
    result = {'schema': 'aie-optimizer-program-cpu-check-v1', 'status': 'passed' if passed else 'failed',
        'candidate_id': m['plan']['id'], 'source_digest': m['source_digest'], 'compile': build, 'run': run,
        'frames': m['frames'], 'outputs_checked': sum(elements(o['type']) for o in m['plan']['program']['outputs']) * m['frames'],
        'original_source_checked': m['oracle']['independent_source_checked'],
        'scope': 'complete generated compute DAG; no ADF scheduling or hardware latency proof'}
    (root / 'cpu_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
