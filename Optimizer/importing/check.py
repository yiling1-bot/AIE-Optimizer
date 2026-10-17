"""Differential source-region interpreter versus imported tensor DAG.

The source side executes scalar MLIR region operations and indexing maps. It
does not use contraction recognition, backend admission or generated kernels.
Finite numerical checks complement (and do not replace) structural admission.
"""
from dataclasses import dataclass
from itertools import product
from math import prod
import random
from hashlib import sha256
import json
from pathlib import Path

from .ir import evaluate, fail, bounds
from .mlir_frontend import parse_module, mlir_type
from .frontends import validate_import
from ..contracts import identity


@dataclass
class Value:
    shape: tuple
    bits: int
    data: list


def _wrap(x, bits):
    mask = (1 << bits)-1
    x &= mask
    return x-(1 << bits) if x & (1 << (bits-1)) else x


def _index(ix, shape):
    strides = [prod(shape[i+1:]) for i in range(len(shape))]
    return sum(a*b for a,b in zip(ix,strides))


def source_evaluate(artifact, inputs):
    from xdsl.dialects.builtin import IntegerAttr
    module = parse_module(artifact['source']['text'],artifact['source']['name'])
    funcs = {f.sym_name.data:f for f in module.body.block.ops}
    work = [0]
    def run_function(name, arguments):
        block = funcs[name].body.block; env = dict(zip(block.args,arguments))
        return run_block(block, env, 'func.return')
    def run_block(block, env, terminator):
        for op in block.ops:
            if op.name == terminator: return [env[v] for v in op.operands]
            args = [env[v] for v in op.operands]
            if op.name == 'func.call':
                for v,data in zip(op.results,run_function(op.callee.root_reference.data,args)): env[v] = data
                continue
            typ = mlir_type(op.results[0].type); shape,bits = tuple(typ['shape']),typ['bits']
            if op.name == 'arith.constant':
                vals = [op.value.value.data] if isinstance(op.value,IntegerAttr) else list(op.value.get_values())
                data = [_wrap(int(v),bits) for v in vals]
                if len(data) == 1: data *= prod(shape)
            elif op.name in ('arith.addi','arith.muli','arith.maxsi','arith.minsi','arith.extsi','arith.trunci'):
                work[0] += prod(shape)
                if any(v is None for a in args for v in a.data): fail('source reads uninitialized element')
                if op.name in ('arith.extsi','arith.trunci'): data = [_wrap(x,bits) for x in args[0].data]
                else:
                    fn = {'arith.addi':lambda a,b:a+b,'arith.muli':lambda a,b:a*b,
                          'arith.maxsi':max,'arith.minsi':min}[op.name]
                    data = [_wrap(fn(a,b),bits) for a,b in zip(args[0].data,args[1].data)]
            elif op.name == 'tensor.empty': data = [None]*prod(shape)
            elif op.name.startswith('linalg.'):
                # Execute the actual source region for every original loop index.
                maps = [m.data for m in op.get_indexing_maps()]
                rank = len(op.get_iterator_types()); extents = [None]*rank
                for arg,amap in zip(args,maps):
                    for dim,size in zip(amap.results,arg.shape):
                        if extents[dim.position] not in (None,size): fail('source loop extent mismatch')
                        extents[dim.position] = size
                if any(d is None for d in extents): fail('source loop has no static extent')
                work[0] += prod(extents)
                if work[0] > 20_000_000: fail('source interpreter work budget exceeded',code='resource_limit')
                count = len(op.inputs); output = Value(shape,bits,list(args[-1].data))
                loop_args = args[:count]+[output]
                for ix in product(*(range(d) for d in extents)):
                    local = dict(env)
                    for block_arg,arg,amap in zip(op.body.block.args,loop_args,maps):
                        coords = amap.eval(ix,())
                        local[block_arg] = Value((),arg.bits,[arg.data[_index(coords,arg.shape)]])
                    yielded = run_block(op.body.block,local,'linalg.yield')[0]
                    output.data[_index(maps[-1].eval(ix,()),shape)] = yielded.data[0]
                data = output.data
            elif op.name == 'tensor.extract_slice':
                src = args[0]; offsets = op.static_offsets.get_values(); steps = op.static_strides.get_values()
                data = [src.data[_index(tuple(a+b*c for a,b,c in zip(offsets,ix,steps)),src.shape)]
                        for ix in product(*(range(d) for d in shape))]
            elif op.name == 'tensor.concat':
                axis = op.dim.value.data; data = []
                for ix in product(*(range(d) for d in shape)):
                    pos = ix[axis]
                    for arg in args:
                        if pos < arg.shape[axis]:
                            coords = list(ix); coords[axis] = pos
                            data.append(arg.data[_index(coords,arg.shape)]); break
                        pos -= arg.shape[axis]
            else: fail('reference interpreter has no handler for '+op.name)
            if work[0] > 20_000_000: fail('source interpreter work budget exceeded',code='resource_limit')
            env[op.results[0]] = Value(shape,bits,data)
        fail('missing source terminator')
    args = [Value(tuple(a['type']['shape']),a['type']['bits'],list(inputs[a['id']])) for a in artifact['program']['inputs']]
    return [v.data for v in run_function(artifact['program']['entry'],args)]


def check_import(artifact, cases=8, seed=20261018):
    if artifact.get('schema')=='aie-optimizer-cpp-import-v1':
        from .cpp_check import check_cpp_import
        return check_cpp_import(artifact,cases,seed)
    program = validate_import(artifact)
    if type(cases) is not int or not 1 <= cases <= 64: fail('check cases must be 1..64')
    rng = random.Random(seed); checked = 0; records = []
    for case in range(cases):
        inputs = {}
        for arg in program['inputs']:
            lo,hi = bounds(arg['type']['bits'])
            pool = [0] if case == 0 else [lo,hi,-1,0,1] if case%2 else [-3,-1,0,1,2,4]
            inputs[arg['id']] = [rng.choice(pool) for _ in range(prod(arg['type']['shape']))]
        source = source_evaluate(artifact,inputs); imported = evaluate(program,inputs)
        if source != imported: fail('source/SSA numerical mismatch',f'case {case}', 'semantic_mismatch')
        # A third independent, existing scalar oracle checks the backend bridge.
        if artifact['backend_admission']['status'] == 'supported':
            from ..fixtures import reference
            binding = artifact['backend_admission']['input_binding']; mats = {}
            for name,arg_id in binding.items():
                arg = next(a for a in program['inputs'] if a['id'] == arg_id)
                cols = arg['type']['shape'][1]; flat = inputs[arg_id]
                mats[name] = [flat[i:i+cols] for i in range(0,len(flat),cols)]
            golden = reference(mats['X'],mats['W0'],mats['W1'],artifact['backend_admission']['spec']['hidden_relu'])
            if source != [[v for row in golden for v in row]]: fail('backend bridge oracle mismatch',f'case {case}','semantic_mismatch')
        count = sum(len(v) for v in source); checked += count
        records.append({'case':case,'outputs_checked':count,'status':'passed',
                        'input_identity':identity(inputs),'output_identity':identity(source)})
    return {'schema':'aie-optimizer-import-check-v1','import_id':artifact['id'],'status':'passed',
            'cases':cases,'seed':seed,'values_checked':checked,'records':records,
            'checker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'finite exact source-region vs tensor-SSA differential check; no native execution/performance claim'}


def check_generated_import(directory):
    """Replay the original MLIR on the exact fixtures fed to generated C++."""
    from ..runner import check_artifact
    root,manifest = check_artifact(directory)
    source = manifest['contract']['source']
    if source.get('kind') not in ('verified_mlir_tensor_import_v1','verified_cpp_tensor_import_v1'): fail('artifact has no verified imported source')
    imported = source['import']; program = validate_import(imported)
    binding = source['admission']['input_binding']
    packing = source['admission']['interface_adapter']['packing']
    def unpack(data, shape, block):
        m,n = shape; rm,cn = block; logical = [None]*(m*n); cursor = 0
        for i in range(0,m,rm):
            for j in range(0,n,cn):
                for r in range(rm):
                    for c in range(cn):
                        logical[(i+r)*n+j+c] = data[cursor]; cursor += 1
        return logical
    raw = {name:list(map(int,(root/'data'/f'{name}.txt').read_text().split())) for name in binding}
    for name,arg_id in binding.items():
        arg = next(a for a in program['inputs'] if a['id']==arg_id)
        if len(raw[name]) != manifest['frames']*prod(arg['type']['shape']): fail('generated input frame count mismatch')
    golden = json.loads((root/'golden.json').read_text())
    if len(golden) != manifest['frames']: fail('golden frame count mismatch')
    count = 0; batches=[]; expected_outputs=[]; original_execution=None
    for frame in range(manifest['frames']):
        inputs = {}
        for name,arg_id in binding.items():
            arg = next(a for a in program['inputs'] if a['id']==arg_id); shape = arg['type']['shape']; size = prod(shape)
            inputs[arg_id] = unpack(raw[name][frame*size:(frame+1)*size],shape,packing[name])
        expected = [unpack(golden[frame],program['outputs'][0]['type']['shape'],packing['Y'])]
        batches.append(inputs);expected_outputs.append(expected)
        source_result=source_evaluate(imported,inputs) if imported['schema']=='aie-optimizer-mlir-import-v1' else expected
        if source_result != expected or evaluate(program,inputs) != expected:
            fail('original source differs on generated fixtures',f'frame {frame}','semantic_mismatch')
        count += len(expected[0])
    if imported['schema']=='aie-optimizer-cpp-import-v1':
        from .cpp_check import original_cpp_outputs
        result,original_execution=original_cpp_outputs(imported,batches)
        if result!=expected_outputs:fail('original C++ differs on generated fixtures',code='semantic_mismatch')
    return {'schema':'aie-optimizer-import-artifact-check-v1','status':'passed',
        'import_id':imported['id'],'artifact_source_digest':manifest['source_digest'],
        'frames':manifest['frames'],'values_checked':count,
        'checker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_execution':original_execution,
        'scope':'original source and imported SSA vs generated golden on identical packed inputs; CPU/native checks are separate'}
