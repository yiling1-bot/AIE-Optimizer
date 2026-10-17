"""Verified MLIR tensor SSA import using a pinned parser, not text templates.

The accepted language is intentionally closed: static integer tensors, pure
straight-line functions, elementwise arithmetic, and structured contractions.
Unsupported effects, control flow, encodings and arithmetic flags fail closed.
"""
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from ..contracts import identity
from .ir import GraphBuilder, ImportFailure, fail, tensor_type, signed

PARSER_VERSION = '0.71.0'


def implementation_identity():
    names = ('mlir_frontend.py', 'mlir_compat.py', 'ir.py', 'admission.py', 'frontends.py')
    return identity({n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def parse_module(text, name='<memory>'):
    try:
        from xdsl.context import Context
        from xdsl.parser import Parser
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        from xdsl.dialects.tensor import Tensor
        from .mlir_compat import dialects
        from xdsl.dialects.memref import MemRef
        from xdsl.dialects.scf import Scf
        from xdsl.dialects.cf import Cf
        actual = version('xdsl')
    except ImportError as exc:
        raise ImportFailure('missing_dependency', 'install Optimizer/requirements-import.txt') from exc
    if actual != PARSER_VERSION:
        fail(f'expected xdsl {PARSER_VERSION}, found {actual}', code='parser_version')
    if len(text.encode('utf-8')) > 2_000_000:
        fail('MLIR source exceeds 2 MB budget', code='resource_limit')
    ctx = Context()
    Arith,Linalg = dialects()
    for dialect in (Builtin, Func, Arith, Tensor, Linalg, MemRef, Scf, Cf): ctx.load_dialect(dialect)
    try:
        module = Parser(ctx, text, name).parse_module()
        module.verify()
    except Exception as exc:
        raise ImportFailure('mlir_parse_or_verify', str(exc), name) from exc
    return module


def mlir_type(typ):
    from xdsl.dialects.builtin import TensorType, IntegerType, NoneAttr, Signedness
    shape = []
    if isinstance(typ, TensorType):
        if typ.encoding != NoneAttr(): fail('tensor encoding requires a dedicated layout adapter', code='unsupported_layout')
        shape = list(typ.get_shape()); typ = typ.get_element_type()
    if not isinstance(typ, IntegerType) or typ.signedness.data != Signedness.SIGNLESS:
        fail('only scalar or ranked tensor signless integer types supported', code='unsupported_type')
    return tensor_type(shape, typ.width.data)


def check_fields(op, allowed, site):
    unknown = (set(op.attributes) | set(op.properties)) - set(allowed)
    if unknown: fail('unhandled attributes/properties: '+', '.join(sorted(unknown)), site)


ARITH = {'arith.addi': 'add', 'arith.muli': 'mul', 'arith.maxsi': 'max_signed',
         'arith.minsi': 'min_signed', 'arith.extsi': 'extend_signed', 'arith.trunci': 'truncate'}


def check_arith(op, site):
    check_fields(op, ('overflowFlags',), site)
    flags = op.properties.get('overflowFlags') or op.attributes.get('overflowFlags')
    if flags is not None and flags.data:
        fail('overflow/poison flags require a separate proof', site)
    if op.regions: fail('arithmetic regions not supported', site)


class Importer:
    def __init__(self, module):
        self.graph = GraphBuilder(); self.functions = {}; self.stack = []
        check_fields(module, ('sym_name',), 'module')
        for op in module.body.block.ops:
            if op.name != 'func.func': fail('module must contain only func.func operations', op.name)
            name = op.sym_name.data
            if name in self.functions: fail('duplicate function symbol: '+name)
            self.functions[name] = op

    def lookup(self, env, value, site):
        if value not in env: fail('value is outside the supported SSA scope', site)
        return env[value]

    def function(self, name, arguments=None):
        if name not in self.functions: fail('external/unresolved call: '+name, code='external_call')
        if name in self.stack or len(self.stack) >= 32:
            fail('recursive call or call-depth limit reached', name, 'recursive_call')
        op = self.functions[name]
        check_fields(op, ('sym_name','function_type','sym_visibility'), '@'+name)
        if len(op.body.blocks) != 1: fail('a defined single-block function is required', '@'+name)
        block = op.body.block
        if arguments is None:
            arguments = [self.graph.input(a.name_hint or f'arg{i}', mlir_type(a.type), i) for i,a in enumerate(block.args)]
        if len(arguments) != len(block.args): fail('call arity mismatch', '@'+name)
        for value, arg in zip(arguments, block.args):
            if self.graph.values[value]['type'] != mlir_type(arg.type): fail('call type mismatch', '@'+name)
        env = dict(zip(block.args, arguments)); self.stack.append(name)
        try:
            for i, inner in enumerate(block.ops):
                site = '/'.join('@'+s for s in self.stack)+f'/op{i}:{inner.name}'
                if inner.name == 'func.return':
                    check_fields(inner, (), site)
                    result = [self.lookup(env,v,site) for v in inner.operands]
                    if any(not self.graph.values[v]['initialized'] for v in result): fail('returning uninitialized tensor', site)
                    return result
                self.operation(inner, env, site)
            fail('missing function return', '@'+name)
        finally:
            self.stack.pop()

    def operation(self, op, env, site, lift_shape=None):
        origin = {'site': site, 'location': str(op.location), 'operation': op.name,
                  'results': [v.name_hint for v in op.results], 'assembly': str(op)}
        if op.name == 'func.call':
            if lift_shape is not None: fail('calls inside generic regions not supported', site)
            check_fields(op, ('callee',), site)
            if op.callee.nested_references.data: fail('nested symbol references not supported', site)
            args = [self.lookup(env,v,site) for v in op.operands]
            results = self.function(op.callee.root_reference.data, args)
            if len(results) != len(op.results): fail('call result count mismatch', site)
            for value, result in zip(op.results, results): env[value] = result
            return
        if len(op.results) != 1: fail('one-result pure operation required', site)
        typ = mlir_type(op.results[0].type)
        if lift_shape is not None:
            if typ['shape']: fail('generic body must be scalar', site)
            typ = tensor_type(lift_shape, typ['bits'])
        args = [self.lookup(env,v,site) for v in op.operands]
        if op.name == 'arith.constant':
            from xdsl.dialects.builtin import IntegerAttr, DenseIntOrFPElementsAttr
            check_fields(op, ('value',), site)
            val = op.value
            if isinstance(val, IntegerAttr): values = [val.value.data]
            elif isinstance(val, DenseIntOrFPElementsAttr): values = list(val.get_values())
            else: fail('only integer scalar/dense constants supported', site)
            result = self.graph.add('constant', (), typ, {'values': values}, origin)
        elif op.name in ARITH:
            check_arith(op, site)
            if lift_shape is not None:
                args = [self.broadcast(v, lift_shape, origin) for v in args]
            result = self.graph.add(ARITH[op.name], args, typ, {}, origin)
        elif op.name == 'tensor.empty':
            check_fields(op, (), site)
            if args: fail('dynamic allocation sizes unsupported', site)
            result = self.graph.add('empty', (), typ, {}, origin)
        elif op.name == 'linalg.fill':
            check_fields(op, ('operandSegmentSizes',), site)
            if len(op.inputs) != 1 or len(op.outputs) != 1: fail('one scalar fill/output required', site)
            self.copy_region(op,site)
            src = args[0]
            if self.graph.values[args[1]]['type'] != typ: fail('fill output type mismatch', site)
            # Do not read the empty destination. The complete domain is written.
            result = self.graph.add('splat', (src,), typ, {}, origin)
        elif op.name in ('linalg.generic', 'linalg.matmul'):
            result = self.structured(op, env, typ, site, origin)
        elif op.name == 'linalg.transpose':
            check_fields(op, ('permutation','operandSegmentSizes'), site)
            self.copy_region(op,site)
            if len(args) != 2 or self.graph.values[args[1]]['type'] != typ: fail('transpose destination mismatch', site)
            result = self.graph.add('transpose', (args[0],), typ, {'permutation': list(op.permutation.get_values())}, origin)
        elif op.name == 'tensor.extract_slice':
            check_fields(op, ('static_offsets','static_sizes','static_strides','operandSegmentSizes'), site)
            if len(args) != 1: fail('only static slices supported', site)
            sizes = list(op.static_sizes.get_values())
            if sizes != typ['shape']: fail('rank-reducing/dynamic slices not supported', site)
            result = self.graph.add('slice', args, typ, {'offsets':list(op.static_offsets.get_values()),
                'strides':list(op.static_strides.get_values())}, origin)
        elif op.name == 'tensor.concat':
            check_fields(op, ('dim',), site)
            result = self.graph.add('concat', args, typ, {'axis':op.dim.value.data}, origin)
        else:
            fail('no semantic handler for '+op.name, site, 'unsupported_operation')
        env[op.results[0]] = result

    def copy_region(self, op, site):
        if len(op.body.blocks) != 1: fail('single-block copy region required', site)
        block = op.body.block; ops = list(block.ops)
        if len(block.args) != 2 or len(ops) != 1 or ops[0].name != 'linalg.yield' or list(ops[0].operands) != [block.args[0]]:
            fail('named copy operation has a non-copy body', site)
        check_fields(ops[0], (), site)
        for arg,operand in zip(block.args,op.operands):
            bt,ot = mlir_type(arg.type),mlir_type(operand.type)
            if bt['shape'] or bt['bits'] != ot['bits']:
                fail('named copy block-argument types disagree with operands',site)

    def broadcast(self, value, shape, origin):
        typ = self.graph.values[value]['type']
        if typ['shape'] == shape: return value
        if typ['shape']: fail('unsupported pointwise indexing/broadcast', origin)
        return self.graph.add('splat', (value,), tensor_type(shape,typ['bits']), {}, origin)

    def structured(self, op, env, typ, site, origin):
        from xdsl.ir.affine import AffineDimExpr
        from xdsl.dialects.linalg.attrs import IteratorType
        allowed = ('indexing_maps','iterator_types','operandSegmentSizes','doc')
        check_fields(op, allowed, site)
        if len(op.outputs) != 1 or len(op.body.blocks) != 1:
            fail('one-output, single-block tensor structured operation required', site)
        operands = list(op.inputs)+list(op.outputs)
        if any(not mlir_type(v.type)['shape'] for v in operands): fail('structured operands must be tensors', site)
        if mlir_type(op.outputs[0].type) != typ: fail('structured result/output mismatch', site)
        maps = []
        iterators = [v.data for v in op.get_iterator_types()]
        for attr in op.get_indexing_maps():
            amap = attr.data
            if amap.num_symbols or amap.num_dims != len(iterators) or any(not isinstance(x, AffineDimExpr) for x in amap.results):
                fail('only projected dimension maps supported', site)
            dims = [x.position for x in amap.results]
            if any(not 0 <= p < len(iterators) for p in dims): fail('indexing map dimension out of range',site)
            maps.append(dims)
        if len(maps) != len(operands): fail('structured indexing-map arity mismatch', site)
        if len(op.body.block.args) != len(operands): fail('structured block-argument arity mismatch', site)
        for operand, block_arg in zip(operands, op.body.block.args):
            if mlir_type(operand.type)['bits'] != mlir_type(block_arg.type)['bits'] or mlir_type(block_arg.type)['shape']:
                fail('structured block-argument type mismatch', site)
        reductions = [i for i,t in enumerate(iterators) if t == IteratorType.REDUCTION]
        if reductions:
            return self.contraction(op, env, typ, maps, iterators, site, origin)
        if any(t != IteratorType.PARALLEL for t in iterators) or len(iterators) != len(typ['shape']):
            fail('unsupported iterator semantics', site)
        expected = list(range(len(typ['shape'])))
        if any(m != expected for m in maps): fail('pointwise maps must be identity', site)
        for value in operands:
            if mlir_type(value.type)['shape'] != typ['shape']: fail('pointwise domain mismatch', site)
        inner_env = dict(env)
        inner_env.update((a,self.lookup(env,v,site)) for a,v in zip(op.body.block.args,operands))
        for i, inner in enumerate(op.body.block.ops):
            inner_site = site+f'/body/op{i}:{inner.name}'
            if inner.name == 'linalg.yield':
                check_fields(inner, (), inner_site)
                if len(inner.operands) != 1: fail('one yield required', inner_site)
                value = self.broadcast(self.lookup(inner_env,inner.operands[0],inner_site),typ['shape'],origin)
                if self.graph.values[value]['type'] != typ or not self.graph.values[value]['initialized']:
                    fail('uninitialized or mistyped pointwise result', inner_site)
                return value
            if inner.name != 'arith.constant' and inner.name not in ARITH:
                fail('only pure arithmetic in pointwise bodies', inner_site)
            self.operation(inner, inner_env, inner_site, typ['shape'])
        fail('missing structured yield', site)

    def contraction(self, op, env, typ, maps, iterators, site, origin):
        from xdsl.dialects.linalg.attrs import IteratorType
        if len(op.inputs) != 2 or len(iterators) != 3 or len(typ['shape']) != 2 or len(maps[2]) != 2:
            fail('only rank-two matrix contraction recognized', site)
        i,j = maps[2]
        if i == j or iterators[i] != IteratorType.PARALLEL or iterators[j] != IteratorType.PARALLEL:
            fail('contraction output must map two parallel dimensions', site)
        ks = [d for d,t in enumerate(iterators) if t == IteratorType.REDUCTION]
        if len(ks) != 1: fail('one contraction dimension required', site)
        k = ks[0]
        if sorted(maps[0]) != sorted([i,k]) or sorted(maps[1]) != sorted([k,j]):
            fail('contraction maps do not describe a matrix product', site)
        block = op.body.block; symbolic = {a:('arg',n,mlir_type(a.type)['bits']) for n,a in enumerate(block.args)}
        yielded = None
        for inner in block.ops:
            if inner.name == 'linalg.yield':
                check_fields(inner, (), site)
                if len(inner.operands) != 1: fail('one contraction yield required', site)
                yielded = symbolic.get(inner.operands[0]); continue
            if inner.name not in ('arith.extsi','arith.muli','arith.addi') or len(inner.results) != 1:
                fail('unrecognized contraction scalar computation', site)
            check_arith(inner, site)
            if any(v not in symbolic for v in inner.operands): fail('contraction capture not supported', site)
            result_type = mlir_type(inner.results[0].type)
            if result_type['shape']: fail('contraction body must be scalar', site)
            symbolic[inner.results[0]] = (inner.name, *(symbolic[v] for v in inner.operands), result_type['bits'])
        def input_arg(term, number):
            if term == ('arg',number,typ['bits']): return True
            return (len(term) == 3 and term[0] == 'arith.extsi' and term[2] == typ['bits'] and
                    term[1] == ('arg',number,mlir_type(block.args[number].type)['bits']) and term[1][2] < typ['bits'])
        if not yielded or yielded[0] != 'arith.addi' or yielded[-1] != typ['bits']:
            fail('contraction must add its full product into the output accumulator', site)
        lhs,rhs = yielded[1:3]; acc = ('arg',2,typ['bits'])
        mul = lhs if rhs == acc else rhs if lhs == acc else None
        if not mul or mul[0] != 'arith.muli' or mul[-1] != typ['bits'] or not (
                (input_arg(mul[1],0) and input_arg(mul[2],1)) or (input_arg(mul[1],1) and input_arg(mul[2],0))):
            fail('signed product and accumulation pattern not proven', site)
        args = [self.lookup(env,v,site) for v in list(op.inputs)+list(op.outputs)]
        for idx, expected in enumerate(([i,k],[k,j])):
            if maps[idx] != expected:
                old = self.graph.values[args[idx]]['type']
                args[idx] = self.graph.add('transpose',(args[idx],),tensor_type(old['shape'][::-1],old['bits']),{'permutation':[1,0]},origin)
        return self.graph.add('matmul', args, typ, {'promotion':'signed', 'accumulation':'modular'}, origin)


def import_mlir_text(text, entry=None, source_name='<memory>'):
    from .admission import admit_two_layer
    module = parse_module(text,source_name); importer = Importer(module)
    if entry is None:
        candidates = [n for n,f in importer.functions.items() if not f.sym_visibility or f.sym_visibility.data == 'public']
        if len(candidates) != 1: fail('select an entry function explicitly', code='ambiguous_entry')
        entry = candidates[0]
    outputs = importer.function(entry)
    program = {'schema':'aie-optimizer-tensor-ssa-v1','entry':entry,
        'inputs':importer.graph.inputs,'nodes':importer.graph.nodes,
        'outputs':[{'position':i,'value':v,'type':importer.graph.values[v]['type']} for i,v in enumerate(outputs)],
        'effects':'pure_tensor_function','physical_mapping':None}
    program['id'] = identity(program)
    result = {'schema':'aie-optimizer-mlir-import-v1','status':'imported',
        'source':{'name':source_name,'text':text,'sha256':sha256(text.encode('utf-8')).hexdigest()},
        'frontend':{'parser':'xdsl','parser_version':PARSER_VERSION,'implementation':implementation_identity()},
        'program':program,'backend_admission':admit_two_layer(program),
        'scope':'source tensor semantics and signature; no native interface/mapping import'}
    result['id'] = identity(result)
    return result


def import_mlir(path, entry=None):
    path = Path(path)
    # Preserve CRLF and all other source bytes: provenance hashes must identify
    # the actual UTF-8 input file, not Python's universal-newline normalization.
    try:
        text = path.read_bytes().decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ImportFailure('source_encoding','MLIR input must be UTF-8',str(path)) from exc
    return import_mlir_text(text, entry, str(path.resolve()))


@lru_cache(maxsize=16)
def _replay(text, entry, source_name, implementation):
    return json.dumps(import_mlir_text(text,entry,source_name),sort_keys=True)


def validate_import(artifact):
    if artifact.get('schema') != 'aie-optimizer-mlir-import-v1': fail('not an MLIR import artifact', code='invalid_import')
    try:
        expected = json.loads(_replay(artifact['source']['text'], artifact['program']['entry'],
                             artifact['source']['name'], implementation_identity()))
    except (KeyError,TypeError) as exc:
        raise ImportFailure('invalid_import','incomplete import evidence') from exc
    if artifact != expected: fail('import evidence fails source/semantic replay', code='invalid_import')
    return expected['program']


def export_mlir(artifact):
    validate_import(artifact)
    return str(parse_module(artifact['source']['text'],artifact['source']['name']))+'\n'
