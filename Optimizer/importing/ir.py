"""Pure, typed SSA tensor IR. No implicit saturation, layout or device mapping.

Integer values are signed representatives of signless bitvectors. add/mul and
matmul wrap at their declared width; trunci is NOT a saturating quantizer.
"""
from itertools import product
from math import prod


class ImportFailure(ValueError):
    def __init__(self, code, message, site=None):
        self.code, self.message, self.site = code, message, site
        super().__init__(f'{code}: {message}' + (f' at {site}' if site else ''))

    def diagnostic(self):
        return {'code': self.code, 'message': self.message, 'site': self.site}


def fail(message, site=None, code='unsupported_semantics'):
    raise ImportFailure(code, message, site)


def signed(value, bits):
    return (int(value) + (1 << (bits-1))) % (1 << bits) - (1 << (bits-1))


def bounds(bits):
    return [-(1 << (bits-1)), (1 << (bits-1))-1]


def tensor_type(shape, bits):
    shape = list(shape)
    if bits not in (8, 16, 32, 64):
        fail('only signless i8/i16/i32/i64 are supported', code='unsupported_type')
    if len(shape) > 4 or any(type(d) is not int or d <= 0 for d in shape):
        fail('static positive dimensions, rank <= 4 required', code='unsupported_shape')
    if prod(shape) > 1_048_576:
        fail('tensor exceeds import element budget', code='resource_limit')
    return {'shape': shape, 'bits': bits, 'representation': 'signless_integer',
            'layout': 'logical_indices; physical_layout_unspecified'}


class GraphBuilder:
    def __init__(self):
        self.inputs, self.nodes, self.values = [], [], {}

    def input(self, name, typ, position):
        node = {'id': f'arg{position}', 'name': name, 'position': position,
                'type': typ, 'range': bounds(typ['bits']), 'initialized': True}
        self.inputs.append(node)
        self.values[node['id']] = node
        return node['id']

    def add(self, op, operands, typ, attrs, origin):
        if len(self.nodes) >= 10000:
            fail('expanded graph exceeds 10000 nodes', code='resource_limit')
        args = [self.values[v] for v in operands]
        shape, bits = typ['shape'], typ['bits']
        if op != 'empty' and any(not v['initialized'] for v in args):
            fail('read of tensor.empty without complete initialization', origin,
                 'uninitialized_value')
        interval = bounds(bits)
        if op == 'constant':
            if len(attrs['values']) not in (1, prod(shape)):
                fail('constant shape mismatch', origin)
            vals = [signed(v, bits) for v in attrs['values']]
            attrs = {'values': vals}
            interval = [min(vals), max(vals)]
        elif op == 'empty':
            pass
        elif op == 'splat':
            if args[0]['type']['shape'] or args[0]['type']['bits'] != bits:
                fail('splat requires a scalar of the same element type', origin)
            interval = args[0]['range']
        elif op in ('add', 'mul', 'max_signed', 'min_signed'):
            if any(a['type'] != typ for a in args) or len(args) != 2:
                fail('elementwise operands must have identical types', origin)
            (lo, hi), (ol, oh) = [a['range'] for a in args]
            if op == 'add': interval = [lo+ol, hi+oh]
            elif op == 'mul': interval = [min(lo*ol, lo*oh, hi*ol, hi*oh), max(lo*ol, lo*oh, hi*ol, hi*oh)]
            elif op == 'max_signed': interval = [max(lo, ol), max(hi, oh)]
            else: interval = [min(lo, ol), min(hi, oh)]
        elif op in ('extend_signed', 'truncate'):
            old = args[0]['type']
            if old['shape'] != shape or not ((old['bits'] < bits) if op == 'extend_signed' else (old['bits'] > bits)):
                fail('invalid integer conversion', origin)
            interval = args[0]['range']
        elif op == 'matmul':
            a, b, init = args
            ash, bsh = a['type']['shape'], b['type']['shape']
            if len(ash) != 2 or len(bsh) != 2 or ash[1] != bsh[0] or shape != [ash[0], bsh[1]] or init['type'] != typ:
                fail('matmul dimensions/accumulator do not match', origin)
            if a['type']['bits'] > bits or b['type']['bits'] > bits:
                fail('matmul requires non-narrowing signed promotion', origin)
            products = [u*v for u in a['range'] for v in b['range']]
            interval = [ash[1]*min(products)+init['range'][0], ash[1]*max(products)+init['range'][1]]
        elif op == 'transpose':
            perm = attrs['permutation']; old = args[0]['type']
            if sorted(perm) != list(range(len(shape))) or shape != [old['shape'][p] for p in perm] or bits != old['bits']:
                fail('invalid transpose permutation/type', origin)
            interval = args[0]['range']
        elif op == 'slice':
            old = args[0]['type']; offsets, strides = attrs['offsets'], attrs['strides']
            if len(shape) != len(old['shape']) or len(offsets) != len(shape) or len(strides) != len(shape) or bits != old['bits']:
                fail('only rank-preserving slices supported', origin)
            if any(o < 0 or s <= 0 or o+(n-1)*s >= d for o,s,n,d in zip(offsets,strides,shape,old['shape'])):
                fail('slice exceeds its source domain', origin)
            interval = args[0]['range']
        elif op == 'concat':
            axis = attrs['axis']
            if not args or not 0 <= axis < len(shape): fail('invalid concat axis', origin)
            if any(len(a['type']['shape']) != len(shape) or a['type']['bits'] != bits or
                   any(a['type']['shape'][i] != shape[i] for i in range(len(shape)) if i != axis) for a in args):
                fail('concat operand type mismatch', origin)
            if sum(a['type']['shape'][axis] for a in args) != shape[axis]: fail('concat extent mismatch', origin)
            interval = [min(a['range'][0] for a in args), max(a['range'][1] for a in args)]
        else:
            fail('unknown tensor IR operator: '+op, origin)
        wraps = interval[0] < bounds(bits)[0] or interval[1] > bounds(bits)[1]
        node = {'id': f'v{len(self.nodes)}', 'op': op, 'operands': list(operands),
                'type': typ, 'attributes': attrs, 'range': bounds(bits) if wraps else interval,
                'may_wrap': wraps, 'initialized': op != 'empty', 'origin': origin}
        self.nodes.append(node); self.values[node['id']] = node
        return node['id']


def indices(shape):
    return product(*(range(d) for d in shape))


def offset(index, shape):
    result = 0
    for i, d in zip(index, shape): result = result*d+i
    return result


def evaluate(program, inputs):
    """Evaluate flat logical-index arrays, independently of packing/codegen."""
    expected = {x['id'] for x in program['inputs']}
    if set(inputs) != expected: fail('input IDs differ from imported signature', code='invalid_inputs')
    values, types = {}, {}
    for arg in program['inputs']:
        data = inputs[arg['id']]; lo, hi = bounds(arg['type']['bits'])
        if len(data) != prod(arg['type']['shape']) or any(type(v) is not int or not lo <= v <= hi for v in data):
            fail('input shape or signed integer domain mismatch', arg['id'], 'invalid_inputs')
        values[arg['id']], types[arg['id']] = list(data), arg['type']
    for node in program['nodes']:
        op, operands, typ, attrs = node['op'], node['operands'], node['type'], node['attributes']
        args = [values[v] for v in operands]; shape, bits = typ['shape'], typ['bits']
        if op == 'empty': result = None
        elif op == 'constant':
            result = attrs['values'] * (prod(shape) if len(attrs['values']) == 1 else 1)
        elif op == 'splat': result = args[0]*prod(shape)
        elif op in ('extend_signed', 'truncate'): result = [signed(v, bits) for v in args[0]]
        elif op in ('add', 'mul', 'max_signed', 'min_signed'):
            fn = {'add': lambda a,b:a+b, 'mul':lambda a,b:a*b, 'max_signed':max, 'min_signed':min}[op]
            result = [signed(fn(a,b), bits) for a,b in zip(*args)]
        elif op == 'matmul':
            m,n = shape; k = types[operands[0]]['shape'][1]
            if m*n*k > 20_000_000: fail('interpreter matmul work budget exceeded', code='resource_limit')
            result = [signed(args[2][i*n+j]+sum(args[0][i*k+r]*args[1][r*n+j] for r in range(k)), bits)
                      for i in range(m) for j in range(n)]
        elif op == 'transpose':
            src = types[operands[0]]['shape']; perm = attrs['permutation']; result = []
            for ix in indices(shape):
                orig = [0]*len(ix)
                for j,p in enumerate(perm): orig[p] = ix[j]
                result.append(args[0][offset(orig, src)])
        elif op == 'slice':
            src = types[operands[0]]['shape']
            result = [args[0][offset([o+i*s for o,i,s in zip(attrs['offsets'], ix, attrs['strides'])], src)] for ix in indices(shape)]
        elif op == 'concat':
            axis = attrs['axis']; result = []
            for ix in indices(shape):
                pos = ix[axis]
                for arg, val in zip(args, operands):
                    src = types[val]['shape']
                    if pos < src[axis]:
                        orig = list(ix); orig[axis] = pos
                        result.append(arg[offset(orig, src)]); break
                    pos -= src[axis]
        else: fail('unknown tensor IR operator: '+op)
        values[node['id']], types[node['id']] = result, typ
    return [values[x['value']] for x in program['outputs']]
