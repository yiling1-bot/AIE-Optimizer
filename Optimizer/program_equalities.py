"""Callable tensor equalities, guarded by exact bitvector and shape semantics.

These are trusted, versioned algebraic rules, not a claim of a formal theorem
prover. The same ProgramRuleRegistry replays every certificate application.
"""
from copy import deepcopy

from .importing.ir import GraphBuilder, tensor_type
from .program_ir import live_nodes, program_from_builder
from .program_rules import default_program_rules


def _values(program):
    return {n['id']: n for n in program['inputs'] + program['nodes']}


def _rewrite(build):
    def rewrite(program, step):
        graph = GraphBuilder(); mapped = {}; values = _values(program)
        for arg in program['inputs']:
            mapped[arg['id']] = graph.input(arg['name'], arg['type'], arg['position'])
        for node in program['nodes']:
            args = [mapped[v] for v in node['operands']]
            origin = {'rule': deepcopy(step), 'source_origin': deepcopy(node['origin'])}
            if node['id'] == step['node_id']:
                mapped[node['id']] = build(graph, node, args, values, mapped, origin, step)
            else:
                mapped[node['id']] = graph.add(node['op'], args, node['type'],
                    deepcopy(node['attributes']), deepcopy(node['origin']))
        return program_from_builder(graph, [mapped[o['value']] for o in program['outputs']], program['entry'])
    return rewrite


def _op_matches(ops):
    def matches(program):
        for n in live_nodes(program):
            if n['op'] in ops: yield {'node_id': n['id']}
    return matches


def _commute(g, n, args, values, mapped, origin, step):
    return g.add(n['op'], args[::-1], n['type'], {}, origin)


def _associate_matches(program):
    values = _values(program)
    for n in live_nodes(program):
        if n['op'] not in ('add', 'mul'): continue
        for side in (0, 1):
            child = values[n['operands'][side]]
            if child.get('op') == n['op']:
                yield {'node_id': n['id'], 'side': side}


def _associate(g, n, args, values, mapped, origin, step):
    side = step['side']; child = values[n['operands'][side]]
    x, y = [mapped[v] for v in child['operands']]
    if side == 0:
        right = g.add(n['op'], [y, args[1]], n['type'], {}, origin)
        return g.add(n['op'], [x, right], n['type'], {}, origin)
    left = g.add(n['op'], [args[0], x], n['type'], {}, origin)
    return g.add(n['op'], [left, y], n['type'], {}, origin)


def _n_matches(program):
    for n in live_nodes(program):
        if n['op'] == 'matmul' and n['type']['shape'][1] > 1:
            yield {'node_id': n['id'], 'split': n['type']['shape'][1] // 2}


def _partition_n(g, n, args, values, mapped, origin, step):
    a, b, init = args; m, width = n['type']['shape']; split = step['split']
    parts = []
    for begin, extent in ((0, split), (split, width - split)):
        def cut(v):
            typ = g.values[v]['type']; shape = [typ['shape'][0], extent]
            return g.add('slice', [v], tensor_type(shape, typ['bits']),
                {'offsets': [0, begin], 'strides': [1, 1]}, origin)
        parts.append(g.add('matmul', [a, cut(b), cut(init)],
            tensor_type([m, extent], n['type']['bits']), deepcopy(n['attributes']), origin))
    return g.add('concat', parts, n['type'], {'axis': 1}, origin)


def _zero(v, values):
    # Deliberately syntactic: unknown expressions are not presumed to be zero.
    node = values[v]
    return node.get('op') == 'constant' and set(node['attributes']['values']) == {0}


def _chain_matches(program):
    values = _values(program)
    for n in live_nodes(program):
        if n['op'] != 'matmul' or not _zero(n['operands'][2], values): continue
        bits = n['type']['bits']
        for side in (0, 1):
            child = values[n['operands'][side]]
            if child.get('op') != 'matmul' or not _zero(child['operands'][2], values): continue
            leaves = child['operands'][:2] + [n['operands'][1-side]]
            if child['type']['bits'] == bits and all(values[v]['type']['bits'] == bits for v in leaves):
                yield {'node_id': n['id'], 'side': side}


def _chain(g, n, args, values, mapped, origin, step):
    side = step['side']; child = values[n['operands'][side]]
    a, b = [mapped[v] for v in child['operands'][:2]]; bits = n['type']['bits']
    def mm(x, y, typ):
        zero = g.add('constant', [], typ, {'values': [0]}, origin)
        return g.add('matmul', [x, y, zero], typ, deepcopy(n['attributes']), origin)
    if side == 0:
        inner_type = tensor_type([g.values[b]['type']['shape'][0], n['type']['shape'][1]], bits)
        return mm(a, mm(b, args[1], inner_type), n['type'])
    inner_type = tensor_type([n['type']['shape'][0], g.values[a]['type']['shape'][1]], bits)
    return mm(mm(args[0], a, inner_type), b, n['type'])


def _factor_matches(program):
    values = _values(program)
    for n in live_nodes(program):
        if n['op'] != 'add': continue
        left, right = [values[v] for v in n['operands']]
        if left.get('op') != 'matmul' or right.get('op') != 'matmul': continue
        if left['operands'][0] != right['operands'][0]: continue
        b, c = [values[v] for v in (left['operands'][1], right['operands'][1])]
        a = values[left['operands'][0]]; bits = n['type']['bits']
        if b['type'] == c['type'] and all(v['type']['bits'] == bits for v in (a, b, c)):
            yield {'node_id': n['id']}


def _factor(g, n, args, values, mapped, origin, step):
    left, right = [values[v] for v in n['operands']]
    a, b, z0 = [mapped[v] for v in left['operands']]
    _, c, z1 = [mapped[v] for v in right['operands']]
    bc = g.add('add', [b, c], g.values[b]['type'], {}, origin)
    seed = g.add('add', [z0, z1], n['type'], {}, origin)
    return g.add('matmul', [a, bc, seed], n['type'], deepcopy(left['attributes']), origin)


def _transpose_matches(program):
    values = _values(program)
    for n in live_nodes(program):
        if n['op'] != 'transpose': continue
        child = values[n['operands'][0]]
        if child.get('op') == 'transpose': yield {'node_id': n['id']}


def _transpose(g, n, args, values, mapped, origin, step):
    child = values[n['operands'][0]]
    p = child['attributes']['permutation']; q = n['attributes']['permutation']
    perm = [p[i] for i in q]; source = mapped[child['operands'][0]]
    if perm == list(range(len(perm))): return source
    return g.add('transpose', [source], n['type'], {'permutation': perm}, origin)


def default_tensor_equalities():
    registry = default_program_rules()
    registry.register('tensor.matmul.partition_n', 1, _n_matches, _rewrite(_partition_n),
        'disjoint output columns and initializer columns; ordered concatenation')
    registry.register('tensor.elementwise.commute', 1, _op_matches(('add', 'mul', 'min_signed', 'max_signed')),
        _rewrite(_commute), 'same typed modular add/mul or signed min/max commute')
    registry.register('tensor.elementwise.associate', 1, _associate_matches, _rewrite(_associate),
        'same-width modular add/mul associate; no conversion or quantizer is moved')
    registry.register('tensor.matmul.associate', 1, _chain_matches, _rewrite(_chain),
        'matrix associativity over one modular integer ring; both initializers zero; no narrowing')
    registry.register('tensor.matmul.factor_left', 1, _factor_matches, _rewrite(_factor),
        'A*B+Z0+A*C+Z1=A*(B+C)+(Z0+Z1), all multiplicands at accumulator width')
    registry.register('tensor.transpose.compose', 1, _transpose_matches, _rewrite(_transpose),
        'compose logical permutations; identity permutation returns its source')
    return registry
