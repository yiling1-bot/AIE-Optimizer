"""Factored AND-OR implementation space for an exact two-layer INT8 program.

This is an executable semantic representation prototype. It does not lift C++,
implement equality saturation, lower to AIE, bind an actual core count, model
protocols or predict latency. Native feasibility remains UNKNOWN.
"""
from dataclasses import asdict, dataclass, replace
from itertools import product
from hashlib import sha256
import json


@dataclass(frozen=True)
class Spec:
    m: int = 32
    k: int = 32
    hidden: int = 32
    n: int = 32
    hidden_relu: bool = True

    def validate(self):
        if any(type(v) is not int or v <= 0 for v in (self.m, self.k, self.hidden, self.n)):
            raise ValueError('positive integer dimensions required')
        if type(self.hidden_relu) is not bool:
            raise ValueError('explicit pointwise activation required')
        if max(self.k, self.hidden) * 128 * 128 > 2**31-1:
            raise ValueError('INT32 full reduction range not established')


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    name: str = ''
    axis: int = -1
    begin: int = 0
    end: int = 0
    stage: str = ''
    relu: bool = False


@dataclass(frozen=True)
class TensorType:
    rows: int
    cols: int
    dtype: str
    max_abs: int


@dataclass(frozen=True)
class Alternative:
    name: str
    expression: Expr
    derivation: tuple
    deliveries: tuple
    execution_intent: str


@dataclass(frozen=True)
class Selection:
    family: str
    kernel_l0: str
    kernel_l1: str
    delivery: str


KERNELS = ('dot_product', 'blocked_k_reduction', 'outer_product')


def mm(a, b, stage): return Expr('matmul', (a, b), stage=stage)
def quant(a, stage, relu=False): return Expr('quantize', (a,), stage=stage, relu=relu)
def cut(a, axis, begin, end): return Expr('slice', (a,), axis=axis, begin=begin, end=end)
def ranges(size, chunk): return [(i, min(size, i+chunk)) for i in range(0, size, chunk)]


def infer(expr, spec):
    if expr.op == 'input':
        shapes = {'X': (spec.m, spec.k), 'W0': (spec.k, spec.hidden), 'W1': (spec.hidden, spec.n)}
        return TensorType(*shapes[expr.name], 'int8', 128)
    ts = [infer(a, spec) for a in expr.args]
    if expr.op == 'matmul':
        a, b = ts
        if a.cols != b.rows or a.dtype != 'int8' or b.dtype != 'int8':
            raise ValueError('GEMM requires matching INT8 operands; quantization barriers are explicit')
        bound = a.cols*a.max_abs*b.max_abs
        if bound > 2**31-1: raise ValueError('INT32 GEMM range exceeded')
        return TensorType(a.rows, b.cols, 'int32', bound)
    if expr.op == 'quantize':
        if ts[0].dtype != 'int32': raise ValueError('quantization requires a complete INT32 value')
        return TensorType(ts[0].rows, ts[0].cols, 'int8', 127 if expr.relu else 128)
    if expr.op == 'slice':
        t = ts[0]
        if expr.axis not in (0, 1) or not 0 <= expr.begin < expr.end <= (t.rows, t.cols)[expr.axis]:
            raise ValueError('invalid logical slice')
        return TensorType(expr.end-expr.begin if expr.axis == 0 else t.rows,
                          expr.end-expr.begin if expr.axis == 1 else t.cols, t.dtype, t.max_abs)
    if expr.op == 'sum':
        if not ts or any((t.rows, t.cols, t.dtype) != (ts[0].rows, ts[0].cols, 'int32') for t in ts):
            raise ValueError('partial reduction requires equal INT32 domains')
        bound = sum(t.max_abs for t in ts)
        if bound > 2**31-1: raise ValueError('INT32 partial reduction range exceeded')
        return TensorType(ts[0].rows, ts[0].cols, 'int32', bound)
    if expr.op == 'concat':
        if not ts or expr.axis not in (0, 1): raise ValueError('invalid concatenation')
        if len({t.dtype for t in ts}) != 1: raise ValueError('incompatible concatenation dtype')
        if expr.axis == 0:
            if len({t.cols for t in ts}) != 1: raise ValueError('column mismatch')
            return TensorType(sum(t.rows for t in ts), ts[0].cols, ts[0].dtype, max(t.max_abs for t in ts))
        if len({t.rows for t in ts}) != 1: raise ValueError('row mismatch')
        return TensorType(ts[0].rows, sum(t.cols for t in ts), ts[0].dtype, max(t.max_abs for t in ts))
    if expr.op == 'rematerialize': return ts[0]
    raise ValueError('unsupported expression operator')


def push_slice(expr, axis, begin, end, spec):
    """Propagate a logical slice through supported pure tensor operations."""
    t = infer(expr, spec)
    if not 0 <= begin < end <= (t.rows, t.cols)[axis]: raise ValueError('invalid propagated slice')
    if begin == 0 and end == (t.rows, t.cols)[axis]: return expr
    if expr.op == 'matmul':
        a, b = expr.args
        return replace(expr, args=(push_slice(a, 0, begin, end, spec), b) if axis == 0
                       else (a, push_slice(b, 1, begin, end, spec)))
    if expr.op in ('quantize', 'rematerialize'):
        return replace(expr, args=(push_slice(expr.args[0], axis, begin, end, spec),))
    if expr.op == 'sum':
        return replace(expr, args=tuple(push_slice(a, axis, begin, end, spec) for a in expr.args))
    if expr.op == 'slice' and expr.axis == axis:
        return push_slice(expr.args[0], axis, expr.begin+begin, expr.begin+end, spec)
    if expr.op == 'concat':
        if expr.axis != axis:
            return replace(expr, args=tuple(push_slice(a, axis, begin, end, spec) for a in expr.args))
        parts = []; offset = 0
        for child in expr.args:
            ct = infer(child, spec); length = (ct.rows, ct.cols)[axis]
            lo, hi = max(begin, offset), min(end, offset+length)
            if lo < hi: parts.append(push_slice(child, axis, lo-offset, hi-offset, spec))
            offset += length
        return parts[0] if len(parts) == 1 else replace(expr, args=tuple(parts))
    return cut(expr, axis, begin, end)


def rewrite_k_partition(expr, stage, chunk, spec):
    """Introduce INT32 partials and a full reduction, retaining surrounding Q."""
    children = tuple(rewrite_k_partition(a, stage, chunk, spec) for a in expr.args)
    node = replace(expr, args=children)
    if node.op != 'matmul' or node.stage != stage: return node
    a, b = node.args; k = infer(a, spec).cols
    if k <= chunk: return node
    return Expr('sum', tuple(mm(push_slice(a, 1, lo, hi, spec), push_slice(b, 0, lo, hi, spec), stage)
                             for lo, hi in ranges(k, chunk)))


def partition_output(expr, axis, chunk, spec, rematerialize=False):
    """Partition output domains and propagate the partitions into computation."""
    t = infer(expr, spec); size = (t.rows, t.cols)[axis]
    if size <= chunk: return expr
    def mark(e, instance):
        children = tuple(mark(a, instance) for a in e.args)
        node = replace(e, args=children)
        if node.op == 'matmul' and node.stage == 'L1':
            return replace(node, args=(Expr('rematerialize', (node.args[0],), name=instance), node.args[1]))
        return node
    pieces = []
    for lo, hi in ranges(size, chunk):
        piece = push_slice(expr, axis, lo, hi, spec)
        pieces.append(mark(piece, f'output_axis{axis}_{lo}_{hi}') if rematerialize else piece)
    return Expr('concat', tuple(pieces), axis=axis)


def expression_id(expr):
    return sha256(json.dumps(asdict(expr), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def expand_expressions(spec=Spec(), max_candidates=32, max_depth=2):
    """Bounded rewrite closure, retaining alternatives and their derivations.

    This explores expression rewrites without latency-based greedy acceptance.
    It is not an e-graph congruence-closure/equality-saturation implementation.
    """
    if max_candidates < 1 or max_depth < 0: raise ValueError('invalid expansion budget')
    seed = build(spec)['materialized_layers'].expression
    entries = [{'id': expression_id(seed), 'expression': seed, 'derivations': [()], 'depth': 0}]
    seen = {entries[0]['id']: 0}; cursor = 0; budget_reached = False
    actions = [('rows', max(1, spec.m//d)) for d in (2, 4)]
    actions += [('K0', max(1, spec.k//d)) for d in (2, 4)]
    actions += [('K1', max(1, spec.hidden//d)) for d in (2, 4)]
    actions += [('output_rematerialize', max(1, spec.n//d)) for d in (2, 4)]
    while cursor < len(entries):
        current = entries[cursor]; cursor += 1
        if current['depth'] >= max_depth: continue
        for action, chunk in actions:
            expr = current['expression']
            if action == 'rows': candidate = partition_output(expr, 0, chunk, spec)
            elif action == 'output_rematerialize': candidate = partition_output(expr, 1, chunk, spec, True)
            else: candidate = rewrite_k_partition(expr, 'L0' if action == 'K0' else 'L1', chunk, spec)
            if infer(candidate, spec) != infer(seed, spec): raise ValueError('rewrite output type mismatch')
            identity = expression_id(candidate)
            if identity == current['id']: continue
            trace = current['derivations'][0]+(f'{action}:{chunk}',)
            if identity in seen:
                entry = entries[seen[identity]]
                if trace not in entry['derivations']: entry['derivations'].append(trace)
                continue
            if len(entries) >= max_candidates:
                budget_reached = True
                continue
            seen[identity] = len(entries)
            entries.append({'id': identity, 'expression': candidate, 'derivations': [trace], 'depth': current['depth']+1})
    return entries, budget_reached


def build(spec=Spec()):
    spec.validate()
    x, w0, w1 = (Expr('input', name=n) for n in ('X', 'W0', 'W1'))
    hidden = lambda a, b: quant(mm(a, b, 'L0'), 'Q0', spec.hidden_relu)
    complete = lambda a, b: quant(mm(hidden(a, w0), b, 'L1'), 'Q1')
    row_chunk = max(1, spec.m//4)
    hidden_chunk = max(1, spec.hidden//4)
    out_chunk = max(1, spec.n//4)

    seed = complete(x, w1)
    hidden_reduced = rewrite_k_partition(seed, 'L1', hidden_chunk, spec)

    alternatives = [
        Alternative('materialized_layers', seed, ('source_semantics',),
                    ('dma_buffer', 'shared_buffer'), 'complete H then execute the next layer'),
        Alternative('row_streaming', partition_output(seed, 0, row_chunk, spec),
                    ('row_partition', 'propagate_pointwise_quantization', 'compose_two_layer_row_regions'),
                    ('dma_chunk', 'stream_chunk'), 'produce and consume each row chunk'),
        Alternative('hidden_slice_accumulation', hidden_reduced,
                    ('partition_second_GEMM_reduction', 'propagate_H_column_slice', 'retain_full_K0_quantization'),
                    ('H_fragment_then_INT32_reduce',), 'consume H fragments and accumulate Y before final quantization'),
        Alternative('row_and_hidden_fusion', partition_output(hidden_reduced, 0, max(1, spec.m//2), spec),
                    ('row_partition', 'partition_second_GEMM_reduction', 'compose_region_rewrites'),
                    ('local_H_candidate',), 'candidate for temporal ownership on a shared tile pool'),
        Alternative('recompute_H_per_output_slice', partition_output(seed, 1, out_chunk, spec, True),
                    ('output_column_partition', 'explicit_H_rematerialization'),
                    ('rematerialized_H',), 'repeat the first layer for separate output regions'),
    ]
    goal = TensorType(spec.m, spec.n, 'int8', 128)
    for alt in alternatives:
        if infer(alt.expression, spec) != goal: raise ValueError('candidate output contract mismatch')
    return {a.name: a for a in alternatives}


def selections(graph):
    for alt in graph.values():
        for k0, k1, delivery in product(KERNELS, KERNELS, alt.deliveries):
            yield Selection(alt.name, k0, k1, delivery)


def validate_selection(graph, choice):
    alt = graph[choice.family]
    if choice.kernel_l0 not in KERNELS or choice.kernel_l1 not in KERNELS:
        raise ValueError('unknown child implementation')
    if choice.delivery not in alt.deliveries:
        raise ValueError('delivery choice is incompatible with the selected whole-region implementation')
    return alt


def gemm(a, b, implementation):
    m, k, n = len(a), len(b), len(b[0])
    y = [[0]*n for _ in range(m)]
    if implementation == 'dot_product':
        for i in range(m):
            for j in range(n): y[i][j] = sum(a[i][r]*b[r][j] for r in range(k))
    elif implementation == 'blocked_k_reduction':
        for lo, hi in ranges(k, 8):
            for i in range(m):
                for j in range(n): y[i][j] += sum(a[i][r]*b[r][j] for r in range(lo, hi))
    elif implementation == 'outer_product':
        for r in range(k):
            for i in range(m):
                for j in range(n): y[i][j] += a[i][r]*b[r][j]
    else: raise ValueError('unknown GEMM implementation')
    return y


def evaluate(graph, choice, spec, inputs):
    spec.validate()
    alt = validate_selection(graph, choice)
    if infer(alt.expression, spec) != TensorType(spec.m, spec.n, 'int8', 128):
        raise ValueError('selected expression type mismatch')
    for name, shape in {'X': (spec.m, spec.k), 'W0': (spec.k, spec.hidden), 'W1': (spec.hidden, spec.n)}.items():
        value = inputs[name]
        if len(value) != shape[0] or any(len(row) != shape[1] for row in value): raise ValueError('input shape mismatch')
        if any(type(v) is not int or not -128 <= v <= 127 for row in value for v in row): raise ValueError('INT8 input required')
    def run(e):
        if e.op == 'input': return inputs[e.name]
        # No cross-use memoization: rematerialize is an explicit execution choice.
        args = [run(a) for a in e.args]
        if e.op == 'matmul': return gemm(*args, choice.kernel_l0 if e.stage == 'L0' else choice.kernel_l1)
        if e.op == 'quantize':
            lower = 0 if e.relu else -128
            return [[min(127, max(lower, v)) for v in row] for row in args[0]]
        if e.op == 'slice':
            return args[0][e.begin:e.end] if e.axis == 0 else [row[e.begin:e.end] for row in args[0]]
        if e.op == 'sum': return [[sum(a[i][j] for a in args) for j in range(len(args[0][0]))] for i in range(len(args[0]))]
        if e.op == 'concat':
            return [row for a in args for row in a] if e.axis == 0 else [sum((a[i] for a in args), []) for i in range(len(args[0]))]
        if e.op == 'rematerialize': return args[0]
        raise ValueError('unsupported expression')
    return run(alt.expression)


def logical_stats(expr, spec):
    result = {'macs_L0': 0, 'macs_L1': 0, 'H_fragment_count': 0, 'largest_H_fragment_elements': 0}
    def visit(e):
        if e.op == 'matmul':
            a, b = (infer(child, spec) for child in e.args)
            result['macs_'+e.stage] += a.rows*a.cols*b.cols
        if e.op == 'quantize' and e.stage == 'Q0':
            t = infer(e, spec)
            result['H_fragment_count'] += 1
            result['largest_H_fragment_elements'] = max(result['largest_H_fragment_elements'], t.rows*t.cols)
        for child in e.args: visit(child)
    visit(expr)
    result['total_macs'] = result['macs_L0']+result['macs_L1']
    result['memory_note'] = 'fragment size is not peak allocated storage; materialization and lifetimes need native lowering'
    return result


