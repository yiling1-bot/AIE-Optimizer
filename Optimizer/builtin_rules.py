"""First executable semantic rules. Paper links motivate mechanisms, not proofs."""
from .registry import Decision, Registry, Rule
from .semantics import Expr, infer, mm, partition_output, push_slice, ranges


def chunks(size, alignment):
    return sorted({size//d for d in (2, 4) if size % d == 0 and size//d % alignment == 0})


def partition_check(node, spec, p, axis, alignment):
    if set(p) != {'chunk'} or type(p['chunk']) is not int:
        return Decision('DENY', 'one integer chunk required')
    size = (infer(node, spec).rows, infer(node, spec).cols)[axis]
    if not 0 < p['chunk'] < size or p['chunk'] % alignment or size % p['chunk']:
        return Decision('DENY', 'partition must be proper, equal and microtile aligned')
    return Decision('ALLOW', 'disjoint ordered slices exactly cover the original domain')


def k_check(node, spec, p):
    if node.op != 'matmul':
        return Decision('DENY', 'requires an INT8 matmul before its quantization barrier')
    if set(p) != {'chunk'} or type(p['chunk']) is not int:
        return Decision('DENY', 'one integer chunk required')
    size = infer(node.args[0], spec).cols
    if not 0 < p['chunk'] < size or p['chunk'] % 8 or size % p['chunk']:
        return Decision('DENY', 'K partition must have complete 8-wide microtiles')
    return Decision('ALLOW', 'all INT32 partials reduced before the unchanged quantization barrier')


def split_k(node, spec, p):
    a, b = node.args
    return Expr('sum', tuple(mm(push_slice(a, 1, lo, hi, spec),
                               push_slice(b, 0, lo, hi, spec), node.stage)
                             for lo, hi in ranges(infer(a, spec).cols, p['chunk'])))


def default_registry():
    registry = Registry()
    registry.register(Rule('tensor.partition_rows', 1, 'M01-domain-partition',
        ('https://arxiv.org/abs/2311.04980',),
        'concat of an ordered, disjoint and complete row partition equals the original pure tensor',
        lambda e,s: ({'chunk': c} for c in chunks(infer(e,s).rows,4)) if e.op == 'quantize' else (),
        lambda e,s,p: partition_check(e,s,p,0,4),
        lambda e,s,p: partition_output(e,0,p['chunk'],s)))
    registry.register(Rule('tensor.partition_k_int32', 1, 'M02-exact-partial-reduction',
        ('https://arxiv.org/abs/2311.04980', 'https://arxiv.org/abs/2504.09688'),
        'INT8 GEMM is the sum of disjoint K products in bounded INT32; never move Q across sum',
        lambda e,s: ({'chunk': c} for c in chunks(infer(e.args[0],s).cols,8)) if e.op == 'matmul' else (),
        k_check, split_k))
    registry.register(Rule('tensor.partition_outputs_rematerialize', 1, 'M03-rematerialization',
        ('https://dspace.mit.edu/entities/publication/a1eb8c3a-91fe-40d7-978d-16b31cf11178',),
        'pure producer recomputation preserves values; ordered column slices reconstruct the output',
        lambda e,s: ({'chunk': c} for c in chunks(infer(e,s).cols,8))
            if e.op == 'quantize' and e.stage == 'Q1' else (),
        lambda e,s,p: (partition_check(e,s,p,1,8) if e.op == 'quantize' and e.stage == 'Q1'
                      else Decision('DENY', 'current rematerialization recognizer requires Q1')),
        lambda e,s,p: partition_output(e,1,p['chunk'],s,True)))
    registry.register(Rule('tensor.propagate_slice', 1, 'M01-domain-partition', (),
        'slice distribution follows the index semantics of pure GEMM, pointwise Q, sum and concat',
        lambda e,s: ({},) if e.op == 'slice' and e.args[0].op not in ('input','slice') else (),
        lambda e,s,p: Decision('ALLOW','valid slice distributed through pure supported operations')
            if e.op == 'slice' and not p else Decision('DENY','parameter-free slice rule'),
        lambda e,s,p: push_slice(e.args[0],e.axis,e.begin,e.end,s)))
    return registry
