"""Replayable, operator-local graph rewrites for whole-program tensor SSA.

Integer matmul partitions are identities in the declared modular accumulator
ring; the original initializer is included exactly once. No quantizer moves.
"""
from copy import deepcopy

from .contracts import identity
from .importing.ir import GraphBuilder, tensor_type
from .program_ir import live_nodes, program_from_builder, validate_program


class ProgramRuleRegistry:
    def __init__(self): self.rules = {}

    def register(self, rule_id, version, matches, rewrite, proof):
        if rule_id in self.rules: raise ValueError('duplicate program rule')
        self.rules[rule_id] = {'version': version, 'matches': matches, 'rewrite': rewrite, 'proof': proof}

    def catalog(self):
        return [{'id': k, 'version': v['version'], 'proof': v['proof']} for k, v in self.rules.items()]

    def applications(self, program):
        for rid, rule in self.rules.items():
            for match in rule['matches'](program):
                yield {'rule_id': rid, 'version': rule['version'], **match}

    def apply(self, program, step):
        validate_program(program)
        rule = self.rules.get(step.get('rule_id'))
        if rule is None or step.get('version') != rule['version']:
            raise ValueError('unknown program rule/version')
        if step not in list(self.applications(program)): raise ValueError('rewrite guard/match failed')
        result = rule['rewrite'](program, step)
        validate_program(result)
        return result


def _matches(axis):
    def matches(program):
        values = {v['id']: v for v in program['inputs'] + program['nodes']}
        for node in live_nodes(program):
            if node['op'] != 'matmul': continue
            size = values[node['operands'][0]]['type']['shape'][axis]
            if size > 1: yield {'node_id': node['id'], 'split': size // 2}
    return matches


def _partition(axis):
    def rewrite(program, step):
        graph = GraphBuilder(); mapped = {}
        for arg in program['inputs']:
            mapped[arg['id']] = graph.input(arg['name'], arg['type'], arg['position'])
        def cut(value, dim, begin, size, origin):
            old = graph.values[value]['type']; shape = list(old['shape']); shape[dim] = size
            offsets = [0] * len(shape); offsets[dim] = begin
            return graph.add('slice', [value], tensor_type(shape, old['bits']),
                             {'offsets': offsets, 'strides': [1] * len(shape)}, origin)
        for n in program['nodes']:
            args = [mapped[v] for v in n['operands']]
            if n['id'] != step['node_id']:
                mapped[n['id']] = graph.add(n['op'], args, n['type'], deepcopy(n['attributes']), deepcopy(n['origin']))
                continue
            origin = {'rewrite': deepcopy(step), 'source_origin': deepcopy(n['origin'])}
            a, b, init = args; m, width = n['type']['shape']; split = step['split']
            size = graph.values[a]['type']['shape'][axis]
            partials = []
            for begin, extent in ((0, split), (split, size - split)):
                aa = cut(a, axis, begin, extent, origin)
                if axis == 1:
                    bb = cut(b, 0, begin, extent, origin)
                    seed = init if begin == 0 else graph.add('constant', [], n['type'], {'values': [0]}, origin)
                    typ = n['type']
                else:
                    bb = b; seed = cut(init, 0, begin, extent, origin)
                    typ = tensor_type([extent, width], n['type']['bits'])
                partials.append(graph.add('matmul', [aa, bb, seed], typ, deepcopy(n['attributes']), origin))
            mapped[n['id']] = graph.add('add' if axis == 1 else 'concat', partials, n['type'],
                                         {} if axis == 1 else {'axis': 0}, origin)
        return program_from_builder(graph, [mapped[o['value']] for o in program['outputs']], program['entry'])
    return rewrite


def default_program_rules():
    registry = ProgramRuleRegistry()
    registry.register('tensor.matmul.partition_k', 1, _matches(1), _partition(1),
                      'split contraction domain; modular sum; initializer once; quantization unchanged')
    registry.register('tensor.matmul.partition_m', 1, _matches(0), _partition(0),
                      'disjoint output rows and initializer slices; ordered concatenation')
    return registry


def replay(program, trace, registry=None):
    registry = registry or default_program_rules()
    if not isinstance(trace, list) or len(trace) > 8: raise ValueError('rewrite trace budget exceeded')
    for step in trace: program = registry.apply(program, step)
    return program


def semantic_variants(program, max_variants=8, max_depth=1, registry=None):
    registry = registry or default_program_rules()
    if type(max_variants) is not int or not 1 <= max_variants <= 64 or not 0 <= max_depth <= 3:
        raise ValueError('finite semantic expansion budget required')
    variants = [(program, [])]; seen = {identity(program)}; truncated = False
    for current, trace in variants:
        if len(trace) >= max_depth: continue
        for step in registry.applications(current):
            changed = registry.apply(current, step); key = identity(changed)
            if key in seen: continue
            if len(variants) >= max_variants: truncated = True; continue
            seen.add(key); variants.append((changed, trace + [step]))
    return variants, {'semantic_variants': len(variants), 'semantic_budget_truncated': truncated,
                      'rules': registry.catalog(), 'equivalence': 'trusted local rule replay; not unbounded equality saturation'}
