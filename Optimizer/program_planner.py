"""Composable graph partition, scalar fusion and execution plans.

No application names, layer counts, W0/W1 bindings or whole-program modes are
used. Costs are evaluated after complete plans are constructed.
"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from .contracts import identity
from .program_ir import elements, live_nodes, seal, transfer_elements, validate_contract
from .program_rules import default_program_rules, replay, semantic_variants

INLINE = {'constant', 'splat', 'add', 'mul', 'min_signed', 'max_signed',
          'extend_signed', 'truncate', 'transpose', 'slice', 'concat'}


def implementation_identity():
    names = ('program_ir.py', 'program_rules.py', 'program_planner.py', 'program_backend.py', 'program_kernels.py',
             'program_egraph.py', 'program_equalities.py')
    return identity({n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def make_plan(contract, *, trace=None, cuts=(), fuse=False, matmul_order='ijk', registry=None,
              lowerings=None, implementation_registry=None):
    base, target = validate_contract(contract, registry)
    trace = [] if trace is None else trace
    program = replay(base, trace, registry); nodes = live_nodes(program)
    if any(v['type']['bits'] not in (8, 16, 32) for v in program['inputs'] + nodes):
        raise ValueError('device lowering supports i8/i16/i32; i64 remains in imported IR')
    if type(fuse) is not bool or matmul_order not in ('ijk', 'ikj'):
        raise ValueError('unknown fusion/matmul lowering')
    if (any(type(c) is not int for c in cuts) or list(cuts) != sorted(set(cuts)) or
            any(not 0 < c < len(nodes) for c in cuts)):
        raise ValueError('cuts must form an ordered disjoint complete node cover')
    count = len(cuts) + 1
    if count > min(target.max_compute_tiles, target.columns * (target.rows - 2)):
        raise ValueError('physical core budget exceeded')
    values = {v['id']: v for v in program['inputs'] + program['nodes']}
    from .program_kernels import default_implementations
    implementations = implementation_registry or default_implementations()
    requested = {} if lowerings is None else dict(lowerings); selected = {}; versions = {}
    chunks = [nodes[a:b] for a, b in zip((0, *cuts), (*cuts, len(nodes)))]
    owner = {n['id']: i for i, chunk in enumerate(chunks) for n in chunk}
    declared_outputs = {o['value'] for o in program['outputs']}
    uses = defaultdict(set)
    for n in nodes:
        for v in n['operands']: uses[v].add(n['id'])
    kernels = []
    for index, chunk in enumerate(chunks):
        local = {n['id'] for n in chunk}
        required = {v for n in chunk for v in n['operands'] if v not in local}
        # Preserve unused signature inputs and input-valued results explicitly.
        if index == 0: required.update(a['id'] for a in program['inputs'] if not uses[a['id']])
        out = [n['id'] for n in chunk if n['id'] in declared_outputs or
               any(owner[user] != index for user in uses[n['id']])]
        if index == count - 1:
            for o in program['outputs']:
                if o['value'] not in owner:
                    required.add(o['value'])
                    if o['value'] not in out: out.append(o['value'])
        fused = []
        if fuse:
            for n in chunk:
                if n['id'] in out or n['op'] not in INLINE: continue
                if n['op'] == 'constant' or (len(uses[n['id']]) == 1 and uses[n['id']] <= local):
                    fused.append(n['id'])
        materialized = [n['id'] for n in chunk if n['id'] not in fused]
        extra = 0
        for v in materialized:
            n = values[v]
            default = 'scalar.matmul.' + matmul_order if n['op'] == 'matmul' else 'scalar.elementwise'
            impl = implementations.resolve(requested.get(v, default), n, values, target)
            selected[v] = impl.id; versions[impl.id] = impl.version
            extra = max(extra, impl.scratch_bytes)
        inputs = [v['id'] for v in program['inputs'] + nodes if v['id'] in required]
        # Conservative sum, deliberately not a lifetime/physical-bank proof.
        scratch = sum((elements(values[v]['type']) * values[v]['type']['bits'] // 8 + 31) // 32 * 32
                      for v in materialized if v not in out)
        buffers = sum(transfer_elements(values[v]['type']) * values[v]['type']['bits'] // 8 * 2
                      for v in inputs + out)
        stack = max(8192, (scratch + extra + 2047) // 1024 * 1024)
        if stack + buffers > target.tile_memory_bytes:
            raise ValueError('conservative stack plus double-buffer memory exceeds tile budget')
        kernels.append({'id': f'k{index}', 'tile': [index % target.columns, 2 + index // target.columns],
            'nodes': [n['id'] for n in chunk], 'inputs': inputs, 'outputs': out,
            'fused': fused, 'materialized': materialized, 'scratch_bytes': scratch,
            'stack_bytes': stack, 'double_buffer_bytes': buffers, 'implementation_scratch_bytes': extra})
    if set(requested) - set(selected): raise ValueError('implementation choice targets a missing or fused node')
    # Bound recursive scalar expansion before emitting native source.
    expression_sizes = {}
    for kernel in kernels:
        fused = set(kernel['fused'])
        for v in kernel['nodes']:
            n = values[v]
            size = 1 + sum(expression_sizes[a] if a in fused else 1 for a in n['operands'])
            if size > 256: raise ValueError('scalar fusion expression budget exceeded')
            expression_sizes[v] = size
    return seal({'schema': 'aie-optimizer-program-plan-v1', 'contract_id': contract['id'],
        'program': program, 'rewrite_trace': trace, 'cuts': list(cuts), 'fuse': fuse,
        'matmul_order': matmul_order, 'kernels': kernels, 'planned_compute_tiles': count,
        'lowerings': selected, 'implementation_versions': versions,
        'implementation': implementation_identity(),
        'resources_scope': 'conservative local arrays and double-buffer extents; compiled mapping/banks unverified'})


def validate_plan(contract, plan, registry=None, implementation_registry=None):
    expected = make_plan(contract, trace=plan['rewrite_trace'], cuts=plan['cuts'], fuse=plan['fuse'],
                         matmul_order=plan['matmul_order'], registry=registry,
                         lowerings=plan['lowerings'], implementation_registry=implementation_registry)
    if plan != expected: raise ValueError('program plan fails graph/rule/deployment replay')
    return plan


def enumerate_plans(contract, *, max_plans=24, max_variants=6, max_depth=1, registry=None, evaluator=None,
                    max_implementations=4, implementation_registry=None):
    base, target = validate_contract(contract, registry)
    if type(max_plans) is not int or not 1 <= max_plans <= 256: raise ValueError('plan budget must be 1..256')
    variants, report = semantic_variants(base, max_variants, max_depth, registry)
    proposals = []
    # Give different semantic families a complete one-core lowering first.
    for program, trace in variants:
        choices = [(False, 'ijk'), (True, 'ijk')]
        if any(n['op'] == 'matmul' for n in live_nodes(program)): choices.append((True, 'ikj'))
        for fuse, order in choices:
            proposals.append((trace, (), fuse, order))
    for program, trace in variants:
        nodes = live_nodes(program)
        cuts = [i + 1 for i, n in enumerate(nodes[:-1]) if n['op'] not in ('constant', 'empty', 'splat')]
        if target.max_compute_tiles >= 2:
            for cut in cuts: proposals.append((trace, (cut,), True, 'ijk'))
        if target.max_compute_tiles >= 3:
            # A bounded three-region plan, not an exhaustive set partition.
            if len(cuts) >= 2: proposals.append((trace, (cuts[len(cuts)//3], cuts[2*len(cuts)//3]), True, 'ijk'))
    plans, refused, seen, equivalent = [], [], {}, []
    if type(max_implementations) is not int or not 1 <= max_implementations <= 16:
        raise ValueError('implementation budget must be 1..16')
    from .program_kernels import default_implementations
    implementations = implementation_registry or default_implementations()
    expanded = []
    for trace, cuts, fuse, order in proposals:
        expanded.append((trace, cuts, fuse, order, {}))
        # Keep scalar reference and fuse before introducing implementation alternatives.
        if not fuse or order != 'ijk' or max_implementations == 1: continue
        graph = replay(base, trace, registry); vals = {v['id']: v for v in graph['inputs'] + graph['nodes']}
        options = {n['id']: [name for name in implementations.options(n, vals, target) if implementations.entries[name].explore]
                   for n in live_nodes(graph) if n['op'] == 'matmul'}
        eligible = {v: names for v, names in options.items() if names}
        choices = []
        if eligible:
            choices.append({v: names[0] for v, names in eligible.items()})
            choices.append({v: names[-1] for v, names in eligible.items()})
            choices.extend({v: name} for v, names in eligible.items() for name in names)
        unique = set()
        for choice in choices:
            key = identity(choice)
            if key in unique: continue
            if len(unique) >= max_implementations - 1: break
            unique.add(key); expanded.append((trace, cuts, fuse, order, choice))
    for trace, cuts, fuse, order, lowerings in expanded:
        if len(plans) >= max_plans: break
        try: plan = make_plan(contract, trace=trace, cuts=cuts, fuse=fuse, matmul_order=order, registry=registry,
                             lowerings=lowerings, implementation_registry=implementations)
        except ValueError as exc:
            refused.append({'trace': trace, 'cuts': list(cuts), 'fuse': fuse, 'reason': str(exc)}); continue
        from .program_backend import emit_compute
        physical_key = identity({'compute': emit_compute(plan, implementations)[0], 'kernels': [
            {key: k[key] for key in ('id', 'tile', 'inputs', 'outputs', 'stack_bytes', 'double_buffer_bytes')}
            for k in plan['kernels']]})
        if physical_key in seen:
            equivalent.append({'proposal_id': plan['id'], 'retained_plan_id': seen[physical_key]}); continue
        seen[physical_key] = plan['id']; plans.append(plan)
    evaluations = [{'plan_id': p['id'], 'result': evaluator(p) if evaluator else
                    {'status': 'needs_evaluator', 'conditional_latency_ns': None, 'latency_ns': None}}
                   for p in plans]
    from .phase_costs import finite_score
    cohorts = defaultdict(list); unknown = []
    for plan, row in zip(plans, evaluations):
        result = row['result']; score = finite_score(result)
        if score is None: unknown.append(plan['id']); continue
        cohorts[(result['comparison_key'], result['scenario_id'])].append(
            {'plan_id': plan['id'], 'conditional_latency_ns': score, 'cores': plan['planned_compute_tiles']})
    rankings = []
    for (comparison, scenario), rows in cohorts.items():
        rows.sort(key=lambda r: (r['conditional_latency_ns'], r['cores'], r['plan_id']))
        frontier = [r for r in rows if not any(
            q['conditional_latency_ns'] <= r['conditional_latency_ns'] and q['cores'] <= r['cores'] and
            (q['conditional_latency_ns'] < r['conditional_latency_ns'] or q['cores'] < r['cores']) for q in rows)]
        rankings.append({'comparison_key': comparison, 'scenario_id': scenario, 'order': rows,
                         'conditional_frontier': frontier, 'native_admitted': False})
    # Keep unknown candidates; no structural score masquerades as latency.
    report.update(plans=len(plans), plan_budget=max_plans, proposal_count=len(expanded),
        implementation_catalog=implementations.catalog(),
        plan_budget_truncated=len(plans) >= max_plans and len(expanded) > len(plans),
        refused=refused, identical_lowerings=equivalent, evaluations=evaluations,
        conditional_rankings=rankings, unknown_candidates=unknown,
        reference_plan_id=plans[0]['id'] if plans else None,
        optimality='bounded semantic variants and contiguous topological partitions; no global optimum claim')
    return plans, report
