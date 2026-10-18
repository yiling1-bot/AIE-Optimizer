"""Whole-program first-frame cost requests using Analyzer's pinned registry.

This adapter has no training/observation input. Generated source fingerprints
are not compiled-cost admission; all returned times remain conditional.
"""
from dataclasses import asdict
from hashlib import sha256

from .contracts import identity
from .program_ir import seal, transfer_elements, original_contract
from .program_planner import validate_plan
from .analyzer_bridge import (SIGNATURE_SCHEMA, ClockDomain, ClockedCausalGraph,
    CausalRule, CausalParent, DeclaredEvent, diagnose_clocked_graph,
    resolve_program_costs, load_cost_family, dependency_hashes)


def cost_request(contract, plan, *, core_period_ps=800, rewrite_registry=None, implementation_registry=None):
    validate_plan(contract, plan, rewrite_registry, implementation_registry)
    if type(core_period_ps) is not int or core_period_ps <= 0: raise ValueError('explicit positive core period required')
    plio_hz = contract['target']['plio_hz']
    if 10**12 % plio_hz: raise ValueError('integer picosecond PLIO clock required')
    from .program_backend import emit_compute
    _, bodies = emit_compute(plan, implementation_registry)
    program = plan['program']; values = {v['id']: v for v in program['inputs'] + program['nodes']}
    components, roots, rules, events = {}, [DeclaredEvent('request', 0)], [], {}
    producers = {}
    clocks = {'core': core_period_ps, 'transfer': 10**12 // plio_hz}
    def component(kind, operations, interactions, body=None):
        facts = {'architecture': contract['target']['architecture'], 'clocks': clocks,
            'units': {'duration': 'ps'}, 'boundaries': {'kind': kind,
                'service': 'declared active service; causal waits are separate', 'frame_scope': 'first_frame_only'},
            'operations': operations, 'interactions': interactions,
            'compile_identity': identity({'generator': plan['implementation'], 'toolchain': contract['target']['toolchain'],
                'body_sha256': sha256(body.encode()).hexdigest() if body else None,
                'operations': operations, 'interactions': interactions,
                'evidence_level': 'generated_source_only; compiled_context_not_recovered'})}
        cid = identity({'mechanism': kind, **facts})
        components[cid] = {'schema': SIGNATURE_SCHEMA, 'component_id': cid, 'mechanism': kind, **facts}
        return cid
    def event(name, parents, clock='core', cost=None):
        parents = list(dict.fromkeys(parents))
        rules.append(CausalRule(name, ClockDomain(clocks[clock]),
                     tuple(CausalParent(p, 0, cost) for p in parents), name))
        return name
    def transfer(name, parent, typ, direction):
        cid = component('program_frame_transfer', {'bytes': transfer_elements(typ) * typ['bits'] // 8,
            'type': typ, 'transport': 'typed_padded_frame'},
            {'direction': direction, 'arbitration': 'declared_independent_logical_link; native_channel_unknown'})
        return event(name, [parent], 'transfer', cid)
    for i, arg in enumerate(program['inputs']):
        events[arg['id']] = transfer(f'input:{i}:ready', 'request', arg['type'], 'external_to_kernel')
    for kernel in plan['kernels']:
        kid = kernel['id']
        ready = []
        for v in kernel['inputs']:
            parent = events[v]
            if v in producers:
                parent = transfer(f'{producers[v]}:{v}:to:{kid}', parent, values[v]['type'], 'kernel_to_kernel')
            ready.append(parent)
        ports = {'inputs': [values[v]['type'] for v in kernel['inputs']],
                 'outputs': [values[v]['type'] for v in kernel['outputs']],
                 'ownership': 'first use only; implicit whole-buffer wrapper acquire/release',
                 'placement': kernel['tile']}
        startup = event(kid + ':startup', ['request'], cost=component('program_actor_startup', {}, ports))
        acquired = event(kid + ':acquire', [startup] + ready,
                         cost=component('program_wrapper_acquire', {}, ports))
        region = [{'op': values[v]['op'], 'type': values[v]['type'], 'attributes': values[v]['attributes'],
                   'operand_types': [values[a]['type'] for a in values[v]['operands']],
                   'materialized': v in kernel['materialized'],
                   'implementation': plan['lowerings'].get(v),
                   'implementation_version': plan['implementation_versions'].get(plan['lowerings'].get(v))} for v in kernel['nodes']]
        work = {'operators': region, 'scalar_fusion': bool(kernel['fused']),
                'matmul_order': plan['matmul_order'] if any(n['op'] == 'matmul' for n in region) else None,
                'scratch_bytes': kernel['scratch_bytes'], 'implementation_scratch_bytes': kernel['implementation_scratch_bytes'],
                'lowering': 'registered_operator_implementations'}
        done = event(kid + ':compute', [acquired], cost=component('program_complete_region', work, ports, bodies[kid]))
        released = event(kid + ':release', [done], cost=component('program_wrapper_release', {}, ports))
        for v in kernel['outputs']:
            events[v] = released; producers[v] = kid
    output_events = []
    for i, out in enumerate(program['outputs']):
        output_events.append(transfer(f'output:{i}:complete', events[out['value']], out['type'], 'kernel_to_external'))
    complete = event('all_outputs:complete', output_events)
    graph = ClockedCausalGraph(tuple(roots), tuple(rules), tuple(components),
                              ('program:' + contract['id'], 'plan:' + plan['id']))
    request = {'schema': 'aie-optimizer-program-cost-request-v1', 'contract_id': contract['id'], 'plan_id': plan['id'],
        'components': list(components.values()), 'graph': asdict(graph), 'output_event': complete,
        'planned_compute_tiles': plan['planned_compute_tiles'],
        'comparison_key': identity({'contract': original_contract(contract)['id'], 'clocks': clocks, 'frame': 0,
                                    'metric': 'request_origin_to_all_complete_outputs'}),
        'dependency_hashes': dependency_hashes(),
        'adapter_sha256': sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(), 'native_admitted': False,
        'scope': 'first frame; all source requests at zero; declared independent links; no native resource/protocol admission'}
    return seal(request)


def evaluate_costs(contract, plan, registry, selection, *, core_period_ps=800, rewrite_registry=None):
    if set(selection) != {'cost_by_component', 'member_by_group'}:
        raise ValueError('explicit family and correlation selections required')
    request = cost_request(contract, plan, core_period_ps=core_period_ps, rewrite_registry=rewrite_registry)
    resolution = resolve_program_costs(request['components'], registry)
    result = {'schema': 'aie-optimizer-program-cost-result-v1', 'plan_id': plan['id'], 'request_id': request['id'],
        'status': resolution['status'], 'resolution': resolution, 'conditional_latency_ns': None, 'latency_ns': None,
        'comparison_key': request['comparison_key'], 'scenario_id': identity({'registry': registry, 'members': selection['member_by_group']}),
        'native_admitted': False, 'scope': request['scope']}
    if resolution['status'] != 'reused': return result
    assignment, selected = {}, []
    for row in resolution['components']:
        cid = row['component_id']; options = row['compatible_costs']
        cost_id = selection['cost_by_component'].get(cid)
        if cost_id is None and len(options) == 1: cost_id = options[0]['cost_id']
        if cost_id is None: return {**result, 'status': 'needs_model_selection'}
        family = load_cost_family(row, cost_id)
        if family['referenced_documents']: raise ValueError('mechanism-specific cost adapter required')
        member = selection['member_by_group'].get(family['correlation_group'])
        matching = [m for m in family['inline_models'] if m['model_id'] == member]
        if len(matching) != 1: return {**result, 'status': 'needs_model_selection'}
        params = matching[0]['parameters']
        if set(params) != {'duration_ps'} or type(params['duration_ps']) is not int or params['duration_ps'] < 0:
            raise ValueError('exact nonnegative active service duration required; endpoint observations refused')
        assignment[cid] = params['duration_ps']
        selected.append({'component': cid, 'family': cost_id, 'member': member, 'origin': family['origin']})
    raw = request['graph']
    graph = ClockedCausalGraph(tuple(DeclaredEvent(**e) for e in raw['declared_events']),
        tuple(CausalRule(r['id'], ClockDomain(**r['clock']), tuple(CausalParent(**p) for p in r['parents']),
                        r['observation_semantics']) for r in raw['rules']), tuple(raw['cost_names']), tuple(raw['evidence']))
    timing = diagnose_clocked_graph(graph, assignment)
    return {**result, 'status': 'conditional_estimate', 'selected_costs': selected, 'timing': timing,
            'conditional_latency_ns': timing['times_ps'][request['output_event']] / 1000,
            'limitations': ['no native accuracy/ranking admission', 'no repeated-frame slot reuse or arbitration recovery']}


class ProgramEvaluator:
    def __init__(self, contract, registry, selection):
        self.contract, self.registry, self.selection = contract, registry, selection

    def __call__(self, plan): return evaluate_costs(self.contract, plan, self.registry, self.selection)
