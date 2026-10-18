"""Joint tensor e-DAG, registered implementation and core-plan exploration.

Every visited complete DAG reaches deployment enumeration before latency is
ranked. All semantic and physical budgets are explicit; no MAC/arena top-k is
used as a latency filter. Evaluators receive the certified contract AND plan.
"""
from collections import defaultdict

from .contracts import identity
from .global_extract import extract_dags
from .phase_costs import finite_score
from .program_egraph import TensorEGraph, bind_extraction
from .program_ir import seal, validate_contract
from .program_planner import enumerate_plans, make_plan


def _physical_key(plan, implementations=None):
    from .program_backend import emit_compute
    return identity({'compute': emit_compute(plan, implementations)[0],
        'outputs': plan['program']['outputs'], 'inputs': plan['program']['inputs'],
        'kernels': [{key:k[key] for key in ('id', 'tile', 'inputs', 'outputs', 'stack_bytes', 'double_buffer_bytes')}
                    for k in plan['kernels']]})


def enumerate_egraph_plans(contract, *, evaluator=None, registry=None, implementation_registry=None,
        max_rounds=2, max_applications=24, max_terms=1200, max_states=10000, max_dags=16,
        max_plans=64, max_plans_per_dag=16, max_implementations=4):
    base, _ = validate_contract(contract)
    if contract['source']['kind'] == 'verified_egraph_extraction':
        raise ValueError('search must start at the original contract')
    for name, value, cap in (('max_dags', max_dags, 256), ('max_plans', max_plans, 256),
            ('max_plans_per_dag', max_plans_per_dag, 256), ('max_states', max_states, 100000)):
        if type(value) is not int or not 1 <= value <= cap: raise ValueError(name + ' outside finite search budget')
    graph = TensorEGraph(base, registry)
    saturation = graph.saturate(max_rounds=max_rounds, max_applications=max_applications, max_terms=max_terms)
    selections = []; selected_ids = set(); seeds = []
    def collect(selected):
        selected = graph.canonical_selection(selected)
        key = identity(selected)
        if key in selected_ids: return True
        if len(selections) >= max_dags: return False
        selected_ids.add(key); selections.append(selected); return True
    initial = graph.witness_selection(max_states=max_states)
    if initial is not None: collect(initial)
    # One witness per rule family before secondary witnesses from a family.
    # This addresses DFS partition explosion without using MACs as latency.
    by_rule = defaultdict(list)
    for step,rhs in zip(graph.steps, graph.step_results): by_rule[step['rewrite']['rule_id']].append(rhs)
    for offset in range(max((len(v) for v in by_rule.values()), default=0)):
        for rid, terms in by_rule.items():
            if offset >= len(terms) or len(selections) >= max_dags: continue
            selection = graph.witness_selection(terms[offset], max_states=max_states)
            admitted = selection is not None and collect(selection)
            seeds.append({'rule_id': rid, 'term_id': terms[offset], 'finite_witness_found': selection is not None,
                          'admitted': admitted,
                          'dag_id':identity(graph.canonical_selection(selection)) if selection is not None else None})
    def visitor(selected): return collect(selected)
    _, _, extraction = extract_dags(graph.nodes(), [graph.find(r) for r in graph.roots],
        max_states=max_states, keep=1, visit_complete=visitor)
    # Include the original source contract independently of the finite extractor
    # traversal. A rewrite cycle or budget cannot evict the executable reference.
    reference = make_plan(contract, implementation_registry=implementation_registry)
    entries = [{'contract': contract, 'plan': reference, 'dag_id': 'original'}]
    physical = {_physical_key(reference, implementation_registry): reference['id']}
    families, family_reports = [], []
    for selected in selections:
        derived = bind_extraction(contract, graph, selected)
        plans, report = enumerate_plans(derived, max_plans=max_plans_per_dag, max_variants=1,
            max_depth=0, registry=graph.registry, max_implementations=max_implementations,
            implementation_registry=implementation_registry)
        dag_id = identity(selected)
        families.append([{'contract': derived, 'plan': p, 'dag_id': dag_id} for p in plans])
        family_reports.append({'dag_id': dag_id, 'certificate_id': derived['source']['certificate']['id'],
            'program_id': derived['program']['id'], 'plans': len(plans),
            'deployment_budget_truncated': report['plan_budget_truncated'], 'refused': report['refused']})
    duplicates = []; proposals = sum(len(f) for f in families); considered = 0
    # Round-robin admission gives every visited structural family a first plan
    # before a family consumes the remaining physical budget.
    for index in range(max((len(f) for f in families), default=0)):
        for family in families:
            if index >= len(family): continue
            entry = family[index]; considered += 1; key = _physical_key(entry['plan'], implementation_registry)
            if key in physical:
                duplicates.append({'plan_id': entry['plan']['id'], 'retained_plan_id': physical[key]}); continue
            if len(entries) >= max_plans: continue
            physical[key] = entry['plan']['id']; entries.append(entry)
    evaluations = {}; cohorts = defaultdict(list); unknown = []
    for entry in entries:
        plan = entry['plan']
        result = evaluator(entry['contract'], plan) if evaluator is not None else {
            'plan_id': plan['id'], 'status': 'needs_evaluator', 'conditional_latency_ns': None,
            'latency_ns': None, 'native_admitted': False}
        if type(result) is not dict or result.get('plan_id') != plan['id']:
            raise ValueError('evaluation belongs to another plan')
        score = finite_score(result)
        evaluations[plan['id']] = result
        if score is None: unknown.append(plan['id']); continue
        cohorts[(result['comparison_key'], result['scenario_id'])].append(plan)
    from .joint_extract import rank_plans
    rankings = [rank_plans(plans, evaluations) for plans in cohorts.values()]
    report = seal({'schema': 'aie-optimizer-program-egraph-space-v1',
        'original_contract_id': contract['id'], 'reference_plan_id': reference['id'],
        'saturation': saturation, 'dag_extraction': extraction, 'dag_families': family_reports,
        'max_dags': max_dags, 'diversity_seeds': seeds, 'retained_dag_assignments': len(selections),
        'plans_evaluated': len(entries), 'max_plans': max_plans, 'max_plans_per_dag': max_plans_per_dag,
        'physical_proposals': proposals, 'physical_proposals_considered': considered,
        'physical_budget_omitted': proposals - len(duplicates) - (len(entries)-1),
        'identical_generated_programs': duplicates, 'evaluations': evaluations,
        'conditional_rankings': rankings, 'unknown_plan_ids': unknown,
        'structural_latency_prunes': 0, 'latency_ns': None, 'native_admitted': False,
        'scope': 'bounded shared multi-output DAG choices and contiguous topological core partitions; '
                 'one representative per e-class; no arbitrary cloning, placement/routing or native optimality claim'})
    return entries, report
