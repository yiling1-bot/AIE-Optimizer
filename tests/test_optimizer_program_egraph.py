import copy
from itertools import product
from math import prod
from pathlib import Path
import random
import tempfile
import unittest

from Optimizer.contracts import Target, identity
from Optimizer.importing.ir import GraphBuilder, evaluate, tensor_type
from Optimizer.program_ir import make_program, program_from_builder, program_from_import, seal, validate_contract
from Optimizer.program_rules import ProgramRuleRegistry
from Optimizer.program_equalities import default_tensor_equalities
from Optimizer.program_egraph import TensorEGraph, bind_extraction, check_extraction
from Optimizer.global_extract import Node, extract_dags
from Optimizer.program_planner import make_plan
from Optimizer.program_backend import generate, cpu_check, check_artifact
from Optimizer.program_costs import cost_request
from Optimizer.program_search import enumerate_egraph_plans

ROOT = Path(__file__).resolve().parents[1]


def factor_program(bits=16, narrow=False, multi=True):
    g = GraphBuilder(); typ = lambda shape: tensor_type(shape, bits)
    a = g.input('a', typ([2, 3]), 0)
    b = g.input('b', tensor_type([3, 2], 8 if narrow else bits), 1)
    c = g.input('c', tensor_type([3, 2], 8 if narrow else bits), 2)
    z0 = g.add('constant', [], typ([2, 2]), {'values': [7]}, {'site': 'z0'})
    z1 = g.add('constant', [], typ([2, 2]), {'values': [-3]}, {'site': 'z1'})
    left = g.add('matmul', [a, b, z0], typ([2, 2]), {}, {'site': 'left'})
    right = g.add('matmul', [a, c, z1], typ([2, 2]), {}, {'site': 'right'})
    y = g.add('add', [left, right], typ([2, 2]), {}, {'site': 'sum'})
    return program_from_builder(g, [y, left] if multi else [y])


def chain_program(bits=16, nonzero=False):
    g = GraphBuilder(); typ = lambda shape: tensor_type(shape, bits)
    a = g.input('a', typ([8, 2]), 0); b = g.input('b', typ([2, 8]), 1)
    c = g.input('c', typ([8, 2]), 2)
    z0 = g.add('constant', [], typ([8, 8]), {'values': [1 if nonzero else 0]}, {})
    ab = g.add('matmul', [a, b, z0], typ([8, 8]), {}, {})
    z1 = g.add('constant', [], typ([8, 2]), {'values': [0]}, {})
    y = g.add('matmul', [ab, c, z1], typ([8, 2]), {}, {})
    return program_from_builder(g, [y])


def inputs(program, rng):
    result = {}
    for a in program['inputs']:
        bits = a['type']['bits']; pool = [-(1 << (bits-1)), (1 << (bits-1))-1, -1, 0, 1, 5]
        result[a['id']] = [rng.choice(pool) for _ in range(prod(a['type']['shape']))]
    return result


def apply_root(graph, rid, root=0):
    term = graph.roots[root]
    step = next(s for s in graph.applications(term) if s['rule_id'] == rid)
    return graph.apply(term, step)


def selections(graph, **kwargs):
    result = []
    _, _, report = extract_dags(graph.nodes(), [graph.find(r) for r in graph.roots],
        max_states=10000, keep=1, visit_complete=lambda row: result.append(row), **kwargs)
    return result, report


class TensorEqualityTests(unittest.TestCase):
    def test_registered_algebra_rules_match_extreme_bitvector_reference(self):
        registry = default_tensor_equalities(); rng = random.Random(621)
        seen = set()
        for bits in (8, 16, 32):
            programs = [factor_program(bits), chain_program(bits)]
            g = GraphBuilder(); t = tensor_type([3], bits)
            a = g.input('a', t, 0); b = g.input('b', t, 1); c = g.input('c', t, 2)
            x = g.add('mul', [a, b], t, {}, {}); y = g.add('mul', [x, c], t, {}, {})
            p = g.add('transpose', [y], t, {'permutation': [0]}, {})
            q = g.add('transpose', [p], t, {'permutation': [0]}, {})
            programs.append(program_from_builder(g, [q]))
            for program in programs:
                for step in registry.applications(program):
                    changed = registry.apply(program, step); seen.add(step['rule_id'])
                    for _ in range(5):
                        batch = inputs(program, rng)
                        self.assertEqual(evaluate(program, batch), evaluate(changed, batch), step)
        self.assertEqual(seen, {r['id'] for r in registry.catalog()})

    def test_no_factorization_across_narrow_signed_multiplicands(self):
        p = factor_program(32, narrow=True)
        self.assertNotIn('tensor.matmul.factor_left', {s['rule_id'] for s in default_tensor_equalities().applications(p)})

    def test_no_reassociation_across_nonzero_initializer_or_conversion(self):
        p = chain_program(nonzero=True)
        self.assertNotIn('tensor.matmul.associate', {s['rule_id'] for s in default_tensor_equalities().applications(p)})
        g = GraphBuilder(); t8 = tensor_type([2, 2], 8); t32 = tensor_type([2, 2], 32)
        a = g.input('a', t8, 0); b = g.input('b', t8, 1)
        zero = g.add('constant', [], t32, {'values': [0]}, {})
        m = g.add('matmul', [a, b, zero], t32, {}, {})
        narrow = g.add('truncate', [m], t8, {}, {})
        y = g.add('matmul', [narrow, b, zero], t32, {}, {})
        p = program_from_builder(g, [y])
        self.assertNotIn('tensor.matmul.associate', {s['rule_id'] for s in default_tensor_equalities().applications(p)})


class TensorEGraphTests(unittest.TestCase):
    def test_shared_multiple_outputs_reverse_local_extraction_preference(self):
        p = factor_program(); graph = TensorEGraph(p)
        apply_root(graph, 'tensor.matmul.factor_left')
        costs = lambda n: 10 if n['op'] == 'matmul' else 1 if n['op'] == 'add' else 0
        nodes = graph.nodes(costs); root, shared = map(graph.find, graph.roots)
        local = extract_dags(nodes, [root], keep=1)[0][0]
        joint = extract_dags(nodes, [root, shared], keep=1)[0][0]
        self.assertEqual(local['metrics']['cost'], 12)
        self.assertEqual(joint['metrics']['cost'], 21)
        program = graph.materialize(joint['selected'])
        self.assertEqual(sum(n['op'] == 'matmul' for n in program['nodes']), 2)
        self.assertIn(program['outputs'][1]['value'], program['nodes'][-1]['operands'])
        # Independent tiny-space exhaustive oracle, with each demanded node paid once.
        by_class = {c: [n for n in nodes if n.eclass == c] for c in graph.classes()}
        brute = []
        for options in product(*by_class.values()):
            chosen = dict(zip(by_class, options)); seen = set(); active = set()
            def visit(c):
                if c in active: raise ValueError('cycle')
                if c in seen: return 0
                active.add(c); n = chosen[c]; total = n.cost + sum(visit(v) for v in n.children)
                active.remove(c); seen.add(c); return total
            try: brute.append(sum(visit(c) for c in (root, shared)))
            except ValueError: pass
        self.assertEqual(joint['metrics']['cost'], min(brute))

    def test_congruence_rebuild_merges_users_and_origins_do_not_block_sharing(self):
        b = GraphBuilder(); t = tensor_type([4], 16)
        x = b.input('x', t, 0); y = b.input('y', t, 1)
        a = b.add('add', [x, y], t, {}, {'line': 1})
        duplicate = b.add('add', [x, y], t, {}, {'line': 77})
        reverse = b.add('add', [y, x], t, {}, {})
        left = b.add('mul', [a, x], t, {}, {})
        right = b.add('mul', [reverse, x], t, {}, {})
        graph = TensorEGraph(program_from_builder(b, [left, right, duplicate]))
        # The raw original duplicate add is already hash-consed.
        self.assertEqual(sum(n['op'] == 'add' for n in graph.terms.values()), 2)
        add = next(tid for tid,n in graph.terms.items() if n['op'] == 'add')
        graph.apply(add, graph.applications(add)[0])
        self.assertEqual(graph.find(graph.roots[0]), graph.find(graph.roots[1]))
        self.assertGreater(graph.congruence_merges, 0)
        rows, _ = selections(graph)
        for row in rows:
            p = graph.materialize(row)
            self.assertEqual(p['outputs'][0]['value'], p['outputs'][1]['value'])

    def test_cycles_are_refused_while_input_valued_duplicate_outputs_survive(self):
        b = GraphBuilder(); t = tensor_type([2, 3], 16)
        x = b.input('x', t, 0); b.input('unused', tensor_type([], 8), 1)
        a = b.add('transpose', [x], tensor_type([3, 2], 16), {'permutation': [1, 0]}, {})
        y = b.add('transpose', [a], t, {'permutation': [1, 0]}, {})
        p = program_from_builder(b, [y, x, y]); graph = TensorEGraph(p)
        apply_root(graph, 'tensor.transpose.compose')
        rows, report = selections(graph)
        self.assertGreater(report['cycle_refusals'], 0)
        self.assertEqual(len(rows), 1)
        out = graph.materialize(rows[0]); self.assertEqual(len(out['inputs']), 2)
        self.assertEqual(out['nodes'], [])
        self.assertEqual([o['value'] for o in out['outputs']], ['arg0']*3)

    def test_certificate_replays_shared_choices_and_refuses_resealed_forgeries(self):
        p = factor_program(); g = TensorEGraph(p); apply_root(g, 'tensor.matmul.factor_left')
        row = selections(g)[0][0]; cert = g.certificate(row)
        self.assertEqual(check_extraction(cert, p), g.materialize(row))
        for mutation in ('base', 'roots', 'version', 'program', 'unused', 'catalog', 'engine'):
            bad = copy.deepcopy(cert)
            if mutation == 'base': bad['base_program_id'] = 'wrong'
            if mutation == 'roots': bad['roots'].reverse()
            if mutation == 'version': bad['steps'][0]['rewrite']['version'] = 999
            if mutation == 'program': bad['program_id'] = 'wrong'
            if mutation == 'unused': bad['selected']['unused'] = next(iter(g.terms))
            if mutation == 'catalog': bad['rules'][0]['proof'] = 'claimed'
            if mutation == 'engine': bad['implementation'] = 'wrong'
            bad = seal({k:v for k,v in bad.items() if k != 'id'})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): check_extraction(bad, p)

    def test_new_rule_registration_uses_same_certificate_and_backend_interfaces(self):
        registry = ProgramRuleRegistry()
        def matches(p):
            for n in p['nodes']:
                if n['op'] == 'transpose' and n['attributes']['permutation'] == [0]:
                    yield {'node_id': n['id']}
        def rewrite(p, step):
            g = GraphBuilder(); arg = p['inputs'][0]; x = g.input(arg['name'], arg['type'], 0)
            return program_from_builder(g, [x], p['entry'])
        registry.register('extension.identity_transpose', 1, matches, rewrite, 'rank-one transpose is identity')
        b = GraphBuilder(); t = tensor_type([2], 16); x = b.input('x', t, 0)
        y = b.add('transpose', [x], t, {'permutation': [0]}, {})
        c = make_program(program_from_builder(b, [y])); graph = TensorEGraph(c['program'], registry)
        apply_root(graph, 'extension.identity_transpose'); row = selections(graph)[0][0]
        derived = bind_extraction(c, graph, row); validate_contract(derived, registry)
        make_plan(derived, registry=registry)
        with self.assertRaises(ValueError): validate_contract(derived)

    def test_deep_ssa_and_shared_dag_do_not_depend_on_python_recursion(self):
        b = GraphBuilder(); t = tensor_type([1], 8); x = b.input('x', t, 0)
        one = b.add('constant', [], t, {'values': [1]}, {})
        for _ in range(1050): x = b.add('add', [x, one], t, {}, {})
        p = program_from_builder(b, [x]); g = TensorEGraph(p)
        rows, report = selections(g)
        self.assertTrue(report['search_exhausted']); self.assertEqual(len(rows), 1)
        self.assertEqual(evaluate(p, {'arg0': [127]}), evaluate(g.materialize(rows[0]), {'arg0': [127]}))

    def test_term_and_extraction_budgets_are_explicit(self):
        p = chain_program(); g = TensorEGraph(p); before = len(g.terms)
        term = g.roots[0]; step = next(s for s in g.applications(term) if s['rule_id'] == 'tensor.matmul.associate')
        with self.assertRaises(ValueError): g.apply(term, step, max_terms=before)
        self.assertEqual(len(g.terms), before); self.assertEqual(g.steps, [])
        g.apply(term, step)
        _, _, report = extract_dags(g.nodes(), [g.find(term)], max_complete=1)
        self.assertTrue(report['complete_budget_reached']); self.assertFalse(report['search_exhausted'])
        with self.assertRaises(ValueError): extract_dags(g.nodes(), [g.find(term)], max_complete=0)

    def test_diversity_witness_preserves_source_elsewhere_and_visitors_can_bound_admission(self):
        p = factor_program(); g = TensorEGraph(p)
        g.saturate(max_rounds=1,max_applications=8)
        rid = 'tensor.matmul.factor_left'
        rhs = next(rhs for step,rhs in zip(g.steps,g.step_results) if step['rewrite']['rule_id']==rid)
        source = g.materialize(g.witness_selection())
        factored = g.materialize(g.witness_selection(rhs))
        self.assertEqual(sum(n['op']=='matmul' for n in source['nodes']),2)
        self.assertEqual(sum(n['op']=='matmul' for n in factored['nodes']),2)
        self.assertEqual(sum(n['op']=='add' for n in factored['nodes']),2)
        nodes = [Node('first','A',(),1),Node('second','A',(),2)]
        rows=[]
        def visit(row):
            if rows:return False
            rows.append(row)
        _,_,report=extract_dags(nodes,['A'],keep=2,visit_complete=visit)
        self.assertTrue(report['visitor_budget_reached']);self.assertFalse(report['search_exhausted'])
        self.assertEqual(report['complete_assignments'],1);self.assertEqual(report['unexplored_states'],1)

    def test_extraction_traversal_preference_does_not_change_cost_optimum(self):
        nodes=[Node('cheap','A',(),1),Node('expensive','A',(),20)]
        best,_,report=extract_dags(nodes,['A'],keep=1,option_priority=lambda n:0 if n.id=='expensive' else 1)
        self.assertEqual(best[0]['metrics']['cost'],1);self.assertTrue(report['search_exhausted'])


class CertifiedSearchTests(unittest.TestCase):
    def test_rule_family_seeding_exposes_matrix_reassociation_under_small_budget(self):
        c=make_program(chain_program(),Target(max_compute_tiles=1))
        entries,report=enumerate_egraph_plans(c,max_rounds=1,max_applications=8,max_dags=6,
            max_plans=12,max_plans_per_dag=3,max_implementations=1)
        work=[]
        for e in entries:
            p=e['plan']['program'];values={v['id']:v for v in p['inputs']+p['nodes']}
            work.append(sum(prod(n['type']['shape'])*values[n['operands'][0]]['type']['shape'][1]
                for n in p['nodes'] if n['op']=='matmul'))
        self.assertIn(256,work);self.assertIn(64,work)
        self.assertEqual(report['structural_latency_prunes'],0)

    def test_joint_search_retains_unknown_and_compares_only_compatible_cost_scenarios(self):
        c = make_program(factor_program(), Target(max_compute_tiles=2))
        received = []
        def evaluator(contract, plan):
            validate_contract(contract); received.append(plan['id'])
            request = cost_request(contract, plan)
            if plan['fuse']:
                return {'plan_id': plan['id'], 'status': 'needs_measurement', 'conditional_latency_ns': None}
            # Explicit software-only ranking witness: use different cohort for 2 cores.
            return {'plan_id': plan['id'], 'status': 'conditional_estimate', 'conditional_latency_ns': 300,
                'comparison_key': request['comparison_key'], 'scenario_id': 'software_'+str(plan['planned_compute_tiles']),
                'latency_ns': None, 'native_admitted': False}
        entries, report = enumerate_egraph_plans(c, evaluator=evaluator, max_rounds=1, max_applications=4,
            max_dags=4, max_plans=20, max_plans_per_dag=6, max_implementations=1)
        self.assertEqual(len(received), len(entries)); self.assertTrue(report['unknown_plan_ids'])
        self.assertEqual(report['structural_latency_prunes'], 0)
        keys = {cost_request(e['contract'], e['plan'])['comparison_key'] for e in entries}
        self.assertEqual(len(keys), 1)
        self.assertEqual(entries[0]['plan']['id'], report['reference_plan_id'])
        self.assertTrue(any(e['contract']['source']['kind'] == 'verified_egraph_extraction' for e in entries))
        self.assertFalse(report['native_admitted'])

    def test_latency_callback_can_prefer_the_more_expensive_arithmetic_dag(self):
        c = make_program(chain_program(), Target(max_compute_tiles=1))
        registry = ProgramRuleRegistry(); default = default_tensor_equalities().rules['tensor.matmul.associate']
        registry.register('tensor.matmul.associate', **{k:default[k] for k in ('version', 'matches', 'rewrite', 'proof')})
        # Custom registry remains explicitly trusted and passed through all validation.
        def evaluator(contract, plan):
            validate_contract(contract, registry)
            vals = {v['id']:v for v in plan['program']['inputs']+plan['program']['nodes']}
            macs = sum(prod(n['type']['shape'])*vals[n['operands'][0]]['type']['shape'][1]
                       for n in plan['program']['nodes'] if n['op'] == 'matmul')
            return {'plan_id':plan['id'], 'status':'conditional_estimate', 'latency_ns':None,
                'conditional_latency_ns':100 if macs == 256 else 1000,
                'comparison_key':'software_fixed_workload', 'scenario_id':'software_only', 'native_admitted':False}
        entries, report = enumerate_egraph_plans(c, registry=registry, evaluator=evaluator,
            max_rounds=1,max_applications=2,max_dags=8,max_plans=24,max_plans_per_dag=4,max_implementations=1)
        ranking = report['conditional_rankings'][0]['ranked']
        self.assertEqual(ranking[0]['conditional_objective_ns'], 100)
        self.assertTrue(any(r['conditional_objective_ns'] == 1000 for r in ranking))

    def test_extracted_factor_and_reassociated_programs_generate_against_original_oracle(self):
        rng = random.Random(83)
        for p, rid in ((factor_program(multi=False), 'tensor.matmul.factor_left'),
                       (chain_program(), 'tensor.matmul.associate')):
            g = TensorEGraph(p); apply_root(g, rid); rows, _ = selections(g)
            for row in rows:
                derived = bind_extraction(make_program(p), g, row); plan = make_plan(derived, fuse=True)
                for _ in range(3):
                    batch = inputs(p, rng); self.assertEqual(evaluate(p,batch), evaluate(derived['program'],batch))
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)/'candidate'; generate(derived,plan,root,frames=2)
                    self.assertEqual(cpu_check(root)['status'],'passed')
                    self.assertTrue((root/'equivalence.json').is_file())
                    self.assertEqual(check_artifact(root)[1]['oracle']['original_contract_id'],make_program(p)['id'])

    def test_imported_cpp_keeps_original_host_oracle_after_extraction(self):
        from Optimizer.importing.cpp_frontend import import_cpp
        path = ROOT/'tests/fixtures/optimizer_program/residual.cpp'
        original = program_from_import(import_cpp(path,'residual',project_root=path.parent))
        graph = TensorEGraph(original['program']); graph.saturate(max_rounds=1,max_applications=2)
        derived = bind_extraction(original,graph,selections(graph,max_complete=1)[0][0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'cpp'; generate(derived,make_plan(derived,fuse=True),root,frames=2)
            result = cpu_check(root)
            self.assertEqual(result['status'],'passed'); self.assertTrue(result['original_source_checked'])
            self.assertTrue((root/'source_import.json').is_file())

    def test_forged_extracted_contract_is_rejected_before_generation(self):
        p = factor_program(); graph = TensorEGraph(p); row = selections(graph)[0][0]
        derived = bind_extraction(make_program(p),graph,row)
        bad = copy.deepcopy(derived); bad['source']['certificate']['program_id'] = 'forged'
        bad['source']['certificate'] = seal({k:v for k,v in bad['source']['certificate'].items() if k != 'id'})
        bad = seal({k:v for k,v in bad.items() if k != 'id'})
        with self.assertRaises(ValueError): make_plan(bad)


if __name__ == '__main__': unittest.main()
