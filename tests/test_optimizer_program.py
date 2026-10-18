import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

from Optimizer.contracts import Target, identity
from Optimizer.importing.ir import GraphBuilder, evaluate, tensor_type
from Optimizer.program_ir import (make_program, program_from_builder, program_from_import,
                                  validate_program, validate_contract, transfer_elements)
from Optimizer.program_rules import default_program_rules, replay
from Optimizer.program_planner import enumerate_plans, make_plan, validate_plan
from Optimizer.program_backend import generate, cpu_check, check_artifact
from Optimizer.program_costs import cost_request, evaluate_costs
from Optimizer.program_native import parse_typed_output

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/optimizer_program'


def declared_matmul(bits=32, init=7):
    b = GraphBuilder(); a = b.input('samples', tensor_type([3, 5], bits), 0)
    w = b.input('weights', tensor_type([5, 2], bits), 1)
    z = b.add('constant', [], tensor_type([3, 2], bits), {'values': [init]}, {})
    y = b.add('matmul', [a, w, z], tensor_type([3, 2], bits), {}, {})
    return program_from_builder(b, [y])


def software_registry(request, duration=800):
    from Optimizer.analyzer_bridge import DIMENSIONS, REGISTRY_SCHEMA, FAMILY_SCHEMA
    return {'schema': REGISTRY_SCHEMA, 'registry_id': 'software-test-only',
        'supported_mechanisms': sorted({c['mechanism'] for c in request['components']}),
        'entries': [{'cost_id': 'cost_' + c['component_id'], 'mechanism': c['mechanism'],
            'applicability': {d: {'kind': 'exact', 'value': c[d]} for d in DIMENSIONS},
            'cost_family': {'schema': FAMILY_SCHEMA, 'origin': 'software_conditional', 'model_refs': [],
                'inline_models': [{'model_id': 'declared', 'parameters': {'duration_ps': duration}}],
                'correlation_group': 'fixture'}, 'evidence_refs': [],
            'discrepancy': {'status': 'uncalibrated', 'bound_ps': None, 'evidence_refs': []}}
                    for c in request['components']]}


class ProgramCoreTests(unittest.TestCase):
    def test_modular_partitions_preserve_nonzero_initializer_and_extremes(self):
        rng = random.Random(79)
        for bits in (8, 16, 32):
            p = declared_matmul(bits)
            for step in default_program_rules().applications(p):
                changed = replay(p, [step]); self.assertEqual(validate_program(changed), changed)
                for _ in range(8):
                    pool = [-(1 << (bits-1)), (1 << (bits-1))-1, -1, 0, 1, 4]
                    inputs = {a['id']: [rng.choice(pool) for _ in range(__import__('math').prod(a['type']['shape']))]
                              for a in p['inputs']}
                    self.assertEqual(evaluate(p, inputs), evaluate(changed, inputs))

    def test_composed_semantic_rules_replay_on_generated_graph(self):
        p = declared_matmul()
        first = next(default_program_rules().applications(p)); q = replay(p, [first])
        second = next(default_program_rules().applications(q))
        r = replay(p, [first, second]); validate_program(r)
        self.assertGreater(len(r['nodes']), len(q['nodes']))

    def test_rule_guards_and_tampered_certificate_rejected(self):
        p = declared_matmul(); step = next(default_program_rules().applications(p))
        for field, value in (('split', 0), ('node_id', 'v999'), ('version', 999)):
            with self.assertRaises(ValueError): replay(p, [{**step, field: value}])

    def test_forged_type_range_uninitialized_output_and_unknown_op_rejected(self):
        p = declared_matmul()
        for kind in ('range', 'op', 'output', 'operand', 'attrs'):
            q = copy.deepcopy(p)
            if kind == 'range': q['nodes'][-1]['range'] = [0, 1]
            if kind == 'op': q['nodes'][-1]['op'] = 'convolution_magic'
            if kind == 'output': q['outputs'][0]['type'] = tensor_type([1], 8)
            if kind == 'operand': q['nodes'][-1]['operands'][0] = 'future'
            if kind == 'attrs': q['nodes'][-1]['attributes']['approximate'] = True
            q['id'] = identity({k:v for k,v in q.items() if k != 'id'})
            with self.assertRaises(ValueError): validate_program(q)

    def test_signature_direct_outputs_duplicate_results_and_unused_input(self):
        b = GraphBuilder(); x = b.input('x', tensor_type([3], 16), 0)
        b.input('unused', tensor_type([], 8), 1)
        c = make_program(program_from_builder(b, [x, x])); p = make_plan(c, fuse=True)
        self.assertEqual(len(p['kernels'][0]['inputs']), 2)
        self.assertEqual(p['kernels'][0]['outputs'], ['arg0'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'passthrough'; generate(c, p, root, frames=2)
            self.assertEqual(cpu_check(root)['status'], 'passed')

    def test_fusion_reduces_materialization_and_keeps_quantization_boundaries(self):
        b = GraphBuilder(); x = b.input('x', tensor_type([7], 8), 0)
        y = b.add('add', [x, x], tensor_type([7], 8), {}, {})
        z = b.add('mul', [y, y], tensor_type([7], 8), {}, {})
        c = make_program(program_from_builder(b, [z]))
        a, f = make_plan(c), make_plan(c, fuse=True)
        self.assertLess(f['kernels'][0]['scratch_bytes'], a['kernels'][0]['scratch_bytes'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'wrap'; generate(c, f, root, frames=4)
            self.assertEqual(cpu_check(root)['status'], 'passed')

    def test_planner_budget_unknown_cost_and_reference_retention(self):
        c = make_program(declared_matmul(), Target(max_compute_tiles=3))
        plans, report = enumerate_plans(c, max_plans=24, max_variants=3)
        self.assertEqual(report['reference_plan_id'], plans[0]['id'])
        self.assertTrue(any(p['rewrite_trace'] for p in plans))
        self.assertTrue(any(p['planned_compute_tiles'] > 1 for p in plans))
        self.assertTrue(all(e['result']['conditional_latency_ns'] is None for e in report['evaluations']))
        small, _ = enumerate_plans(c, max_plans=2)
        self.assertEqual(len(small), 2)

    def test_illegal_partition_memory_and_i64_remain_explicit(self):
        c = make_program(declared_matmul(), Target(max_compute_tiles=1))
        for cuts in ((-1,), (1, 1), (1,)):
            with self.assertRaises(ValueError): make_plan(c, cuts=cuts)
        b = GraphBuilder(); x = b.input('large', tensor_type([8000], 32), 0)
        with self.assertRaisesRegex(ValueError, 'memory'): make_plan(make_program(program_from_builder(b, [x])))
        b = GraphBuilder(); x = b.input('wide', tensor_type([1], 64), 0)
        with self.assertRaisesRegex(ValueError, 'i64'): make_plan(make_program(program_from_builder(b, [x])))

    def test_plan_replay_rejects_forged_kernels_and_source(self):
        c = make_program(declared_matmul()); p = make_plan(c)
        for kind in ('kernel', 'graph', 'interface'):
            bad, contract = copy.deepcopy(p), copy.deepcopy(c)
            if kind == 'kernel': bad['kernels'][0]['outputs'] = []
            if kind == 'graph': bad['program']['outputs'] = []
            if kind == 'interface': contract['interface']['layout'] = 'guessed'
            with self.assertRaises(ValueError): validate_plan(contract, bad)

    def test_scalar_matmul_orders_and_rewritten_multi_core_generation(self):
        c = make_program(declared_matmul(), Target(max_compute_tiles=3))
        step = next(default_program_rules().applications(c['program']))
        rewritten = replay(c['program'], [step])
        for order, trace, cuts in (('ijk', [], ()), ('ikj', [], ()), ('ijk', [step], (4,))):
            with tempfile.TemporaryDirectory() as tmp:
                plan = make_plan(c, trace=trace, cuts=cuts, fuse=True, matmul_order=order)
                root = Path(tmp) / 'product'; generate(c, plan, root, frames=4)
                self.assertEqual(cpu_check(root)['status'], 'passed')

    def test_artifact_replays_codegen_and_rejects_edited_files(self):
        c = make_program(declared_matmul()); p = make_plan(c, fuse=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'artifact'; generate(c, p, root, frames=1)
            check_artifact(root)
            (root / 'compute.hpp').write_text('// replaced')
            with self.assertRaisesRegex(ValueError, 'changed'): check_artifact(root)

    def test_cost_coverage_and_conditional_clock_solver(self):
        c = make_program(declared_matmul()); p = make_plan(c, fuse=True)
        request = cost_request(c, p); registry = software_registry(request)
        selection = {'cost_by_component': {}, 'member_by_group': {'fixture': 'declared'}}
        result = evaluate_costs(c, p, registry, selection)
        self.assertEqual(result['status'], 'conditional_estimate')
        self.assertGreater(result['conditional_latency_ns'], 0)
        self.assertFalse(result['native_admitted']); self.assertIsNone(result['latency_ns'])
        registry['entries'].pop()
        result = evaluate_costs(c, p, registry, selection)
        self.assertIsNone(result['conditional_latency_ns'])

    def test_fusion_and_loop_order_invalidate_kernel_cost_reuse(self):
        c = make_program(declared_matmul()); a, b = make_plan(c), make_plan(c, fuse=True, matmul_order='ikj')
        registry = software_registry(cost_request(c, a))
        result = evaluate_costs(c, b, registry, {'cost_by_component': {}, 'member_by_group': {'fixture': 'declared'}})
        self.assertNotEqual(result['status'], 'conditional_estimate')
        self.assertIsNone(result['conditional_latency_ns'])

    def test_multi_core_cost_graph_has_valid_links_and_all_output_join(self):
        c = make_program(declared_matmul()); p = make_plan(c, cuts=(1,), fuse=True)
        request = cost_request(c, p)
        result = evaluate_costs(c, p, software_registry(request),
                                {'cost_by_component': {}, 'member_by_group': {'fixture': 'declared'}})
        self.assertEqual(result['status'], 'conditional_estimate')
        self.assertTrue(any(':to:' in e['id'] for e in result['timing']['events']))

    def test_typed_native_output_padding_and_timestamp_checks(self):
        typ = tensor_type([3], 16)
        self.assertEqual(transfer_elements(typ), 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'output.txt'; path.write_text('T 5 ns\n-32768 32767 1 0 0 0 0 0\n')
            data, times = parse_typed_output(path, typ, 1)
            self.assertEqual(times, [5.]); self.assertEqual(data[:3], [-32768, 32767, 1])
            path.write_text('T 5 ns\n32768 0 0 0 0 0 0 0\n')
            with self.assertRaisesRegex(ValueError, 'type'): parse_typed_output(path, typ, 1)

    def test_new_registered_rule_uses_existing_search_and_backend(self):
        from Optimizer.program_rules import ProgramRuleRegistry
        b = GraphBuilder(); x = b.input('x', tensor_type([3], 16), 0)
        y = b.input('y', tensor_type([3], 16), 1)
        out = b.add('add', [x, y], tensor_type([3], 16), {}, {})
        c = make_program(program_from_builder(b, [out])); registry = ProgramRuleRegistry()
        def matches(p): return [{'node_id': n['id']} for n in p['nodes'] if n['op'] == 'add']
        def change(p, step):
            graph = GraphBuilder(); ids = {}
            for a in p['inputs']: ids[a['id']] = graph.input(a['name'], a['type'], a['position'])
            for n in p['nodes']:
                args = n['operands'][::-1] if n['id'] == step['node_id'] else n['operands']
                ids[n['id']] = graph.add(n['op'], [ids[v] for v in args], n['type'], n['attributes'], n['origin'])
            return program_from_builder(graph, [ids[o['value']] for o in p['outputs']], p['entry'])
        registry.register('test.add.commute', 1, matches, change, 'modular addition commutes')
        plans, _ = enumerate_plans(c, registry=registry, max_variants=2)
        p = next(p for p in plans if p['rewrite_trace'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'plugin'; generate(c, p, root, rewrite_registry=registry)
            self.assertEqual(cpu_check(root, rewrite_registry=registry)['status'], 'passed')

    def test_rank_three_views_nonuniform_constant_splat_and_mixed_outputs(self):
        b = GraphBuilder(); x = b.input('cube', tensor_type([2, 3, 2], 16), 0)
        scalar = b.input('scalar', tensor_type([], 16), 1)
        splat = b.add('splat', [scalar], tensor_type([2, 3, 2], 16), {}, {})
        cst = b.add('constant', [], tensor_type([2, 3, 2], 16), {'values': list(range(12))}, {})
        a = b.add('add', [x, splat], tensor_type([2, 3, 2], 16), {}, {})
        a = b.add('add', [a, cst], tensor_type([2, 3, 2], 16), {}, {})
        t = b.add('transpose', [a], tensor_type([2, 2, 3], 16), {'permutation': [2, 0, 1]}, {})
        s = b.add('slice', [t], tensor_type([1, 2, 2], 16), {'offsets': [1, 0, 0], 'strides': [1, 1, 2]}, {})
        narrow = b.add('truncate', [s], tensor_type([1, 2, 2], 8), {}, {})
        c = make_program(program_from_builder(b, [narrow, scalar]))
        for fuse in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'views'; generate(c, make_plan(c, fuse=fuse), root)
                self.assertEqual(cpu_check(root)['status'], 'passed')

    def test_conditional_ranking_retains_unknown_candidates(self):
        c = make_program(declared_matmul())
        baseline = make_plan(c)
        def evaluator(p):
            if p['id'] != baseline['id']: return {'status': 'needs_measurement', 'conditional_latency_ns': None}
            return evaluate_costs(c, p, software_registry(cost_request(c, p)),
                                  {'cost_by_component': {}, 'member_by_group': {'fixture': 'declared'}})
        plans, report = enumerate_plans(c, max_plans=6, evaluator=evaluator)
        self.assertEqual(len(plans), 6)
        self.assertEqual(len(report['conditional_rankings']), 1)
        self.assertEqual(len(report['unknown_candidates']), 5)
        self.assertFalse(report['conditional_rankings'][0]['native_admitted'])

    def test_no_op_execution_parameters_do_not_create_duplicate_candidates(self):
        b = GraphBuilder(); x = b.input('x', tensor_type([3], 8), 0)
        c = make_program(program_from_builder(b, [x]))
        plans, report = enumerate_plans(c)
        self.assertEqual(len(plans), 1)
        self.assertEqual(len(report['identical_lowerings']), 1)


@unittest.skipUnless(importlib.util.find_spec('xdsl'), 'optional MLIR frontend')
class ProgramImportTests(unittest.TestCase):
    def test_four_layers_compose_without_a_new_program_mode(self):
        from Optimizer.importing.mlir_frontend import import_mlir_text
        text = (FIXTURES / 'three_layer.mlir').read_text()
        text = text.replace('return %y : tensor<4x8xi8>\n  }\n}',
            '%four = func.call @stage(%y, %a) : (tensor<4x8xi8>, tensor<8x8xi8>) -> tensor<4x8xi8>\n return %four : tensor<4x8xi8>\n  }\n}')
        c = program_from_import(import_mlir_text(text))
        self.assertEqual(sum(n['op']=='matmul' for n in c['program']['nodes']), 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'four'; generate(c, make_plan(c, fuse=True), root, frames=2)
            self.assertEqual(cpu_check(root)['status'], 'passed')

    def test_three_applications_use_same_contract_planner_and_generator(self):
        from Optimizer.importing.mlir_frontend import import_mlir
        for path in (FIXTURES / 'three_layer.mlir', FIXTURES / 'fir.mlir', ROOT / 'tests/fixtures/optimizer_import/branched.mlir'):
            with self.subTest(application=path.stem):
                artifact = import_mlir(path)
                self.assertEqual(artifact['backend_admission']['status'], 'unsupported')
                c = program_from_import(artifact); validate_contract(c)
                plans, report = enumerate_plans(c, max_plans=8, max_variants=2)
                for plan in (plans[0], plans[1]):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp) / path.stem; generate(c, plan, root, frames=2)
                        result = cpu_check(root)
                        self.assertEqual(result['status'], 'passed')
                        self.assertTrue(result['original_source_checked'])

    def test_fir_known_numerical_function_and_rank_one_signature(self):
        from Optimizer.importing.mlir_frontend import import_mlir
        p = import_mlir(FIXTURES / 'fir.mlir')['program']
        self.assertEqual(evaluate(p, {'arg0': list(range(10))}), [[-1] * 8])

    @unittest.skipUnless(importlib.util.find_spec('clang'), 'optional C++ frontend')
    def test_multi_output_cpp_is_admitted_and_checked_against_original_compiler(self):
        from Optimizer.importing.cpp_frontend import import_cpp
        artifact = import_cpp(FIXTURES / 'residual.cpp', 'residual', project_root=FIXTURES)
        self.assertEqual(artifact['backend_admission']['status'], 'unsupported')
        c = program_from_import(artifact); p = make_plan(c, fuse=True)
        self.assertEqual(len(c['program']['outputs']), 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'residual'; generate(c, p, root, frames=2)
            self.assertEqual(cpu_check(root)['status'], 'passed')
            self.assertEqual(json.loads((root / 'source_check.json').read_text())['kind'], 'original_cpp_host_execution')

    @unittest.skipUnless(importlib.util.find_spec('clang'), 'optional C++ frontend')
    def test_cpp_interleaved_output_parameter_binding_is_preserved(self):
        from Optimizer.importing.cpp_frontend import import_cpp
        path = ROOT / 'tests/fixtures/optimizer_cpp/two_layer.cpp'
        artifact = import_cpp(path, 'network', project_root=path.parent)
        c = program_from_import(artifact)
        self.assertEqual(c['source']['input_binding'], {'arg0': 'arg0', 'arg1': 'arg1', 'arg2': 'arg3'})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'binding'; generate(c, make_plan(c, fuse=True), root, frames=2)
            self.assertEqual(cpu_check(root)['status'], 'passed')


if __name__ == '__main__': unittest.main()
