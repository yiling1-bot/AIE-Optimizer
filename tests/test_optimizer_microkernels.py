from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from Optimizer.contracts import Target
from Optimizer.importing.ir import GraphBuilder, tensor_type
from Optimizer.program_ir import make_program, program_from_builder
from Optimizer.program_kernels import default_implementations
from Optimizer.program_planner import make_plan, enumerate_plans, validate_plan
from Optimizer.program_backend import generate, cpu_check, emit_compute
from Optimizer.program_costs import cost_request
from Optimizer.program_ir import seal
from tools.microkernel_campaign import validate_batch_predictions


def gemm(m=8, k=16, n=16, bits=8, seed=7):
    b = GraphBuilder(); a = b.input('a', tensor_type([m, k], bits), 0)
    w = b.input('w', tensor_type([k, n], bits), 1)
    init = b.add('constant', [], tensor_type([m, n], 32), {'values': [seed]}, {})
    out = b.add('matmul', [a, w, init], tensor_type([m, n], 32), {}, {})
    return make_program(program_from_builder(b, [out]), Target(max_compute_tiles=3))


class MicrokernelTests(unittest.TestCase):
    def test_both_mmul_grids_match_extreme_signed_inputs_and_nonzero_initializers(self):
        for grid in (1, 2):
            c = gemm(); p = make_plan(c, fuse=True, lowerings={'v1': f'aie_ml.mmul.i8.grid1x{grid}'})
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'kernel'; generate(c, p, root, frames=4)
                self.assertEqual(cpu_check(root)['status'], 'passed')
            self.assertGreater(p['kernels'][0]['implementation_scratch_bytes'], 0)
            self.assertIn('aie::mmul<4,8,8', emit_compute(p)[0])

    def test_illegal_shapes_types_and_overflow_do_not_silently_select_mmul(self):
        for c in (gemm(m=3), gemm(k=7), gemm(n=8), gemm(bits=16), gemm(seed=2**31-1)):
            with self.assertRaises(ValueError): make_plan(c, fuse=True, lowerings={'v1': 'aie_ml.mmul.i8.grid1x2'})

    def test_mixed_per_operator_implementations_compose_in_one_program(self):
        b = GraphBuilder(); a = b.input('x', tensor_type([4, 8], 8), 0); w = b.input('w', tensor_type([8, 8], 8), 1)
        z = b.add('constant', [], tensor_type([4, 8], 32), {'values': [0]}, {})
        p = b.add('matmul', [a, w, z], tensor_type([4, 8], 32), {}, {})
        q = b.add('matmul', [a, w, z], tensor_type([4, 8], 32), {}, {})
        out = b.add('add', [p, q], tensor_type([4, 8], 32), {}, {})
        c = make_program(program_from_builder(b, [out])); plan = make_plan(c, fuse=True, lowerings={p: 'aie_ml.mmul.i8.grid1x1'})
        self.assertEqual(plan['lowerings'][q], 'scalar.matmul.ijk')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'mixed'; generate(c, plan, root)
            self.assertEqual(cpu_check(root)['status'], 'passed')

    def test_registry_addition_needs_no_search_or_codegen_branch(self):
        registry = default_implementations()
        registry.register(replace(registry.entries['scalar.matmul.ikj'], id='extension.matrix', explore=True))
        c = gemm(); plans, report = enumerate_plans(c, max_plans=20, max_variants=1, implementation_registry=registry)
        plan = next(p for p in plans if 'extension.matrix' in p['lowerings'].values())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'extension'; generate(c, plan, root, implementation_registry=registry)
            self.assertEqual(cpu_check(root, implementation_registry=registry)['status'], 'passed')

    def test_implementation_choice_is_bound_to_plan_and_cost_signature(self):
        c = gemm(); scalar = make_plan(c, fuse=True)
        native = make_plan(c, fuse=True, lowerings={'v1': 'aie_ml.mmul.i8.grid1x1'})
        a, b = cost_request(c, scalar), cost_request(c, native)
        ca = next(s for s in a['components'] if s['mechanism'] == 'program_complete_region')
        cb = next(s for s in b['components'] if s['mechanism'] == 'program_complete_region')
        self.assertNotEqual(ca['compile_identity'], cb['compile_identity'])
        native['lowerings']['v1'] = 'scalar.matmul.ijk'
        with self.assertRaises(ValueError): validate_plan(c, native)

    def test_guarded_candidates_include_scalar_and_native_and_keep_unknown_cost(self):
        plans, report = enumerate_plans(gemm(), max_plans=10, max_variants=1)
        self.assertTrue(any('aie_ml.mmul.i8.grid1x1' in p['lowerings'].values() for p in plans))
        self.assertTrue(any('aie_ml.mmul.i8.grid1x2' in p['lowerings'].values() for p in plans))
        self.assertEqual(len(report['unknown_candidates']), len(plans))

    def test_heldout_batch_binds_prediction_identity_plan_model_and_frames(self):
        cases = [{'name':'ho','role':'HO','plan':{'id':'plan'},'frames':1}]
        prediction = {'plan_id':'plan','model_id':'model','execution_frames':1,
                      'status':'conditional_estimate','conditional_latency_ns':10,'native_admitted':False}
        data = {'schema':'aie-optimizer-prediction-lock-v1','model_id':'model',
                'predictions':{'ho':prediction},'outcomes_read':False}
        validate_batch_predictions(cases,seal(data))
        for key,value in [('plan_id','other'),('model_id','other'),('execution_frames',4),
                          ('conditional_latency_ns',float('inf')),('native_admitted',True)]:
            bad = deepcopy(data);bad['predictions']['ho'][key]=value
            with self.assertRaises(ValueError): validate_batch_predictions(cases,seal(bad))
        bad = seal(data);bad['outcomes_read']=True
        with self.assertRaises(ValueError): validate_batch_predictions(cases,bad)


if __name__ == '__main__': unittest.main()
