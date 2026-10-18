from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

from Optimizer.contracts import identity
from Optimizer.program_planner import make_plan, implementation_identity
from Optimizer.program_ir import seal
from Optimizer.program_cost_model import (fit_line, line_value, geometry, fit_model,
    predict, freeze_predictions, validate_model, validate_ranking)
from test_optimizer_microkernels import gemm


class KernelCostModelTests(unittest.TestCase):
    def model(self, contract):
        return {'id': 'synthetic-unit-test-only', 'design_id': 'unit',
            'domain': {'m': 8, 'n': 16, 'k_min': 8, 'k_max': 32, 'k_step': 8,
                       'target': contract['target'], 'core_period_ps': 800},
            'models': [{'implementation': 'scalar.matmul.ijk', 'version': 1,
                        'compute': fit_line([(8, 80000), (32, 320000)]),
                        'residual': fit_line([(8, 50000), (32, 60000)])}],
            'phase_coverage': {'compute': 'synthetic test', 'transport': 'nominal', 'synchronization': 'aggregate'}}

    def test_exact_fraction_interpolation_and_ceil_no_negative_costs(self):
        line = fit_line([(8, 1), (32, 2)])
        self.assertEqual(line_value(line, 16), 2)
        self.assertEqual(line_value(line, 8), 1)
        with self.assertRaises(ValueError): line_value({'slope': [-1,1], 'intercept': [0,1]}, 8)
        with self.assertRaises(ValueError): fit_line([(8,1), (8,2)])

    def test_geometry_matches_semantics_and_does_not_dispatch_on_application_name(self):
        c = gemm(seed=0); p = make_plan(c, fuse=True)
        g = geometry(c,p)
        self.assertEqual((g['m'],g['k'],g['n']),(8,16,16))
        with self.assertRaises(ValueError): geometry(gemm(seed=1), make_plan(gemm(seed=1), fuse=True))

    def test_fit_rejects_heldout_mapping_before_reading_any_artifact(self):
        design = seal({'implementation': implementation_identity(), 'cases': [
            {'name':'train','role':'TRAIN'}, {'name':'heldout','role':'HO'}]})
        for mapping in ({'heldout':'nonexistent'}, {'train':'nonexistent','heldout':'nonexistent'}):
            with self.assertRaisesRegex(ValueError, 'TRAIN set'): fit_model(design,mapping)

    def test_unpinned_or_resealed_synthetic_models_cannot_claim_native_calibration(self):
        c = gemm(seed=0); model = self.model(c)
        with self.assertRaises(ValueError): validate_model(model)

    def test_prediction_consumes_model_and_program_only_and_keeps_accuracy_explicit(self):
        c = gemm(seed=0); p = make_plan(c, fuse=True); model = self.model(c)
        with patch('Optimizer.program_cost_model.validate_model', return_value=model): result = predict(model,c,p)
        self.assertEqual(result['status'],'conditional_estimate')
        self.assertGreater(result['conditional_latency_ns'],0)
        self.assertIsNone(result['latency_ns']); self.assertFalse(result['native_admitted'])
        self.assertEqual(set(result['components_ps']),{'input','output','compute','residual'})

    def test_unknown_implementation_and_geometry_never_become_zero_latency(self):
        c = gemm(seed=0); model = self.model(c)
        for candidate, plan in ((c, make_plan(c, fuse=True, lowerings={'v1':'aie_ml.mmul.i8.grid1x1'})),
                                (gemm(k=40,seed=0), make_plan(gemm(k=40,seed=0),fuse=True))):
            with patch('Optimizer.program_cost_model.validate_model', return_value=model): result = predict(model,candidate,plan)
            self.assertEqual(result['status'],'needs_measurement'); self.assertIsNone(result['conditional_latency_ns'])

    def test_single_frame_model_refuses_a_multi_frame_execution(self):
        c = gemm(seed=0); p = make_plan(c,fuse=True); model = self.model(c)
        with patch('Optimizer.program_cost_model.validate_model', return_value=model): result = predict(model,c,p,frames=4)
        self.assertEqual(result['status'],'needs_measurement'); self.assertIsNone(result['conditional_latency_ns'])

    def test_prediction_lock_has_no_outcome_input_and_refuses_uncovered_heldout(self):
        c = gemm(seed=0); p = make_plan(c,fuse=True); model = self.model(c)
        design = {'id':'unit','cases':[{'name':'ho','role':'HO','contract':c,'plan':p}]}
        with patch('Optimizer.program_cost_model.validate_model', return_value=model): lock = freeze_predictions(design,model)
        self.assertFalse(lock['outcomes_read']); self.assertEqual(set(lock['predictions']),{'ho'})

    def test_ranking_reports_selection_regret_without_refitting(self):
        lock = seal({'predictions': {'a':{'plan_id':'pa','conditional_latency_ns':10},
                                      'b':{'plan_id':'pb','conditional_latency_ns':20},
                                      'c':{'plan_id':'pc','conditional_latency_ns':30}}})
        actual = {n:{'status':'verified','candidate_id':'p'+n,'latency_ns':v} for n,v in zip('abc',[15,10,30])}
        r = validate_ranking(lock,actual)
        self.assertEqual(r['selected'],'a'); self.assertEqual(r['selection_regret'],.5)
        self.assertFalse(r['all_pairs_correct']); self.assertEqual(r['strict_concordant_pairs'],2)
        actual['a']['candidate_id']='wrong'
        with self.assertRaises(ValueError): validate_ranking(lock,actual)

    def test_ranking_uses_compiler_build_not_volatile_session_time(self):
        context = {'tool_identity': {'version':'2024.2','build':'5239630 on fixed-build-date'},
                   'platform_sha256':'platform', 'aie_hz':1250000000, 'actual_compute_tiles':1}
        lock = seal({'predictions': {'ho':{'plan_id':'plan','conditional_latency_ns':10,
            'execution_frames':1, 'expected_compiled_context':context}}})
        actual = {'ho':{'status':'verified','candidate_id':'plan','latency_ns':12, 'frames':1,
            'tool_version':'v++ v2024.2\nSW Build 5239630 on fixed-build-date\nStart of session at: later',
            **{k:v for k,v in context.items() if k!='tool_identity'}}}
        self.assertTrue(validate_ranking(lock,actual)['all_pairs_correct'])
        actual['ho']['tool_version']=actual['ho']['tool_version'].replace('5239630','different-build')
        with self.assertRaisesRegex(ValueError,'context differs'): validate_ranking(lock,actual)


if __name__ == '__main__': unittest.main()
