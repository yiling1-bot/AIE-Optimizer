import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from Optimizer.contracts import make_contract,Target
from Optimizer.semantics import Spec
from Optimizer.joint_extract import enumerate_joint_plans,rank_plans,choose_joint
from Optimizer.phase_model import lower_phases
from Optimizer.phase_costs import PhaseEvaluator
from Optimizer.backend import generate
from Optimizer.runner import cpu_check
from Optimizer.search import campaign
from test_optimizer_phases import bind,registry_for


OPTIONS=dict(max_rounds=1,max_terms=100,max_applications=2,max_states=150,max_plans=28,
             max_executions=1,execution_depth=0)


class JointTests(unittest.TestCase):
    def setUp(self):
        self.c=make_contract(Spec(8,16,16,16,False),Target(max_compute_tiles=2))

    def test_all_visited_semantic_dags_reach_deployment_without_structural_top_k(self):
        plans,r=enumerate_joint_plans(self.c,**OPTIONS)
        self.assertGreaterEqual(r['semantic_expressions_seen'],3)
        self.assertEqual(len({p['expression_id'] for p in plans}),r['semantic_expressions_seen'])
        self.assertEqual({p['planned_compute_tiles'] for p in plans},{1,2})
        self.assertEqual(r['structural_proxy_prunes'],0)
        self.assertEqual(len(r['ranking']['unknown_plan_ids']),len(plans))
        self.assertEqual(plans[0]['id'],r['reference_plan_id'])
        self.assertTrue(any('schedule.defer_gather_acquire' in [s['rule'] for s in p['execution_steps']] for p in plans))
        self.assertFalse(r['native_optimality'])

    def test_real_phase_evaluator_selects_joint_core_and_acquire_order(self):
        plans,_=enumerate_joint_plans(self.c,**OPTIONS)
        bindings={p['id']:bind(self.c,p,frames=2) for p in plans}
        graphs=[lower_phases(self.c,p,bindings[p['id']]) for p in plans]
        def duration(s):
            if s['mechanism']=='complete_region':return 100000*s['operations']['region']['output']['rows']
            return 10000 if s['mechanism']=='whole_frame_transfer' else 0
        with tempfile.TemporaryDirectory() as d:
            registry,selection=registry_for(graphs,d,duration)
            evaluator=PhaseEvaluator(self.c,registry,selection,bindings)
            selected,report=enumerate_joint_plans(self.c,evaluator=evaluator,**OPTIONS)
            best=report['ranking']['ranked'][0];winner=next(p for p in selected if p['id']==best['plan_id'])
            self.assertEqual(winner['planned_compute_tiles'],2)
            self.assertIn('schedule.defer_gather_acquire',[s['rule'] for s in winner['execution_steps']])
            self.assertIsNone(report['latency_ns'])
            for row in report['evaluations'].values():
                self.assertIsNone(row['latency_ns']);self.assertFalse(row['native_admitted'])

    def test_unknown_candidate_gets_reserved_evaluation_and_mixed_context_refused(self):
        plans,report=enumerate_joint_plans(self.c,**OPTIONS)
        estimates={p['id']:{'status':'conditional_estimate','plan_id':p['id'],
             'conditional_latency_ns':i+1,'conditional_frame_completion_ns':[i+1,2*i+2],
             'latency_ns':None,'native_admitted':False,'comparison_key':'work','scenario_id':'same-member'} for i,p in enumerate(plans)}
        missing=plans[-1]['id'];estimates[missing]={'status':'needs_measurement'}
        ranking=rank_plans(plans,estimates);report['ranking']=ranking
        self.assertIn(missing,ranking['unknown_plan_ids'])
        self.assertEqual(choose_joint(plans,plans[:3],report)['id'],missing)
        estimates[plans[1]['id']]['scenario_id']='different-member'
        with self.assertRaisesRegex(ValueError,'mixed'):rank_plans(plans,estimates)

    def test_budget_report_and_invalid_scores(self):
        plans,r=enumerate_joint_plans(self.c,**{**OPTIONS,'max_plans':3})
        self.assertTrue(r['plan_budget_reached']);self.assertEqual(len(plans),3)
        invalid={'status':'conditional_estimate','conditional_latency_ns':float('nan'),
                 'native_admitted':False,'latency_ns':None}
        with self.assertRaises(ValueError):enumerate_joint_plans(self.c,evaluator=lambda p:invalid,**{**OPTIONS,'max_plans':1})

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_joint_certificate_and_execution_remain_codegen_replayable(self):
        plans,_=enumerate_joint_plans(self.c,**OPTIONS)
        plan=next(p for p in plans if any(s['rule']=='schedule.defer_gather_acquire' for s in p['execution_steps']))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'candidate';generate(self.c,plan,path,frames=4)
            self.assertEqual(cpu_check(path)['status'],'passed')

    def test_native_campaign_requires_conditional_opt_in_and_pins_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'opt-in'):
                campaign(self.c,Path(d)/'blocked','unused',joint_evaluation={})
            # Budget zero is still invalid; the user cannot hide a native claim
            # by creating a campaign with no actual evaluation allowance.
            with self.assertRaises(ValueError):campaign(self.c,Path(d)/'zero','unused',max_evaluations=0)


if __name__=='__main__':unittest.main()
