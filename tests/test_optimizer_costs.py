import copy
from dataclasses import replace
from pathlib import Path
import json,tempfile,unittest
from Optimizer.backend import generate
from Optimizer.contracts import make_contract,Target
from Optimizer.egraph_rules import default_equality_registry
from Optimizer.ir_planner import make_ir_plan
from Optimizer.stage_costs import stage_request,resolve_compute_costs
from Optimizer.cost_refinement import refine_for_cost
from Optimizer.semantics import Spec,build


class StageCostTests(unittest.TestCase):
    def request(self):
        spec=Spec();c=make_contract(spec);r=default_equality_registry()
        _,trace=refine_for_cost(build(spec)['materialized_layers'].expression,spec,Target(),r)
        p=make_ir_plan(c,execution_steps=trace)
        return c,p,stage_request(c,p)

    def catalog(self,request):
        return {'schema':'aie-compute-cost-catalog-v1','entries':[
            {'stage_id':s['id'],'key':s['key'],'cycles':100,'scope':'isolated_compute',
             'evidence_kind':'compiled_cycle_model','evidence_ref':'synthetic-unit-test','evidence_sha256':'a'*64}
            for s in request['stages']]}

    def test_features_and_unknown_coverage_do_not_invent_latency(self):
        c,p,q=self.request();f=[s['features'] for s in q['stages']]
        self.assertEqual(sum(x['macs'] for x in f),65536)
        self.assertEqual(sum(x['qmm_calls'] for x in f),2)
        self.assertEqual(sum(x['explicit_copy_bytes'] for x in f),0)
        report=resolve_compute_costs(q,{'schema':'aie-compute-cost-catalog-v1','entries':[]})
        self.assertEqual(report['status'],'uncovered');self.assertIsNone(report['latency_ns'])
        self.assertEqual(report['known_serial_compute_cycles_by_kernel'],{})
        known=resolve_compute_costs(q,self.catalog(q))
        self.assertTrue(known['compute_cycles_complete']);self.assertIsNone(known['latency_ns'])
        self.assertEqual(known['known_serial_compute_cycles_by_kernel'],{'fused_0':200})

    def test_no_reuse_across_context_and_no_end_to_end_timing_as_compute(self):
        c,p,q=self.request();catalog=self.catalog(q)
        p2=make_ir_plan(c,groups=2);q2=stage_request(c,p2)
        self.assertFalse(resolve_compute_costs(q2,catalog)['covered_stages'])
        for field,value in [('scope','end_to_end'),('cycles',float('nan')),('cycles',-1),('evidence_sha256','unknown')]:
            bad=copy.deepcopy(catalog);bad['entries'][0][field]=value
            with self.assertRaises(ValueError):resolve_compute_costs(q,bad)

    def test_reject_changed_plan_or_catalog_identity(self):
        c,p,q=self.request();bad=copy.deepcopy(p);bad['row_groups']=2
        with self.assertRaises(ValueError):stage_request(c,bad)
        catalog=self.catalog(q);catalog['entries'].append(catalog['entries'][0])
        with self.assertRaises(ValueError):resolve_compute_costs(q,catalog)
        bad=copy.deepcopy(q);bad['stages'][0]['features']['macs']=0
        with self.assertRaises(ValueError):resolve_compute_costs(bad,self.catalog(q))

    def test_generated_analyzer_request_contains_stages_without_time_claim(self):
        c,p,q=self.request()
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)/'candidate';generate(c,p,d)
            value=json.loads((d/'analyzer_request.json').read_text(encoding='utf-8'))
            self.assertEqual(value['status'],'unsupported');self.assertIsNone(value['latency_ns'])
            self.assertEqual(value['stage_cost_request'],json.loads(json.dumps(q)))


if __name__=='__main__':unittest.main()
