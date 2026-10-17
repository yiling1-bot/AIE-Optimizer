import copy
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Optimizer.contracts import make_contract,validate_contract,Target,import_reviewed_k4
from Optimizer.semantics import Spec,Alternative,Selection,evaluate
from Optimizer.planner import make_plan,enumerate_plans,expression_for,analyzer_request,MODES,gather_regions
from Optimizer.backend import generate
from Optimizer.runner import cpu_check,check_artifact,parse_output,mapping_report,tool_identity,compile_diagnosis
from Optimizer.frontier import verified_frontier
from Optimizer.fixtures import reference
from Optimizer.resources import compiler_resources
from Optimizer.search import choose,campaign


class OptimizerTests(unittest.TestCase):
    def test_placement_failure_proposes_coupled_tree_rewrite(self):
        c=make_contract();plans,_=enumerate_plans(c);failed=make_plan(c,'recompute',8)
        record={'candidate_id':failed['id'],'mode':'recompute','row_groups':8,
                'diagnosis':compile_diagnosis('EXCEPTION: vector\nGlobal placement failed')}
        proposal=choose(plans,[failed],[record])
        self.assertEqual(proposal['mode'],'recompute')
        self.assertEqual(proposal['row_groups'],8);self.assertEqual(proposal['gather'],'tree')
        self.assertTrue(record['diagnosis']['compiler_exception_observed'])

    def test_gather_tree_exact_order_coverage_and_bounded_fanin(self):
        for groups in range(1,9):
            children,regions=gather_regions(groups,'tree');outputs={}
            for g in range(groups):
                out=[]
                for child,lo,hi in children[g]:
                    self.assertLess(child,g)
                    self.assertEqual(outputs[child],list(range(lo,hi)))
                    out.extend(outputs[child])
                out.append(g);outputs[g]=out
                self.assertEqual(out,list(range(*regions[g])))
            self.assertEqual(outputs[groups-1],list(range(groups)))
        self.assertEqual(len(gather_regions(8,'tree')[0][7]),3)
        self.assertEqual(len(gather_regions(8,'flat')[0][7]),7)

    def test_gather_changes_codegen_but_preserves_logical_work_and_cores(self):
        c=make_contract();flat=make_plan(c,'recompute',8);tree=make_plan(c,'recompute',8,'tree')
        self.assertEqual(flat['planned_compute_tiles'],tree['planned_compute_tiles'])
        self.assertEqual(flat['expression_id'],tree['expression_id'])
        self.assertNotEqual(flat['id'],tree['id'])
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'tree';generate(c,tree,d)
            header=(d/'kernels.hpp').read_text()
            signature=next(s for s in header.splitlines() if s.startswith('void fused_7'))
            self.assertIn('In& y3',signature);self.assertIn('In& y5',signature);self.assertIn('In& y6',signature)
            self.assertNotIn('In& y0',signature)

    def test_search_resumes_failed_attempts_and_honors_one_core_cap(self):
        c=make_contract(target=Target(max_compute_tiles=1))
        plans,_=enumerate_plans(c)
        self.assertEqual(choose(plans,[])['planned_compute_tiles'],1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('Optimizer.search.cpu_check',return_value={'status':'failed'}) as host:
                first=campaign(c,tmp,'not_used',max_evaluations=1)
                self.assertEqual(host.call_count,1)
                self.assertEqual(first['stop_reason'],'evaluation_budget')
            with patch('Optimizer.search.cpu_check',side_effect=AssertionError('must reuse failure')):
                second=campaign(c,tmp,'not_used',max_evaluations=1)
                self.assertEqual(second['historical_evaluations_included'],1)
                self.assertFalse(second['frontier']['groups'])

    def test_no_supported_plan_is_reported_before_running_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'no supported deployment'):
                campaign(make_contract(Spec(m=3)),tmp,'not_used')

    def test_portfolio_retains_structural_and_resource_diversity(self):
        plans,_=enumerate_plans(make_contract());attempted=[]
        for _ in range(8):attempted.append(choose(plans,attempted))
        self.assertEqual(len({p['id'] for p in attempted}),8)
        self.assertEqual({p['mode'] for p in attempted[:5]},set(MODES))
        self.assertGreaterEqual(len({p['planned_compute_tiles'] for p in attempted}),3)

    def test_resource_accounting_deduplicates_shared_buffers(self):
        a=dict(tile='aie',column=0,row=2,offset=0,size=1024)
        report={'mapping':{'portInstanceMapping':{'producer':{'bufferInfo':[a]},'consumer':{'bufferInfo':[a]}},
                           'systemMemory':[dict(a,offset=512,size=1024)]}}
        r=compiler_resources(report)
        self.assertEqual(r['memory_modules'],1)
        self.assertEqual(r['reported_address_union_bytes'],1536)
        self.assertEqual(r['unique_reported_allocations'],2)

    def test_compiler_identity_excludes_session_timestamp(self):
        one='v++ v2024.2\nSW Build 1234 on 2024-11-10\nStart of session at 1'
        two=one[:-1]+'2'
        self.assertEqual(tool_identity(one),tool_identity(two))
        self.assertNotEqual(tool_identity(one),tool_identity(one.replace('1234','1235')))

    def test_resource_count_is_derived_from_distinct_slots(self):
        c=make_contract(target=Target(max_compute_tiles=16))
        plans,_=enumerate_plans(c)
        self.assertEqual({p['planned_compute_tiles'] for p in plans},{1,2,4,8,16})
        for p in plans:
            self.assertEqual(p['planned_compute_tiles'],len({tuple(s['tile']) for s in p['slots']}))
            self.assertIsNone(p['compiled_compute_tiles'])
        self.assertEqual(make_plan(c,'materialized',4)['planned_compute_tiles'],8)
        self.assertEqual(make_plan(c,'row_streaming',4)['planned_compute_tiles'],4)

    def test_resource_caps_are_upper_bounds_and_do_not_pad_idle_cores(self):
        c=make_contract(target=Target(max_compute_tiles=3))
        plans,refused=enumerate_plans(c)
        self.assertEqual({p['planned_compute_tiles'] for p in plans},{1,2})
        self.assertTrue(refused)
        with self.assertRaises(ValueError):make_plan(c,'materialized',2)

    def test_invalid_contract_and_incomplete_microtiles_rejected(self):
        c=make_contract();bad=copy.deepcopy(c);bad['numerics']['shift']=1
        with self.assertRaises(ValueError):validate_contract(bad)
        with self.assertRaises(ValueError):make_plan(make_contract(Spec(m=7)),'materialized',1)
        with self.assertRaises(ValueError):make_contract(Spec(k=2**18))

    def test_composed_plan_expressions_match_independent_reference(self):
        rng=random.Random(13)
        for relu in (False,True):
            spec=Spec(8,16,24,16,relu)
            inputs={k:[[rng.randrange(-128,128) for _ in range(n)] for _ in range(m)]
                    for k,m,n in [('X',8,16),('W0',16,24),('W1',24,16)]}
            expected=reference(inputs['X'],inputs['W0'],inputs['W1'],relu)
            for mode in MODES:
                for groups in (1,2):
                    expr=expression_for(spec,mode,groups)
                    alt=Alternative(mode,expr,(),('unbound',),'')
                    choice=Selection(mode,'dot_product','outer_product','unbound')
                    self.assertEqual(evaluate({mode:alt},choice,spec,inputs),expected)

    @unittest.skipUnless(shutil.which('g++'),'requires host C++ compiler')
    def test_generated_cpp_every_structure_and_relu(self):
        with tempfile.TemporaryDirectory() as tmp:
            for relu in (False,True):
                c=make_contract(Spec(8,16,24,16,relu))
                for mode in MODES:
                    for groups in (1,2):
                        with self.subTest(relu=relu,mode=mode,groups=groups):
                            d=Path(tmp)/f'{mode}_{groups}_{relu}'
                            generate(c,make_plan(c,mode,groups),d,frames=3)
                            result=cpu_check(d)
                            self.assertEqual(result['status'],'passed',(d/'cpu_compile.log').read_text())

    def test_generated_graph_and_manifest_cover_every_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=make_contract();p=make_plan(c,'materialized',4);d=Path(tmp)/'fresh'
            m=generate(c,p,d,frames=2)
            self.assertEqual(len(m['kernel_bindings']),8)
            self.assertTrue({'compute.hpp','graph.hpp','kernels.cpp','data/X.txt','golden.json'} <= set(m['files']))
            check_artifact(d)
            with self.assertRaises(FileExistsError):generate(c,p,d)
            (d/'compute.hpp').write_text('changed')
            with self.assertRaises(ValueError):check_artifact(d)

    def test_analyzer_refuses_uncovered_target_without_fake_latency(self):
        c=make_contract();r=analyzer_request(c,make_plan(c,'row_streaming',2))
        self.assertEqual(r['status'],'unsupported');self.assertIsNone(r['latency_ns'])
        self.assertFalse(r['simulation_timing_used_as_prediction'])

    def test_frontier_keeps_tradeoffs_and_separates_conditions(self):
        def point(name,n,l,key='same'):
            return dict(candidate_id=name,actual_compute_tiles=n,latency_ns=l,comparison_key=key,
                        numerical='EXACT',mapping='EXACT',status='verified',
                        evidence_kind='native_simulation',source_digest='digest')
        rows=[point('few',2,100),point('fast',4,70),point('dominated',4,110),
              point('other',1,1,'other'),point('unknown',1,0)]
        result=verified_frontier(rows)
        self.assertEqual([r['candidate_id'] for r in result['groups']['same']],['few','fast'])
        self.assertEqual(len(result['groups']['other']),1);self.assertEqual(len(result['refused']),1)
        rows[0]['evidence_kind']='prediction'
        self.assertEqual(len(verified_frontier(rows)['refused']),2)

    def test_timestamp_parser_requires_complete_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out.txt';p.write_text('T 1 ns\n1 2\nT 2 ns\n3 4\n')
            self.assertEqual(parse_output(p,2,2),([1,2,3,4],[1.,2.]))
            with self.assertRaises(ValueError):parse_output(p,2,3)
            p.write_text('1 2\n')
            with self.assertRaises(ValueError):parse_output(p,2,1)

    def test_semantic_import_refuses_unreviewed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'chain.cpp').write_text('void chain_first(){}')
            with self.assertRaises(ValueError):import_reviewed_k4(tmp)

    def test_mapping_checks_names_positions_and_actual_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.json'
            d={'blockInstances':{'0':{'blockType':'t'}},'blockTypes':{'t':{'type':'function','function':'f'}},
               'mapping':{'blockInstanceMapping':{'0':{'coreInfo':{'column':0,'row':2}}}}}
            p.write_text(json.dumps(d))
            self.assertEqual(mapping_report(p,[{'function':'f','tile':[0,2]}]),{'f':[0,2]})
            with self.assertRaises(ValueError):mapping_report(p,[{'function':'f','tile':[0,3]}])


if __name__=='__main__':unittest.main()
