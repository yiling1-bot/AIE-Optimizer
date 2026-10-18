"""Protocol tests use a small fake backend, never native performance evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from Optimizer.compiled_analyzer import (capture_analyzer, validate_snapshot, file_ref,
    read_ref, strict_json, make_candidate, validate_candidate, evaluate_candidates,
    normalize, AnalyzerEvaluator, seal, prepare_compiled_candidate)
from Optimizer.compiled_search import rank_candidates
from Optimizer.packet_schedules import (dense_packets, packet_signature, schedule_candidates,
    PacketRuleRegistry, PacketRule)


API = '''# Synthetic API responses for adapter protocol tests only.
import json,time
from pathlib import Path
def load(ref): return json.loads(Path(ref['path']).read_text())
def analyze_request(request):
    compiler=load(request['compiler_inputs']);registry=load(request['cost_registry'])
    time.sleep(registry.get('sleep',0))
    status=registry.get('status','conditional_prediction')
    if request['mode']=='costs':status='reused'
    refs=[request[k] for k in ('compiler_inputs','program_binding','cost_registry')]
    refs+=list(request.get('source_csv',{}).values())
    value=10000+request.get('model_selection',{}).get('service',{}).get('model_index',0)*1000
    raw={'schema':'aie-generic-analyzer-result-v1','status':status,
      'Simulator_timestamps_used_as_inputs':False,'native_accuracy_admission':False,
      'graph':{'architecture':compiler.get('architecture','AIE1'),'compile_identity':'synthetic-compile',
        'counts':{'compute_tiles':1},'ports':[{'id':'output','direction':'output'}],
        'routes':[{'PLIO':'O','window_port_id':'output'}]},
      'component_costs':{'components':[{'component_id':'service','status':'reused'}]},
      'hardware_component_costs':[{'component_id':'service','TRAIN_cost_ids':['synthetic-test-only'],
                                  'executable_hardware_binding_admitted':True}],
      'software_contract_prediction_only':registry.get('software_only',False),
      'timing_prediction':{'metric':'global_origin0_to_last_valid_SDK_FILE','value_ps':value},
      'prediction':{'frames':[{'port_id':'output','call_ordinal':0,'FILE_last_ps':value//2},
                              {'port_id':'output','call_ordinal':1,'FILE_last_ps':value}]},
      'input_refs':refs,'unit_fixture_only':True}
    if status!='conditional_prediction':raw['timing_prediction']=None
    return raw
class PreparedAnalyzer:
    def __init__(self,request):self.request=request
    def analyze(self,request):return analyze_request(request)
'''

PARSER = '''# Minimal parser for test CSVs only, not the production transport.
import csv,io
def parse_commands(text,*,stall_extra_ticks):
    commands=[];packets=[];pid=None;ordinal=0
    for row in list(csv.reader(io.StringIO(text)))[1:]:
        if len(row)==1 and row[0].startswith('STALL:'):
            commands.append({'kind':'stall'});continue
        header=pid is None
        if header:pid=int(row[1])&31
        last=row[2]=='1'
        commands.append({'kind':'word','word32':int(row[1]),'last':last,'header':header,'packet_id':pid})
        if last:packets.append({'packet_id':pid});pid=None
    return {'commands':commands,'packets':packets}
'''


def csv_text(ids=(1,2,1,2), payload=17):
    return 'CMD,D,TLAST,TKEEP\n'+''.join(f'DATA,{pid},0,-1\nDATA,{payload+i},1,-1\n' for i,pid in enumerate(ids))


class AnalyzerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);source=self.root/'backend';timing=source/'Model/Timing';timing.mkdir(parents=True)
        (timing/'__init__.py').write_text('')
        (timing/'generic_analyzer.py').write_text(API)
        (timing/'cascade_backpressure_transport.py').write_text(PARSER)
        self.snapshot=capture_analyzer(source,self.root/'snapshot')
        self.request={'schema':'aie-generic-analyzer-request-v1','mode':'predict',
          'compiler_inputs':self.save('compiler.json',{}),'program_binding':self.save('binding.json',{}),
          'cost_registry':self.save('registry.json',{}),'source_csv':{'A':self.save('A.csv',csv_text())},
          'model_selection':{'service':{'cost_id':'synthetic','model_index':0}},'budgets':{'unit_test':1}}

    def save(self,name,value):
        path=self.root/name;path.write_text(value if isinstance(value,str) else json.dumps(value));return file_ref(path)

    def candidate(self,request=None,plan='plan',label=''):
        ref=self.save('request-'+plan+'.json',request or self.request)
        return make_candidate(ref,plan_id=plan,label=label)

    def test_snapshot_copy_is_independent_and_mutation_or_extra_modules_refused(self):
        (self.root/'backend/Model/Timing/generic_analyzer.py').write_text('changed original')
        validate_snapshot(self.snapshot)
        (self.root/'snapshot/Model/Timing/extra.py').write_text('')
        with self.assertRaisesRegex(ValueError,'inventory changed'):validate_snapshot(self.snapshot)
        (self.root/'snapshot/Model/Timing/generic_analyzer.py').write_text('changed snapshot')
        with self.assertRaisesRegex(ValueError,'source changed'):validate_snapshot(self.snapshot)

    def test_strict_json_and_pins_reject_ambiguous_or_changed_evidence(self):
        for raw in ('{"a":1,"a":2}','{"x":NaN}','{"x":Infinity}'):
            with self.assertRaises(ValueError):strict_json(raw)
        ref=self.save('pinned.txt','first');Path(ref['path']).write_text('later')
        with self.assertRaisesRegex(ValueError,'changed'):read_ref(ref)

    def test_candidate_binds_plan_request_and_exact_observation_free_shape(self):
        candidate=self.candidate();validate_candidate(candidate)
        bad=deepcopy(candidate);bad['plan_id']='other'
        with self.assertRaises(ValueError):validate_candidate(bad)
        request=deepcopy(self.request);request['observed_latency_ns']=123
        with self.assertRaisesRegex(ValueError,'observations forbidden'):self.candidate(request,'bad')

    def test_isolated_worker_ignores_parent_Model_import_and_maps_both_metrics(self):
        candidate=self.candidate()
        with patch.dict(sys.modules,{'Model.Timing.generic_analyzer':object()}):
            batch=evaluate_candidates([candidate],self.snapshot,self.root/'run')
        result=batch['evaluations'][0]
        self.assertEqual(result['status'],'conditional_estimate')
        self.assertEqual(result['conditional_latency_ns'],5)
        self.assertEqual(result['conditional_frame_completion_ns'],[5,10])
        self.assertIsNone(result['latency_ns']);self.assertFalse(result['native_admitted'])
        self.assertEqual(result['coverage']['TRAIN_available'],1)

    def test_exact_query_memo_and_distinct_prepared_query_are_reported_separately(self):
        first=self.candidate();same=make_candidate(first['request_ref'],plan_id='same')
        request=deepcopy(self.request);request['model_selection']['service']['model_index']=1
        changed=self.candidate(request,'changed')
        batch=evaluate_candidates([first,same,changed],self.snapshot,self.root/'batch')
        a,b,c=batch['evaluations'];self.assertFalse(a['query_reused']);self.assertTrue(b['query_reused'])
        self.assertEqual(a['raw_result_ref'],b['raw_result_ref']);self.assertTrue(c['prepared_reused'])
        self.assertNotEqual(a['scenario_id'],c['scenario_id'])
        ranking=rank_candidates([first,same,changed],batch['evaluations'])
        self.assertEqual(len(ranking['groups']),2)

    def test_nested_evidence_mutation_is_not_accepted_as_a_cached_label(self):
        candidate=self.candidate();Path(self.request['source_csv']['A']['path']).write_text('changed CSV')
        batch=evaluate_candidates([candidate],self.snapshot,self.root/'tampered')
        self.assertEqual(batch['evaluations'][0]['status'],'invalid_input')
        self.assertIsNone(batch['evaluations'][0]['conditional_latency_ns'])

    def test_uncovered_or_unsupported_candidates_are_preserved_without_zero_cost(self):
        cases=[]
        for i,status in enumerate(('needs_measurement','unsupported')):
            request=deepcopy(self.request);request['cost_registry']=self.save(f'gap-{i}.json',{'status':status})
            cases.append(self.candidate(request,'gap'+str(i)))
        batch=evaluate_candidates(cases,self.snapshot,self.root/'gaps')
        ranking=rank_candidates(cases,batch['evaluations'])
        self.assertEqual(set(ranking['unknown_candidates']),{c['id'] for c in cases})
        self.assertFalse(ranking['groups'])
        self.assertTrue(all(r['conditional_latency_ns'] is None for r in batch['evaluations']))

    def test_software_fixture_costs_cannot_be_ranked_as_native_context_predictions(self):
        request=deepcopy(self.request);request['cost_registry']=self.save('software.json',{'software_only':True})
        candidate=self.candidate(request)
        result=evaluate_candidates([candidate],self.snapshot,self.root/'software')['evaluations'][0]
        self.assertEqual(result['status'],'software_conditional_prediction')
        self.assertIsNone(result['conditional_latency_ns'])

    def test_other_architecture_is_kept_unsupported(self):
        request=deepcopy(self.request);request['compiler_inputs']=self.save('ml.json',{'architecture':'AIE-ML'})
        candidate=self.candidate(request)
        result=evaluate_candidates([candidate],self.snapshot,self.root/'ml')['evaluations'][0]
        self.assertEqual(result['status'],'unsupported');self.assertIsNone(result['conditional_latency_ns'])

    def test_worker_deadline_produces_refusal_and_no_implicit_retry(self):
        request=deepcopy(self.request);request['cost_registry']=self.save('slow.json',{'sleep':2})
        candidate=self.candidate(request)
        batch=evaluate_candidates([candidate],self.snapshot,self.root/'slow',timeout=.05)
        self.assertTrue(batch['timed_out']);self.assertEqual(batch['evaluations'][0]['status'],'refused')
        self.assertIsNone(batch['evaluations'][0]['conditional_latency_ns'])

    def test_all_output_frames_and_endpoint_must_agree(self):
        candidate=self.candidate();batch=evaluate_candidates([candidate],self.snapshot,self.root/'frames')
        good=batch['evaluations'][0];raw=strict_json(read_ref(good['raw_result_ref']))
        for mutation in ('duplicate','missing','endpoint','observation'):
            bad=deepcopy(raw)
            if mutation=='duplicate':bad['prediction']['frames'].append(bad['prediction']['frames'][0])
            if mutation=='missing':bad['prediction']['frames'].pop(0)
            if mutation=='endpoint':bad['timing_prediction']['value_ps']+=1
            if mutation=='observation':bad['Simulator_timestamps_used_as_inputs']=True
            with self.assertRaises(ValueError):normalize(candidate,bad,good,self.snapshot['id'])

    def test_plan_callback_checks_binding_target_and_actual_core_count(self):
        candidate=self.candidate()
        evaluator=AnalyzerEvaluator(self.snapshot,{'plan':candidate},self.root/'callback')
        self.assertEqual(evaluator({'id':'missing'})['status'],'needs_analyzer_binding')
        result=evaluator({'id':'plan','planned_compute_tiles':1})
        self.assertEqual(result['status'],'conditional_estimate')
        with self.assertRaisesRegex(ValueError,'core count'):evaluator({'id':'plan','planned_compute_tiles':2})
        other=AnalyzerEvaluator(self.snapshot,{},self.root/'other-target',architecture='AIE-ML')
        self.assertEqual(other({'id':'missing'})['status'],'unsupported')

    def test_dense_comparison_groups_only_FIFO_preserving_interleavings(self):
        space=schedule_candidates(self.save('base.json',self.request),self.root/'space')
        batch=evaluate_candidates(space['candidates'],self.snapshot,self.root/'space-runs')
        ranking=rank_candidates(space['candidates'],batch['evaluations'])
        self.assertEqual(len(ranking['groups']),1)
        self.assertFalse(ranking['cross_group_ranking_performed'])
        changed=deepcopy(self.request);changed['source_csv']['A']=self.save('different.csv',csv_text(payload=99))
        different=self.candidate(changed,'different')
        result=evaluate_candidates([different],self.snapshot,self.root/'different')['evaluations'][0]
        first=space['candidates'][0]
        ranking=rank_candidates([first,different],[batch['evaluations'][0],result])
        self.assertEqual(len(ranking['groups']),2)

    def test_explicit_dense_policy_refuses_STALL_instead_of_erasing_its_cost(self):
        request=deepcopy(self.request);text=csv_text();text=text.replace('DATA,1,0,-1','STALL:2\nDATA,1,0,-1',1)
        request['source_csv']['A']=self.save('stalled.csv',text)
        candidate=self.candidate(request);candidate=make_candidate(candidate['request_ref'],plan_id='dense',policy='dense_packet_interleaving')
        result=evaluate_candidates([candidate],self.snapshot,self.root/'stalled')['evaluations'][0]
        self.assertEqual(result['status'],'invalid_input')

    def test_packet_registry_preserves_headers_payload_TLAST_and_per_destination_order(self):
        report=schedule_candidates(self.save('base.json',self.request),self.root/'packets')
        expected=packet_signature(dense_packets(read_ref(self.request['source_csv']['A'])))
        self.assertGreater(len(report['candidates']),1)
        for candidate in report['candidates']:
            request=strict_json(read_ref(candidate['request_ref']))
            self.assertEqual(packet_signature(dense_packets(read_ref(request['source_csv']['A']))),expected)
        registry=PacketRuleRegistry();registry.register(PacketRule('bad.reverse_FIFO',1,lambda p:list(reversed(p))))
        with self.assertRaisesRegex(ValueError,'FIFO order'):schedule_candidates(self.save('base2.json',self.request),self.root/'bad-space',registry=registry)

    def test_preparation_does_not_silently_replace_supplied_metadata_or_compile_facts(self):
        extraction=self.save('extraction.json',{'compiler_inputs':self.request['compiler_inputs']})
        preparation=self.save('preparation.json',{'metadata_ref':{'path':'old'},'compiler_inputs':self.request['compiler_inputs']})
        with self.assertRaisesRegex(ValueError,'null metadata'):
            prepare_compiled_candidate(self.snapshot,extraction['path'],preparation['path'],self.root/'bad-prepare')
        self.assertFalse((self.root/'bad-prepare').exists())

    def test_costs_only_does_not_invent_a_first_frame_or_batch_time(self):
        request={k:self.request[k] for k in ('schema','compiler_inputs','program_binding','cost_registry')}
        request['mode']='costs';candidate=self.candidate(request)
        result=evaluate_candidates([candidate],self.snapshot,self.root/'costs-only')['evaluations'][0]
        self.assertEqual(result['status'],'costs_resolved');self.assertIsNone(result['conditional_latency_ns'])
        self.assertFalse(rank_candidates([candidate],[result])['groups'])

    def test_dense_policy_does_not_hide_a_changed_per_destination_invocation_order(self):
        a=self.candidate(plan='first');a=make_candidate(a['request_ref'],plan_id='first',policy='dense_packet_interleaving')
        packets=dense_packets(read_ref(self.request['source_csv']['A']))
        changed=deepcopy(self.request)
        text='CMD,D,TLAST,TKEEP\n'+''.join(','.join(row)+'\n' for p in reversed(packets) for row in p)
        changed['source_csv']['A']=self.save('reverse-invocations.csv',text)
        b=self.candidate(changed,'second');b=make_candidate(b['request_ref'],plan_id='second',policy='dense_packet_interleaving')
        batch=evaluate_candidates([a,b],self.snapshot,self.root/'changed-invocations')
        self.assertEqual(len(rank_candidates([a,b],batch['evaluations'])['groups']),2)

    def test_first_complete_frame_and_batch_objectives_can_choose_different_candidates(self):
        a=self.candidate();b=make_candidate(a['request_ref'],plan_id='other')
        batch=evaluate_candidates([a,b],self.snapshot,self.root/'objectives')
        first,second=deepcopy(batch['evaluations'])
        first.update(conditional_latency_ns=5,conditional_frame_completion_ns=[5,12])
        second.update(conditional_latency_ns=6,conditional_frame_completion_ns=[6,10])
        r=rank_candidates([a,b],[first,second],metric='first_output')
        self.assertEqual(r['groups'][0]['selected_candidate_id'],a['id'])
        r=rank_candidates([a,b],[first,second],metric='batch_complete')
        self.assertEqual(r['groups'][0]['selected_candidate_id'],b['id'])


if __name__=='__main__':unittest.main()
