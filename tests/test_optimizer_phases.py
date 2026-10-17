import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from Optimizer.contracts import make_contract,identity,Target
from Optimizer.ir_planner import make_ir_plan
from Optimizer.execution_ir import build_execution
from Optimizer.builtin_rules import default_registry
from Optimizer.semantics import Spec,build
from Optimizer.phase_model import software_binding,lower_phases,evaluate_graph
from Optimizer.phase_costs import evaluate_phases
from Optimizer.analyzer_bridge import REGISTRY_SCHEMA,FAMILY_SCHEMA,DIMENSIONS


def bind(c,p,frames=1,slots=1):
    b=software_binding(c,p,frames=frames,slots=slots)
    b['clocks']={k:{'period_ps':1000,'phase_ps':0} for k in ('core','transfer')}
    b['id']=identity({k:v for k,v in b.items() if k!='id'});return b


def direct_costs(g):
    return {c['component_id']:100000 if c['mechanism']=='complete_region' else
            10000 if c['mechanism']=='whole_frame_transfer' else 0 for c in g['components']}


def registry_for(graphs,directory,duration=None):
    path=Path(directory)/'software_assumptions.json';path.write_text('{"role":"software_fixture","native":false}',encoding='utf-8')
    ref={'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    components={s['component_id']:s for g in graphs for s in g['components']}
    reg={'schema':REGISTRY_SCHEMA,'registry_id':'explicit-test-assumptions','supported_mechanisms':sorted({s['mechanism'] for s in components.values()}),'entries':[]}
    for cid,s in components.items():
        value=duration(s) if duration else (100000 if s['mechanism']=='complete_region' else 10000 if s['mechanism']=='whole_frame_transfer' else 0)
        reg['entries'].append({'cost_id':cid,'mechanism':s['mechanism'],
            'applicability':{d:{'kind':'exact','value':s[d]} for d in DIMENSIONS},
            'cost_family':{'schema':FAMILY_SCHEMA,'origin':'software_conditional','model_refs':[],
                'inline_models':[{'model_id':'member-a','parameters':{'duration_ps':value}},
                                 {'model_id':'member-b','parameters':{'duration_ps':2*value}}],
                'correlation_group':'shared-fixture-family'},
            'evidence_refs':[{'role':'software_fixture','ref':ref}],
            'discrepancy':{'status':'uncalibrated','bound_ps':None,'evidence_refs':[]}})
    return reg,{'cost_by_component':{},'member_by_group':{'shared-fixture-family':'member-a'}}


class PhaseTests(unittest.TestCase):
    def test_three_inputs_single_and_double_slot_pipeline_matches_manual_timeline(self):
        c=make_contract();p=make_ir_plan(c)
        for slots,expected in ((1,[120000,230000,340000]),(2,[120000,220000,320000])):
            g=lower_phases(c,p,bind(c,p,frames=3,slots=slots));r=evaluate_graph(g,direct_costs(g))
            self.assertEqual(r['frame_completion_ps'],expected)
            self.assertFalse(r['native_admitted'])

    def test_four_input_gather_deferral_exposes_independent_work(self):
        c=make_contract();s=Spec(hidden_relu=False);r=default_registry();state=build_execution(build(s)['materialized_layers'].expression,s,2,'flat')
        ctx={'spec':s,'target':Target(),'groups':2}
        params=next(p for rule,p in r.execution.proposals(state,ctx) if rule=='schedule.defer_gather_acquire')
        _,step=r.execution.apply(state,ctx,'schedule.defer_gather_acquire',params)
        early=make_ir_plan(c,groups=2);deferred=make_ir_plan(c,groups=2,execution_steps=[step])
        times=[]
        for plan in (early,deferred):
            graph=lower_phases(c,plan,bind(c,plan));times.append(evaluate_graph(graph,direct_costs(graph))['first_output_ps'])
        self.assertEqual(times,[230000,130000])

    def test_shared_slot_cannot_be_reused_before_consumer_release(self):
        c=make_contract();p=make_ir_plan(c,groups=2);b=bind(c,p,frames=2)
        lid='fused_1.y0';src='fused_0.y';link=b['links'][lid]
        del b['resource_orders'][link['resource']];link.update(kind='shared',resource=None)
        b['buffers'][lid]=copy.deepcopy(b['buffers'][src]);b['buffers'][lid]['alias_of']=src
        b['id']=identity({k:v for k,v in b.items() if k!='id'})
        g=lower_phases(c,p,b)
        rules={r['id']:r for r in g['graph']['rules']}
        second_output_acquire='actor:fused_0:1:3:ready'
        parents={x['event_id'] for x in rules[second_output_acquire]['parents']}
        release_index=next(i for i,a in enumerate(p['execution']['kernels'][1]['actions']) if a=={'op':'release','port':'y0'})
        self.assertIn(f'actor:fused_1:0:{release_index}',parents)
        self.assertEqual(evaluate_graph(g,direct_costs(g))['first_output_ps'],220000)

    def test_nonmodulo_slot_sequence_is_honored(self):
        c=make_contract();p=make_ir_plan(c);b=bind(c,p,frames=3,slots=2)
        b['buffers']['fused_0.y']['uses']=[1,1,0]
        b['id']=identity({k:v for k,v in b.items() if k!='id'})
        g=lower_phases(c,p,b);rules={r['id']:r for r in g['graph']['rules']}
        self.assertIn('transfer:Y:0:done',{x['event_id'] for x in rules['actor:fused_0:1:3:ready']['parents']})

    def test_incomplete_resource_order_alias_and_deadlock_fail_closed(self):
        c=make_contract();p=make_ir_plan(c)
        for mutation in ('missing','alias','cycle','native'):
            b=bind(c,p,frames=2)
            if mutation=='missing':b['resource_orders'].pop('channel:fused_0.x')
            elif mutation=='alias':b['buffers']['fused_0.w0']['slot_ids']=b['buffers']['fused_0.x']['slot_ids']
            elif mutation=='cycle':b['resource_orders']['channel:fused_0.x'].reverse()
            else:b['origin']='compiled_and_native_approved'
            b['id']=identity({k:v for k,v in b.items() if k!='id'})
            with self.assertRaises(ValueError):lower_phases(c,p,b)
        with self.assertRaisesRegex(ValueError,'budget'):lower_phases(c,p,bind(c,p,frames=3),maximum_events=10)

    def test_shared_transfer_channel_serializes_services(self):
        c=make_contract();p=make_ir_plan(c);b=bind(c,p)
        b['resource_orders']={'one-channel':[{'link':lid,'frame':0} for lid in b['links']]}
        for link in b['links'].values():link['resource']='one-channel'
        b['id']=identity({k:v for k,v in b.items() if k!='id'})
        g=lower_phases(c,p,b)
        self.assertEqual(evaluate_graph(g,direct_costs(g))['first_output_ps'],140000)

    def test_registry_missing_cost_and_correlated_selection(self):
        c=make_contract();p=make_ir_plan(c);b=bind(c,p);g=lower_phases(c,p,b)
        with tempfile.TemporaryDirectory() as d:
            reg,selection=registry_for([g],d)
            a=evaluate_phases(c,p,b,reg,selection)
            self.assertEqual(a['conditional_latency_ns'],120);self.assertIsNone(a['latency_ns'])
            self.assertFalse(a['native_admitted'])
            selection['member_by_group']['shared-fixture-family']='member-b'
            self.assertEqual(evaluate_phases(c,p,b,reg,selection)['conditional_latency_ns'],240)
            selection['member_by_group']={}
            self.assertEqual(evaluate_phases(c,p,b,reg,selection)['status'],'needs_model_selection')
            reg['entries'].pop()
            self.assertEqual(evaluate_phases(c,p,b,reg,selection)['status'],'needs_measurement')

    def test_changed_evidence_and_invalid_duration_are_rejected(self):
        c=make_contract();p=make_ir_plan(c);b=bind(c,p);g=lower_phases(c,p,b)
        with tempfile.TemporaryDirectory() as d:
            reg,selection=registry_for([g],d)
            reg['entries'][0]['cost_family']['inline_models'][0]['parameters']['duration_ps']=-1
            with self.assertRaises(ValueError):evaluate_phases(c,p,b,reg,selection)
            reg['entries'][0]['cost_family']['inline_models'][0]['parameters']['duration_ps']=0
            (Path(d)/'software_assumptions.json').write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):evaluate_phases(c,p,b,reg,selection)

    def test_rehashed_semantic_forgery_is_rejected_before_costing(self):
        c=make_contract();p=make_ir_plan(c);p['expression']['op']='unregistered_algorithm'
        p['id']=identity({k:v for k,v in p.items() if k!='id'})
        with self.assertRaisesRegex(ValueError,'reproducible'):software_binding(c,p)

    def test_changed_lowered_graph_identity_is_rejected(self):
        c=make_contract();p=make_ir_plan(c);g=lower_phases(c,p,bind(c,p))
        g['output_events'][0]='fake'
        with self.assertRaises(ValueError):evaluate_graph(g,direct_costs(g))


if __name__=='__main__':unittest.main()
