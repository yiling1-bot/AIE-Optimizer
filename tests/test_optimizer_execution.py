import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from Optimizer.backend import generate
from Optimizer.builtin_rules import default_registry
from Optimizer.contracts import make_contract,Target
from Optimizer.execution_ir import build_execution,validate_execution
from Optimizer.execution_rules import fusion_candidates
from Optimizer.ir_planner import make_ir_plan,validate_ir_plan,enumerate_ir_plans
from Optimizer.registry import Decision,Match,Rule,walk
from Optimizer.runner import cpu_check,check_artifact
from Optimizer.semantics import Spec,build,partition_output


def scenario(spec=Spec(hidden_relu=False),groups=2,target=Target(),composed=True):
    r=default_registry();expr=build(spec)['materialized_layers'].expression;steps=[]
    if composed:
        for stage,chunk in [('L0',8),('L1',8)]:
            path,_=next((p,n) for p,n in walk(expr) if n.op=='matmul' and n.stage==stage)
            expr,step=r.apply(expr,spec,'tensor.partition_k_int32',Match(path,(('chunk',chunk),)))
            steps.append(step)
        expr,step=r.apply(expr,spec,'tensor.partition_rows',Match((),(('chunk',4),)));steps.append(step)
    state=build_execution(expr,spec,groups,'flat')
    return r,steps,state,{'spec':spec,'target':target,'groups':groups}


def apply_kind(r,state,ctx,rule):
    params=next(p for rid,p in r.execution.proposals(state,ctx) if rid==rule)
    return r.execution.apply(state,ctx,rule,params)


class ExecutionTests(unittest.TestCase):
    def test_fusion_preserves_signature_macs_and_reduces_storage(self):
        r,steps,state,ctx=scenario(Spec(8,16,24,16,False))
        fused,step=apply_kind(r,state,ctx,'lower.fuse_quantized_matmul')
        self.assertEqual(validate_execution(state,ctx),validate_execution(fused,ctx))
        c=make_contract(ctx['spec']);a=make_ir_plan(c,steps,2);b=make_ir_plan(c,steps,2,execution_steps=[step])
        self.assertEqual(sum(x['scheduled_macs'] for x in a['lowering']),sum(x['scheduled_macs'] for x in b['lowering']))
        self.assertLess(sum(x['arena_bytes'] for x in b['lowering']),sum(x['arena_bytes'] for x in a['lowering']))
        self.assertTrue(all(x['fused_quantized_regions']>0 for x in b['lowering']))

    def test_shared_producers_not_removed_by_fusion(self):
        s=Spec(8,16,24,16);expr=partition_output(build(s)['materialized_layers'].expression,1,8,s,False)
        state=build_execution(expr,s,1,'flat');dag=state['regions'][0]
        matches=fusion_candidates(dag)
        self.assertTrue(matches)
        used={v['base'] for op in dag['operators'] for v in op['args']}
        self.assertTrue(used)
        # Materialized INT8 Q0 remains shared by both output regions.
        r=default_registry();ctx={'spec':s,'target':Target(),'groups':1}
        fused,_=apply_kind(r,state,ctx,'lower.fuse_quantized_matmul')
        self.assertEqual(validate_execution(fused,ctx),validate_execution(state,ctx))
        self.assertEqual(sum(op['kind']=='fused_qmm' for op in fused['regions'][0]['operators']),3)

    def test_late_acquire_is_after_compute_and_balanced(self):
        r,_,state,ctx=scenario()
        deferred,_=apply_kind(r,state,ctx,'schedule.defer_gather_acquire')
        kernel=deferred['kernels'][-1];a=kernel['actions']
        compute=next(i for i,x in enumerate(a) if x['op']=='compute')
        for port in kernel['inputs']:
            if port['async']:
                acquire=next(i for i,x in enumerate(a) if x=={'op':'acquire','port':port['name']})
                read=next(i for i,x in enumerate(a) if x.get('from')==port['name'])
                release=next(i for i,x in enumerate(a) if x=={'op':'release','port':port['name']})
                self.assertLess(compute,acquire);self.assertLess(acquire,read);self.assertLess(read,release)
        self.assertEqual(validate_execution(state,ctx),validate_execution(deferred,ctx))

    def test_protocol_rejects_missing_release_early_read_and_duplicate_output(self):
        r,_,state,ctx=scenario();state,_=apply_kind(r,state,ctx,'schedule.defer_gather_acquire')
        for mutation in ('release','acquire','overlap','cycle'):
            bad=copy.deepcopy(state);k=bad['kernels'][-1]
            if mutation in ('release','acquire'):
                k['actions']=[a for a in k['actions'] if a!={'op':mutation,'port':'y0'}]
            elif mutation=='overlap':
                next(a for a in k['actions'] if a.get('from')=='own')['offset']=0
            else:next(p for p in k['inputs'] if p['name']=='y0')['source']='kernel:'+k['id']
            with self.assertRaises(ValueError,msg=mutation):validate_execution(bad,ctx)

    def test_split_adds_real_tile_and_obeys_cap(self):
        r,_,state,ctx=scenario();split,_=apply_kind(r,state,ctx,'deploy.extract_local_compute')
        self.assertEqual(len(split['kernels']),3)
        self.assertEqual(sum(a['op']=='compute' for k in split['kernels'] for a in k['actions']),2)
        self.assertEqual(validate_execution(state,ctx),validate_execution(split,ctx))
        ctx={**ctx,'target':Target(max_compute_tiles=2)}
        with self.assertRaisesRegex(ValueError,'cap'):apply_kind(r,state,ctx,'deploy.extract_local_compute')

    def test_fusion_can_rescue_storage_illegal_seed_and_search_starts_with_baseline(self):
        from Optimizer.search import choose
        limited=make_contract(target=Target(tile_memory_bytes=22000,max_compute_tiles=1))
        plans,report=enumerate_ir_plans(limited,max_candidates=1)
        self.assertTrue(plans)
        self.assertTrue(all(p['execution_steps'] for p in plans))
        self.assertTrue(report['lowering_refused'])
        ordinary,_=enumerate_ir_plans(make_contract(),max_candidates=1)
        selected=choose(ordinary,[])
        self.assertEqual(selected['steps'],[])
        self.assertEqual(selected['execution_steps'],[])

    def test_execution_certificate_and_unknown_conditions_fail_closed(self):
        r,steps,state,ctx=scenario();state,step=apply_kind(r,state,ctx,'lower.fuse_quantized_matmul')
        c=make_contract(ctx['spec']);p=make_ir_plan(c,steps,2,execution_steps=[step])
        validate_ir_plan(c,p)
        bad=copy.deepcopy(p);bad['execution_steps'][0]['version']=9
        with self.assertRaisesRegex(ValueError,'version'):validate_ir_plan(c,bad)
        bad=copy.deepcopy(p);bad['execution']['kernels'][-1]['tile']=[9,3]
        with self.assertRaises(ValueError):validate_ir_plan(c,bad)
        r.execution.register(Rule('test.unknown',1,'test',(),'missing capability evidence',
            lambda s,c:({},),lambda s,c,p:Decision('UNKNOWN','capability unknown'),lambda s,c,p:s))
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):r.execution.apply(state,ctx,'test.unknown',{})

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_fused_cpu_multiple_shapes_groups_relu_and_extreme_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shape in ((8,16,24,16),(16,24,16,24)):
                for relu in (False,True):
                    for groups in (1,2):
                        s=Spec(*shape,relu);r,steps,state,ctx=scenario(s,groups)
                        state,step=apply_kind(r,state,ctx,'lower.fuse_quantized_matmul')
                        p=make_ir_plan(make_contract(s),steps,groups,execution_steps=[step])
                        d=Path(tmp)/f'{shape[0]}_{relu}_{groups}';generate(make_contract(s),p,d,frames=4)
                        self.assertNotIn(b'\r\n',(d/'data/X.txt').read_bytes())
                        self.assertEqual(cpu_check(d)['status'],'passed')

    def test_new_execution_plugin_reaches_codegen_and_replay(self):
        r,steps,state,ctx=scenario()
        # This independent plugin changes a physical placement without adding
        # a mode to the planner or code generator.
        r.execution.register(Rule('test.move_leaf',1,'test',(),'feed-forward edges and tensor ownership unchanged',
            lambda s,c:({'tile':[0,4]},),lambda s,c,p:Decision('ALLOW','target tile available'),
            lambda s,c,p:{**s,'kernels':[{**s['kernels'][0],'tile':p['tile']}]+s['kernels'][1:]}))
        state,step=r.execution.apply(state,ctx,'test.move_leaf',{'tile':[0,4]})
        c=make_contract(ctx['spec']);p=make_ir_plan(c,steps,2,registry=r,execution_steps=[step])
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'plugin';generate(c,p,d,registry=r)
            self.assertEqual(check_artifact(d,r)[1]['kernel_bindings'][0]['tile'],[0,4])
            self.assertIn('adf::tile(0,4)',(d/'graph.hpp').read_text())
            with self.assertRaisesRegex(ValueError,'unknown execution'):check_artifact(d)


if __name__=='__main__':unittest.main()
