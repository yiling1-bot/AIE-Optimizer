import copy
from pathlib import Path
import shutil,tempfile,unittest

from Optimizer.backend import generate
from Optimizer.builtin_rules import default_registry
from Optimizer.contracts import Target,make_contract
from Optimizer.execution_ir import build_execution,validate_execution
from Optimizer.ir_planner import make_ir_plan
from Optimizer.runner import cpu_check
from Optimizer.semantics import Spec,build


def seed(spec,groups=1):
    r=default_registry();ctx={'spec':spec,'target':Target(),'groups':groups}
    state=build_execution(build(spec)['materialized_layers'].expression,spec,groups,'flat')
    rule='lower.fuse_quantized_matmul'
    params=next(p for rid,p in r.execution.proposals(state,ctx) if rid==rule)
    state,step=r.execution.apply(state,ctx,rule,params)
    return r,ctx,state,[step]


def apply(r,ctx,state,rule,grid=None):
    params=next(p for rid,p in r.execution.proposals(state,ctx) if rid==rule and (grid is None or p.get('grid')==grid))
    return r.execution.apply(state,ctx,rule,params)


class KernelRuleTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_grid_and_destination_compositions_match_independent_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shape,groups in [((8,16,16,16),1),((16,24,32,24),2)]:
                for relu in (False,True):
                    for grid in ([1,2],[2,1],[2,2]):
                        spec=Spec(*shape,relu);r,ctx,state,steps=seed(spec,groups)
                        state,step=apply(r,ctx,state,'lower.tile_accumulator_grid',grid);steps.append(step)
                        state,step=apply(r,ctx,state,'memory.bind_output_destination');steps.append(step)
                        c=make_contract(spec);p=make_ir_plan(c,groups=groups,execution_steps=steps)
                        d=Path(tmp)/f'{shape[0]}_{relu}_{grid[0]}_{grid[1]}'
                        generate(c,p,d,frames=4)
                        self.assertEqual(cpu_check(d)['status'],'passed')
                        self.assertTrue(all(x['direct_output'] for x in p['lowering']))
                        self.assertEqual(sum(x['scheduled_macs'] for x in p['lowering']),spec.m*spec.k*spec.hidden+spec.m*spec.hidden*spec.n)

    def test_destination_removes_copy_and_storage_without_changing_values(self):
        spec=Spec(hidden_relu=False);r,ctx,state,steps=seed(spec)
        original=validate_execution(state,ctx)
        direct,step=apply(r,ctx,state,'memory.bind_output_destination');steps.append(step)
        self.assertEqual(validate_execution(direct,ctx),original)
        baseline=make_ir_plan(make_contract(spec),execution_steps=steps[:-1])
        plan=make_ir_plan(make_contract(spec),execution_steps=steps)
        self.assertLess(plan['lowering'][0]['arena_bytes'],baseline['lowering'][0]['arena_bytes'])
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'direct';generate(make_contract(spec),plan,d)
            code=(d/'compute.hpp').read_text()
            region=code[code.index('inline void run_group_0'):code.index('inline void execute')]
            self.assertNotIn('op_pack8(',region)

    def test_unsafe_binding_and_incomplete_grid_are_rejected(self):
        r,ctx,state,_=seed(Spec(4,8,8,8))
        with self.assertRaisesRegex(ValueError,'DENY'):
            r.execution.apply(state,ctx,'lower.tile_accumulator_grid',{'regions':[0],'grid':[2,2]})
        for kind in ('intermediate_output','grid_tail','unknown_annotation'):
            bad=copy.deepcopy(state);op=bad['regions'][0]['operators'][0]
            if kind=='intermediate_output':op['output_direct']=True
            elif kind=='grid_tail':op['accumulator_grid']=[2,2]
            else:op['arbitrary_schedule']='unsafe'
            with self.assertRaises(ValueError,msg=kind):validate_execution(bad,ctx)

    def test_storage_and_grid_rewrites_commute_and_replay(self):
        spec=Spec();r,ctx,base,steps=seed(spec)
        a,sa=apply(r,ctx,base,'memory.bind_output_destination')
        a,sb=apply(r,ctx,a,'lower.tile_accumulator_grid',[2,2])
        b,sc=apply(r,ctx,base,'lower.tile_accumulator_grid',[2,2])
        b,sd=apply(r,ctx,b,'memory.bind_output_destination')
        self.assertEqual(a,b)
        raw=build_execution(build(spec)['materialized_layers'].expression,spec,1,'flat')
        self.assertEqual(r.execution.replay(raw,ctx,steps+[sa,sb]),a)
        self.assertEqual(r.execution.replay(raw,ctx,steps+[sc,sd]),b)


if __name__=='__main__':unittest.main()
