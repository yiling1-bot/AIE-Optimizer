import copy
from pathlib import Path
import shutil,tempfile,unittest
from Optimizer.backend import generate
from Optimizer.builtin_rules import default_registry
from Optimizer.contracts import make_contract,Target
from Optimizer.execution_ir import build_execution,validate_execution
from Optimizer.ir_planner import make_ir_plan
from Optimizer.registry import Match
from Optimizer.runner import cpu_check
from Optimizer.semantics import Spec,build


def fixture(spec,groups=1,columns=False):
    r=default_registry();c=make_contract(spec)
    rule='tensor.partition_outputs_rematerialize' if columns else 'tensor.partition_rows'
    expr,step=r.apply(build(spec)['materialized_layers'].expression,spec,rule,Match((),(('chunk',8 if columns else 4),)))
    context={'spec':spec,'target':Target(),'groups':groups};state=build_execution(expr,spec,groups,'flat');trace=[]
    for rid in ('lower.fuse_quantized_matmul','memory.bind_output_destination'):
        params=next(p for rule,p in r.execution.proposals(state,context) if rule==rid)
        state,proof=r.execution.apply(state,context,rid,params);trace.append(proof)
    return r,c,expr,step,context,state,trace


class DestinationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_disjoint_destination_writes_preserve_multishape_multiframe_values(self):
        with tempfile.TemporaryDirectory() as temp:
            for shape in ((8,16,16,24),(16,24,16,16)):
                for relu in (False,True):
                    for groups in (1,2):
                        s=Spec(*shape,relu);r,c,e,semantic,ctx,state,trace=fixture(s,groups)
                        # A group with only one row slice has no concat to forward.
                        options=[p for rule,p in r.execution.proposals(state,ctx) if rule=='memory.forward_row_concat_destination']
                        if not options:continue
                        _,proof=r.execution.apply(state,ctx,'memory.forward_row_concat_destination',options[0])
                        before=make_ir_plan(c,[semantic],groups,execution_steps=trace)
                        p=make_ir_plan(c,[semantic],groups,execution_steps=trace+[proof])
                        self.assertLess(sum(x['arena_bytes'] for x in p['lowering']),sum(x['arena_bytes'] for x in before['lowering']))
                        self.assertTrue(all(x['forwarded_output_slices']>=2 for x in p['lowering']))
                        d=Path(temp)/f'{shape[0]}_{relu}_{groups}';generate(c,p,d,frames=4)
                        self.assertEqual(cpu_check(d)['status'],'passed')

    def test_overlap_shared_use_and_column_partition_are_rejected(self):
        r,c,e,semantic,ctx,state,trace=fixture(Spec(8,16,16,24))
        params=next(p for rule,p in r.execution.proposals(state,ctx) if rule=='memory.forward_row_concat_destination')
        good,_=r.execution.apply(state,ctx,'memory.forward_row_concat_destination',params)
        for kind in ('overlap','incomplete','shared','wrong_axis'):
            bad=copy.deepcopy(good);dag=bad['regions'][0]
            pieces=[op for op in dag['operators'] if 'destination_offset' in op]
            join=next(op for op in dag['operators'] if op.get('concat_forwarded'))
            if kind=='overlap':pieces[1]['destination_offset']=0
            elif kind=='incomplete':pieces[1]['destination_offset']+=32
            elif kind=='shared':join['args'][1]=copy.deepcopy(join['args'][0])
            else:join['axis']=1
            with self.assertRaises(ValueError):validate_execution(bad,ctx)
        r,c,e,semantic,ctx,state,trace=fixture(Spec(8,16,16,24),columns=True)
        with self.assertRaisesRegex(ValueError,'DENY'):
            r.execution.apply(state,ctx,'memory.forward_row_concat_destination',{'regions':[0]})


if __name__=='__main__':unittest.main()
