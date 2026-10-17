import copy
from dataclasses import replace
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from Optimizer.backend import generate
from Optimizer.builtin_rules import default_registry
from Optimizer.contracts import make_contract, Target
from Optimizer.fixtures import reference
from Optimizer.ir_backend import lower_program
from Optimizer.ir_planner import make_ir_plan, validate_ir_plan, enumerate_ir_plans
from Optimizer.registry import Decision, Match, Rule, walk
from Optimizer.runner import cpu_check, check_artifact
from Optimizer.semantics import Spec, Expr, Alternative, Selection, build, evaluate, partition_output
from Optimizer.search import campaign


def composed(spec, registry=None):
    registry=registry or default_registry()
    expr=build(spec)['materialized_layers'].expression;steps=[]
    # Apply at matched local nodes, including both layers; no whole-program mode.
    for stage in ('L0','L1'):
        path,node=next((p,n) for p,n in walk(expr) if n.op=='matmul' and n.stage==stage)
        expr,step=registry.apply(expr,spec,'tensor.partition_k_int32',Match(path,(('chunk',8),)))
        steps.append(step)
    expr,step=registry.apply(expr,spec,'tensor.partition_rows',Match((),(('chunk',4),)))
    return expr,steps+[step]


class RegistryTests(unittest.TestCase):
    def test_composition_preserves_quantization_boundaries_and_values(self):
        rng=random.Random(51)
        for relu in (False,True):
            spec=Spec(8,16,24,16,relu);expr,steps=composed(spec)
            self.assertEqual(default_registry().replay(spec,steps),expr)
            for values in ((-2,-1,0,1,2),(-128,-127,0,126,127)):
                inp={k:[[rng.choice(values) for _ in range(n)] for _ in range(m)]
                     for k,m,n in [('X',8,16),('W0',16,24),('W1',24,16)]}
                alt=Alternative('candidate',expr,(),('none',),'')
                actual=evaluate({'candidate':alt},Selection('candidate','dot_product','dot_product','none'),spec,inp)
                self.assertEqual(actual,reference(inp['X'],inp['W0'],inp['W1'],relu))

    def test_unknown_and_illegal_quantization_fail_closed(self):
        spec=Spec();expr=build(spec)['materialized_layers'].expression;r=default_registry()
        r.register(Rule('unknown',1,'test',(),'missing premise',lambda e,s: ({},),
                        lambda e,s,p: Decision('UNKNOWN','range proof missing'),lambda e,s,p:e))
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):
            r.apply(expr,spec,'unknown',Match())
        with self.assertRaisesRegex(ValueError,'DENY'):
            r.apply(expr,spec,'tensor.partition_k_int32',Match((),(('chunk',8),)))
        r.register(Rule('bad_q',1,'test',(),'intentionally invalid test rule',lambda e,s: ({},),
            lambda e,s,p: Decision('ALLOW','test'),
            lambda e,s,p: Expr('sum',tuple(Expr('quantize',(a,)) for a in e.args))))
        split,step=r.apply(expr,spec,'tensor.partition_k_int32',Match((0,),(('chunk',8),)))
        with self.assertRaisesRegex(ValueError,'INT32'):
            r.apply(split,spec,'bad_q',Match((0,)))

    def test_trace_tamper_and_version_mismatch_rejected(self):
        c=make_contract(Spec(8,16,24,16));_,steps=composed(Spec(**c['spec']))
        p=make_ir_plan(c,steps,2);validate_ir_plan(c,p)
        bad=copy.deepcopy(p);bad['steps'][0]['parameters']['chunk']=16
        with self.assertRaises(ValueError):validate_ir_plan(c,bad)
        bad=copy.deepcopy(p);bad['steps'][0]['version']=99
        with self.assertRaisesRegex(ValueError,'version'):validate_ir_plan(c,bad)
        bad=copy.deepcopy(p);bad['expression']['relu']=True
        with self.assertRaises(ValueError):validate_ir_plan(c,bad)

    def test_backend_revision_invalidates_cached_candidate(self):
        c=make_contract();p=make_ir_plan(c)
        with patch('Optimizer.ir_planner.implementation_fingerprint',return_value='different_backend'):
            self.assertNotEqual(make_ir_plan(c)['id'],p['id'])
            with self.assertRaises(ValueError):validate_ir_plan(c,p)

    def test_dag_cse_and_rematerialization_change_executed_work(self):
        s=Spec(8,16,24,16);seed=build(s)['materialized_layers'].expression
        shared=partition_output(seed,1,8,s,False);remat=partition_output(seed,1,8,s,True)
        _,one=lower_program(shared,s,1);_,two=lower_program(remat,s,1)
        self.assertEqual(one[0]['scheduled_macs'],s.m*s.k*s.hidden+s.m*s.hidden*s.n)
        self.assertEqual(two[0]['scheduled_macs'],2*s.m*s.k*s.hidden+s.m*s.hidden*s.n)
        self.assertLess(one[0]['arena_bytes'],sum(x['bytes'] for x in one[0]['storage']))

    def test_target_and_microtile_refusals(self):
        with self.assertRaisesRegex(ValueError,'count exceeds'):
            make_ir_plan(make_contract(target=Target(max_compute_tiles=1)),groups=2)
        with self.assertRaisesRegex(ValueError,'microtile'):
            make_ir_plan(make_contract(Spec(m=7)))

    def test_search_uses_registered_rules_without_modes(self):
        r=default_registry();entries,report=r.explore(Spec(8,16,24,16),12,2)
        self.assertGreater(len(entries),5)
        self.assertTrue(any(len(e['steps'])==2 for e in entries))
        self.assertTrue(report['edges'])
        plans,_=enumerate_ir_plans(make_contract(Spec(8,16,24,16)),max_candidates=6)
        self.assertTrue(plans);self.assertTrue(all('mode' not in p for p in plans))
        self.assertEqual({p['planned_compute_tiles'] for p in plans},{1,2,3})

    def test_registered_campaign_executes_and_resumes_without_mode_dispatch(self):
        c=make_contract(Spec(8,16,24,16))
        with tempfile.TemporaryDirectory() as tmp:
            with patch('Optimizer.search.cpu_check',return_value={'status':'failed'}) as host:
                result=campaign(c,tmp,'unused',registered=True,max_candidates=4,max_evaluations=2)
                self.assertEqual(host.call_count,2)
                self.assertEqual(len(result['records']),2)
            with patch('Optimizer.search.cpu_check',side_effect=AssertionError('must reuse')):
                result=campaign(c,tmp,'unused',registered=True,max_candidates=4,max_evaluations=2)
                self.assertEqual(result['historical_evaluations_included'],2)

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_new_plugin_rule_reaches_codegen_without_backend_change(self):
        r=default_registry()
        r.register(Rule('test.reverse_partials',1,'M02-exact-partial-reduction',(),
            'bounded INT32 sum permits permutation of its complete partials',
            lambda e,s: ({},) if e.op=='sum' and len(e.args)>1 else (),
            lambda e,s,p: Decision('ALLOW','permutation within bounded INT32 sum')
                if e.op=='sum' and not p else Decision('DENY','requires sum'),
            lambda e,s,p:replace(e,args=tuple(reversed(e.args)))))
        spec=Spec(8,16,24,16);expr,steps=composed(spec,r)
        path,_=next((p,n) for p,n in walk(expr) if n.op=='sum')
        expr,step=r.apply(expr,spec,'test.reverse_partials',Match(path));steps.append(step)
        c=make_contract(spec);p=make_ir_plan(c,steps,2,registry=r)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'plugin';generate(c,p,d,frames=3,registry=r)
            self.assertIn('adf::heap_size',(d/'graph.hpp').read_text())
            self.assertIn('opt::run_group_0',(d/'kernels.cpp').read_text())
            self.assertEqual(cpu_check(d,registry=r)['status'],'passed')
            with self.assertRaisesRegex(ValueError,'unknown rule'):check_artifact(d)

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_composed_generated_cpp_multiple_shapes_groups_and_relu(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shape in ((8,16,24,16),(16,24,16,24)):
                for relu in (False,True):
                    spec=Spec(*shape,relu);c=make_contract(spec);_,steps=composed(spec)
                    for groups in (1,2):
                        d=Path(tmp)/f'p_{shape[0]}_{relu}_{groups}'
                        generate(c,make_ir_plan(c,steps,groups),d,frames=3)
                        result=cpu_check(d)
                        self.assertEqual(result['status'],'passed',(d/'cpu_compile.log').read_text())

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_rematerialization_generates_exact_cpp(self):
        s=Spec(8,16,24,16);c=make_contract(s);r=default_registry()
        expr=build(s)['materialized_layers'].expression
        expr,step=r.apply(expr,s,'tensor.partition_outputs_rematerialize',Match((),(('chunk',8),)))
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'rematerialized';p=make_ir_plan(c,[step])
            generate(c,p,d,frames=3)
            self.assertEqual(cpu_check(d)['status'],'passed',(d/'cpu_compile.log').read_text())
            self.assertEqual(p['lowering'][0]['scheduled_macs'],2*s.m*s.k*s.hidden+s.m*s.hidden*s.n)

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_partial_copy_preserves_neighbor_values(self):
        s=Spec(4,8,8,8);code,_=lower_program(build(s)['materialized_layers'].expression,s,1)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'compute.hpp').write_text(code,encoding='utf-8')
            (d/'check.cpp').write_text('''#define AIE_OPT_HOST
#include "compute.hpp"
int main() {
  alignas(32) opt::I8 a[64],out[64]; alignas(32) opt::I32 b[64],sum[64];
  for(unsigned i=0;i<64;++i) {a[i]=i;b[i]=i;out[i]=111;sum[i]=777;}
  opt::op_copy8(opt::V8{a,2,8,8,4,0,0},out,8,0,0);
  opt::op_copy32(opt::V32{b,2,8,8,4,0,0},sum,8,0,0);
  for(unsigned i=0;i<64;++i)
    if(out[i]!=(i<16?i:111) || sum[i]!=(i<16?i:777)) return 1;
  return 0;
}
''',encoding='utf-8')
            exe=d/'check.exe'
            subprocess.run(['g++','-std=c++17','-O2',str(d/'check.cpp'),'-o',str(exe)],check=True,capture_output=True,timeout=120)
            subprocess.run([str(exe)],check=True,timeout=30)


if __name__=='__main__':unittest.main()
