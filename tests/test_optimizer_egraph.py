import copy
from dataclasses import replace
import itertools
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from Optimizer.backend import generate
from Optimizer.builtin_rules import default_registry
from Optimizer.contracts import make_contract,identity
from Optimizer.egraph import EGraph,check_certificate
from Optimizer.egraph_rules import Pattern,match_pattern,default_equality_registry,saturate
from Optimizer.egraph_planner import make_egraph_plan,validate_egraph_plan,enumerate_egraph_plans
from Optimizer.global_extract import Node,extract_dags,extract_regions,materialize
from Optimizer.registry import Match,Rule,Decision
from Optimizer.runner import cpu_check
from Optimizer.semantics import Spec,Expr,build,expression_id,cut,infer


class EGraphTests(unittest.TestCase):
    def graph(self,spec=Spec(8,16,16,16)):
        c=make_contract(spec);r=default_equality_registry();g=EGraph(spec,c['id'],r)
        root=g.add(build(spec)['materialized_layers'].expression)
        return c,g,root

    def test_congruence_rebuild_propagates_through_grandparents(self):
        c,g,root=self.graph();seed=g.terms[root];mm=seed.args[0].args[0].args[0]
        rhs,_=g.registry.apply(mm,g.spec,'tensor.partition_k_int32',Match((),(('chunk',8),)))
        other=replace(seed,args=(replace(seed.args[0],args=(replace(seed.args[0].args[0],args=(rhs,)),seed.args[0].args[1])),))
        oid=g.add(other)
        self.assertFalse(g.equivalent(root,oid))
        g.apply(expression_id(mm),'tensor.partition_k_int32',{'chunk':8})
        self.assertTrue(g.equivalent(root,oid))
        state=g.classes();g.rebuild();self.assertEqual(state,g.classes())
        cert=g.certificate([root],[other])
        restored,chosen=check_certificate(cert,g.spec,c['id'],g.registry)
        self.assertEqual(chosen,[other]);self.assertTrue(restored.equivalent(root,oid))

    def test_e_matching_sees_nonrepresentative_child_operator(self):
        c,g,root=self.graph();mat=g.terms[root].args[0]
        g.apply(expression_id(mat),'tensor.partition_k_int32',{'chunk':8})
        matches=list(match_pattern(g,g.find(root),Pattern('quantize',(Pattern('sum'),))))
        self.assertTrue(matches)
        self.assertEqual(matches[0][0].args[0].op,'sum')
        self.assertTrue(g.equivalent(g.add(matches[0][0]),root))

    def test_certificate_rejects_forged_equivalence_context_and_output(self):
        c,g,root=self.graph();cert=g.certificate([root],[g.terms[root]])
        for change in ('context','selected','proof'):
            bad=copy.deepcopy(cert)
            if change=='context':bad['context']='other_input_epoch'
            elif change=='selected':bad['selected']=[next(t['id'] for t in bad['terms'] if t['node']['op']=='input')]
            else:
                bad['equalities']=[{'lhs':root,'rhs':root,'step':{'rule':'missing','version':1,'path':[]}}]
            bad['id']=identity({k:v for k,v in bad.items() if k!='id'})
            with self.assertRaises(ValueError):check_certificate(bad,g.spec,c['id'],g.registry)

    def test_unknown_guard_and_cross_type_union_fail_closed(self):
        c,g,root=self.graph()
        g.registry.register(Rule('unknown',1,'test',(),'unknown premise',lambda e,s:({},),
            lambda e,s,p:Decision('UNKNOWN','numeric proof unavailable'),lambda e,s,p:e))
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):g.apply(root,'unknown',{})
        with self.assertRaisesRegex(ValueError,'shape/dtype'):
            g._union(root,expression_id(g.terms[root].args[0]))

    def test_bounded_saturation_then_codegen_certificate(self):
        c,g,root=self.graph()
        report=saturate(g,max_rounds=2,max_terms=250,max_applications=12)
        self.assertGreater(report['applications'],1);self.assertLessEqual(report['terms'],250)
        best,unknown,er=extract_regions(g,[root],max_states=5000,keep=3)
        self.assertTrue(best);self.assertFalse(unknown)
        self.assertEqual(best[0]['metrics']['latency_ns'],None)
        exprs=[materialize(g,b['selected'],[root])[0] for b in best]
        for expr in exprs:
            cert=g.certificate([root],[expr]);plan=make_egraph_plan(c,cert)
            validate_egraph_plan(c,plan)
            bad=copy.deepcopy(plan);bad['expression']['relu']=True
            with self.assertRaises(ValueError):validate_egraph_plan(c,bad)

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_new_shared_column_rule_combines_with_other_local_equality_and_emits(self):
        with tempfile.TemporaryDirectory() as temp:
            for relu in (False,True):
                c,g,root=self.graph(Spec(8,16,24,16,relu))
                seed=g.terms[root]
                # Local proof in L0 and independent root proof. Congruence composes
                # them without a single original whole-program rewrite trace.
                l0=seed.args[0].args[0].args[0]
                rhs,_=g.apply(expression_id(l0),'tensor.partition_k_int32',{'chunk':8})
                out,_=g.apply(root,'tensor.partition_columns_shared',{'chunk':8})
                def substitute(e):
                    if e==l0:return g.terms[rhs]
                    return replace(e,args=tuple(substitute(a) for a in e.args))
                expr=substitute(g.terms[out]);cert=g.certificate([root],[expr])
                for groups in (1,2):
                    dest=Path(temp)/f'{relu}_{groups}'
                    plan=make_egraph_plan(c,cert,groups)
                    generate(c,plan,dest,frames=3)
                    self.assertEqual(cpu_check(dest)['status'],'passed')


class GlobalExtractionTests(unittest.TestCase):
    def test_shared_dag_global_optimum_beats_independent_output_choice(self):
        nodes=[Node('a_local','A',(),6),Node('a_shared','A',('S',),1),
               Node('b_local','B',(),6),Node('b_shared','B',('S',),1),Node('shared','S',(),8)]
        local=[extract_dags(nodes,[r],keep=1)[0][0]['metrics']['cost'] for r in ('A','B')]
        best,_,report=extract_dags(nodes,['A','B'],keep=1)
        self.assertEqual(sum(local),12);self.assertEqual(best[0]['metrics']['cost'],10)
        self.assertEqual(report['optimality'],'exact_in_supplied_space')
        self.assertEqual(set(best[0]['selected'].values()),{'a_shared','b_shared','shared'})

    def test_cycles_unknown_and_budget_are_never_false_optimal_zero(self):
        nodes=[Node('self','A',('A',),0),Node('known','A',(),5),Node('unknown','A',(),None)]
        best,unknown,report=extract_dags(nodes,['A'])
        self.assertEqual(best[0]['metrics']['cost'],5);self.assertIsNone(unknown[0]['metrics']['cost'])
        self.assertGreater(report['cycle_refusals'],0)
        self.assertEqual(report['optimality'],'exact_in_known_cost_subspace')
        _,_,limited=extract_dags(nodes,['A'],max_states=1)
        self.assertEqual(limited['optimality'],'best_found_budget_limited')

    def test_resource_constraint_changes_global_selection(self):
        nodes=[Node('shareA','A',('S',),0),Node('copyA','A',(),6),
               Node('shareB','B',('S',),0),Node('copyB','B',(),6),Node('S','S',(),8)]
        def resource(choice):
            if 'S' in choice:return None  # shared live value exceeds capacity
            return {'cost':12,'peak':6}
        best,_,report=extract_dags(nodes,['A','B'],evaluate=resource)
        self.assertEqual(best[0]['metrics']['cost'],12);self.assertGreater(report['resource_refusals'],0)

    def test_exhaustive_independent_oracle_on_random_acyclic_spaces(self):
        rng=random.Random(20261015)
        for case in range(30):
            nodes=[];options={}
            for i in range(5):
                options[str(i)]=[]
                for j in range(2):
                    children=tuple(str(k) for k in range(i) if rng.random()<.4)
                    n=Node(f'{i}:{j}',str(i),children,rng.randrange(8));nodes.append(n);options[str(i)].append(n)
            values=[]
            for picks in itertools.product(*options.values()):
                chosen={n.eclass:n for n in picks};used=set();todo=['4','3']
                while todo:
                    c=todo.pop()
                    if c not in used:used.add(c);todo.extend(chosen[c].children)
                values.append(sum(chosen[c].cost for c in used))
            best,_,report=extract_dags(nodes,['4','3'],keep=1)
            self.assertEqual(best[0]['metrics']['cost'],min(values),case)
            self.assertTrue(report['search_exhausted'])

    def test_layout_conversion_is_a_real_costed_dependency(self):
        nodes=[Node('producer4','H:packed4',(),3),Node('convert','H:packed8',('H:packed4',),3),
               Node('native8','H:packed8',(),4),Node('consumer','Y',('H:packed8',),1)]
        best,_,_=extract_dags(nodes,['Y'],keep=1)
        self.assertEqual(best[0]['metrics']['cost'],5)
        self.assertNotIn('H:packed4',best[0]['selected'])

    def test_unknown_branch_not_pruned_after_known_incumbent(self):
        nodes=[Node('cheap','Y',(),1),Node('needsunknown','Y',('U',),10),Node('U','U',(),None)]
        best,unknown,report=extract_dags(nodes,['Y'],keep=1)
        self.assertEqual(best[0]['metrics']['cost'],1);self.assertEqual(len(unknown),1)

    def test_evaluator_discovered_unknown_is_retained_and_scopes_optimality(self):
        nodes=[Node('cheap','Y',(),1),Node('expensive','Y',(),10)]
        def callback(selected):return {'cost':1 if selected['Y']=='cheap' else None}
        best,unknown,report=extract_dags(iter(nodes),['Y'],keep=1,evaluate=callback)
        self.assertEqual(best[0]['metrics']['cost'],1);self.assertEqual(len(unknown),1)
        self.assertEqual(report['optimality'],'exact_in_known_cost_subspace')
        with self.assertRaisesRegex(ValueError,'coverage contract'):
            extract_dags(nodes,['Y'],keep=2,evaluate=callback,evaluator_costs_complete=True)

    def test_cost_after_fusion_does_not_drop_the_better_lowered_program(self):
        from Optimizer.cost_refinement import refine_for_cost
        from Optimizer.contracts import Target
        s=Spec(hidden_relu=False);c=make_contract(s);r=default_equality_registry();g=EGraph(s,c['id'],r)
        # Isolate the fusion projection from the later destination-slice rule.
        r.execution._rules.pop('memory.forward_row_concat_destination')
        seed=build(s)['materialized_layers'].expression;root=g.add(seed)
        row,_=g.apply(root,'tensor.partition_rows',{'chunk':8})
        before,_,_=extract_regions(g,[root],keep=1)
        self.assertNotEqual(materialize(g,before[0]['selected'],[root])[0],seed)
        after,_,report=extract_regions(g,[root],keep=1,
            refinement=lambda e:refine_for_cost(e,s,Target(),r))
        self.assertEqual(materialize(g,after[0]['selected'],[root])[0],seed)
        self.assertEqual(after[0]['metrics']['arena_bytes'],1024)
        self.assertEqual(report['evaluation_stage'],'after_registered_refinement')

    def test_cost_projection_retained_even_with_one_execution_slot(self):
        plans,report=enumerate_egraph_plans(make_contract(Spec(8,16,16,16)),max_applications=4,
            keep=1,max_states=2000,max_executions=1,execution_depth=0)
        self.assertTrue(any(p['execution_steps'] and all(r['direct_output'] for r in p['lowering']) for p in plans))
        for p in plans:validate_egraph_plan(make_contract(Spec(8,16,16,16)),p)


class ScheduleTests(unittest.TestCase):
    def test_scratch_schedule_matches_complete_topological_oracle(self):
        from Optimizer.ir_backend import build_region_dag,emit_region
        from Optimizer.semantics import partition_output
        from Optimizer.scratch_schedule import minimize_scratch,validate_order
        spec=Spec(8,8,8,8)
        dag=build_region_dag(partition_output(build(spec)['materialized_layers'].expression,0,4,spec),spec)
        ops={op['out']['base']:op for op in dag['operators']};orders=[]
        def enumerate_orders(order,remaining):
            if not remaining:orders.append(order);return
            for n in sorted(remaining):
                if all(v['base'] not in ops or v['base'] in order for v in ops[n]['args']):
                    enumerate_orders(order+[n],remaining-{n})
        enumerate_orders([],set(ops))
        costs=[emit_region({**dag,'operators':[ops[n] for n in order]},'test',{})[1]['arena_bytes'] for order in orders]
        bad=orders[costs.index(max(costs))];dag['operators']=[ops[n] for n in bad]
        order,report=minimize_scratch(dag,max_states=100000)
        self.assertTrue(validate_order(dag,order));self.assertEqual(report['arena_bytes'],min(costs))
        self.assertLess(min(costs),max(costs));self.assertEqual(report['optimality'],'exact_for_serial_first_fit_allocator')
        self.assertEqual(emit_region({**dag,'operators':[ops[n] for n in order]},'test',{})[1]['arena_bytes'],report['arena_bytes'])

    @unittest.skipUnless(shutil.which('g++'),'requires host compiler')
    def test_loop_interchange_and_scheduler_reach_exact_cpp(self):
        from Optimizer.execution_ir import build_execution
        from Optimizer.ir_planner import make_ir_plan
        from Optimizer.semantics import partition_output
        spec=Spec(16,16,16,24);c=make_contract(spec);r=default_registry()
        expr,step=r.apply(build(spec)['materialized_layers'].expression,spec,'tensor.partition_rows',Match((),(('chunk',8),)))
        ctx={'spec':spec,'target':__import__('Optimizer.contracts',fromlist=['Target']).Target(),'groups':1}
        state=build_execution(expr,spec,1,'flat');trace=[]
        for rid in ('lower.fuse_quantized_matmul','lower.tile_accumulator_grid','lower.interchange_output_loops',
                    'memory.bind_output_destination'):
            params=next(p for rule,p in r.execution.proposals(state,ctx) if rule==rid)
            state,proof=r.execution.apply(state,ctx,rid,params);trace.append(proof)
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)/'loop';generate(c,make_ir_plan(c,[step],execution_steps=trace),d,frames=4)
            self.assertEqual(cpu_check(d)['status'],'passed')
            source=(d/'compute.hpp').read_text(encoding='utf-8')
            self.assertIn('for(unsigned cb=0;cb<',source)
        order=[op['out']['base'] for op in state['regions'][0]['operators']]
        with self.assertRaisesRegex(ValueError,'DENY'):
            r.execution.apply(state,ctx,'schedule.minimize_scratch',{'region':0,'order':list(reversed(order))})


class LiteratureCapabilityTests(unittest.TestCase):
    def test_ordered_access_proof_not_histogram_and_epochs_fail_closed(self):
        from Optimizer.access import Access
        from Optimizer.mechanisms import run
        limits={'max_dimensions':4,'max_dimension_size':1024,'max_abs_stride':1024,'address_alignment_bytes':4}
        reference=Access('weights','frame7',(2,4),(4,1),0,8,1)
        flat=replace(reference,sizes=(8,),strides=(1,))
        check=lambda a,lim=limits:run('check.access_sequence',reference=reference,candidate=a,limits=lim)
        self.assertEqual(check(flat).status,'ALLOW')
        # Same elements and multiplicities, different order.
        self.assertEqual(check(replace(reference,sizes=(4,2),strides=(1,4))).status,'DENY')
        self.assertEqual(check(replace(flat,epoch='frame6')).status,'DENY')
        self.assertEqual(check(flat,None).status,'UNKNOWN')
        self.assertEqual(run('check.access_sequence',reference=reference,candidate=flat,limits=limits,max_elements=4).status,'UNKNOWN')

    def test_write_alias_and_unimplemented_card_are_rejected(self):
        from Optimizer.access import Access
        from Optimizer.mechanisms import run,catalog
        bad=Access('y','frame1',(4,),(0,),0,4,1,True)
        self.assertEqual(run('check.access_sequence',reference=bad,candidate=bad).status,'DENY')
        self.assertEqual(run('guard.weight_residency',contract=make_contract()).status,'DENY')
        self.assertEqual(len(catalog()),9)
        with self.assertRaisesRegex(ValueError,'unavailable'):run('P5-R03')


if __name__=='__main__':unittest.main()
