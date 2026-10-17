"""A resumable, bounded native portfolio while fast-model coverage is absent."""
import json
from pathlib import Path
import time
from .planner import enumerate_plans, analyzer_request
from .backend import generate
from .runner import check_artifact,cpu_check,native_run,review_native,compile_diagnosis
from .frontier import verified_frontier
from .contracts import identity


def candidate_directory(root,plan):
    if plan.get('schema') in ('aie-optimizer-ir-plan-v2','aie-optimizer-egraph-plan-v3'):
        return root/('ir_'+plan['id'][:20])
    suffix='_tree' if plan.get('gather')=='tree' else ''
    return root/(plan['mode']+'_'+str(plan['row_groups'])+suffix)


def choose(plans, attempted, records=()):
    used={p['id'] for p in attempted}
    pending=[p for p in plans if p['id'] not in used]
    if not pending:return None
    if pending[0].get('schema') in ('aie-optimizer-ir-plan-v2','aie-optimizer-egraph-plan-v3'):
        family=lambda p:(p.get('expression_id'),tuple(sorted({s['rule'] for s in p['steps']+p.get('execution_steps',[])})))
        families={family(p) for p in attempted};counts={p['planned_compute_tiles'] for p in attempted}
        # Diversity is an evaluation allocation policy, not a latency estimate.
        if not attempted:
            return min(pending,key=lambda p:(len(p['steps'])+len(p.get('execution_steps',[])),p['planned_compute_tiles'],p['id']))
        return min(pending,key=lambda p:(family(p) in families,p['planned_compute_tiles'] in counts,
            p['planned_compute_tiles'],sum(r['scheduled_macs'] for r in p['lowering']),p['id']))
    failed={r['candidate_id'] for r in records if r.get('diagnosis',{}).get('code')=='placement_failure_observed'}
    for original in attempted:
        if original['id'] in failed and original.get('gather','flat')=='flat':
            repair=next((p for p in pending if p['mode']==original['mode']
                         and p['row_groups']==original['row_groups'] and p.get('gather')=='tree'),None)
            if repair:return repair
    if not attempted:
        baseline=next((p for p in pending if p['mode']=='materialized' and p['row_groups']==1),None)
        return baseline or min(pending,key=lambda p:(p['planned_compute_tiles'],p['logical_stats']['total_macs'],p['id']))
    modes={p['mode'] for p in attempted};counts={p['planned_compute_tiles'] for p in attempted}
    return min(pending,key=lambda p:(p['mode'] in modes,p['planned_compute_tiles'] in counts,
                                    p['planned_compute_tiles'],p['logical_stats']['total_macs'],p['id']))


def campaign(contract,root,platform,*,max_evaluations=6,wall_seconds=1200,on_progress=None,
             registered=False,registry=None,max_candidates=24,max_depth=2,max_executions=8,execution_depth=2,
             egraph=False,max_states=10000,joint_evaluation=None,allow_conditional_ranking=False):
    started=time.monotonic()
    if type(max_evaluations) is not int or max_evaluations<1 or wall_seconds<=0:
        raise ValueError('positive campaign budgets required')
    if joint_evaluation is not None and not allow_conditional_ranking:
        raise ValueError('joint phase ranking is conditional; explicit opt-in required until native calibration admission')
    joint_identity=None if joint_evaluation is None else identity(joint_evaluation)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    checkpoint=root/'campaign.json'
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        if (old.get('schema')!='aie-optimizer-campaign-v1' or old['contract_id']!=contract['id']
                or old.get('registered',False)!=registered or old.get('egraph',False)!=egraph
                or old.get('joint_evaluation_id')!=joint_identity):
            raise ValueError('campaign belongs to another contract')
    joint_report=None
    if joint_evaluation is not None:
        from .joint_extract import enumerate_joint_plans
        from .phase_costs import PhaseEvaluator
        if set(joint_evaluation)!={'registry','selection','bindings'}:raise ValueError('explicit joint evaluator configuration required')
        evaluator=PhaseEvaluator(contract,joint_evaluation['registry'],joint_evaluation['selection'],bindings=joint_evaluation['bindings'])
        plans,joint_report=enumerate_joint_plans(contract,evaluator=evaluator,registry=registry,max_rounds=max(1,max_depth),
            max_plans=max_candidates,max_states=max_states,max_executions=max_executions,execution_depth=execution_depth)
        refusals=joint_report
    elif egraph:
        from .egraph_planner import enumerate_egraph_plans
        plans,refusals=enumerate_egraph_plans(contract,registry=registry,max_rounds=max(1,max_depth),
            keep=max_candidates,max_states=max_states,max_executions=max_executions,execution_depth=execution_depth)
    elif registered:
        from .ir_planner import enumerate_ir_plans
        plans,refusals=enumerate_ir_plans(contract,max_candidates,max_depth,registry,max_executions,execution_depth)
    else:plans,refusals=enumerate_plans(contract)
    attempted=[];records=[]
    if not plans:raise ValueError('no supported deployment fits the contract: '+str(refusals))
    state={'schema':'aie-optimizer-campaign-v1','contract_id':contract['id'],
        'policy':'bounded structural/core-count diversity; native fallback, no claimed Analyzer ranking',
        'plans':len(plans),'generation_refusals':refusals,'records':records,'decisions':[],
        'registered':registered,'egraph':egraph,
        'joint_evaluation_id':joint_identity,'joint_ranking_conditional':joint_report is not None,
        'max_executions_per_binding':max_executions,'execution_depth':execution_depth,
        'max_evaluations':max_evaluations,'wall_seconds_budget_this_invocation':wall_seconds,
        'analyzer_status':analyzer_request(contract,plans[0],registry),'historical_evaluations_included':0}
    if joint_report is not None:state['policy']='conditional joint phase ranking; reference retained; one unknown exploration in every four evaluations'
    def save():
        state['elapsed_seconds_this_invocation']=time.monotonic()-started
        state['frontier']=verified_frontier(records)
        tmp=root/'campaign.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(checkpoint)
    for plan in plans:
        directory=candidate_directory(root,plan)
        native=directory/'native_result.json';evaluation=directory/'evaluation_result.json'
        if not native.exists() and not evaluation.exists():continue
        _,manifest=check_artifact(directory,registry)
        if manifest['plan']!=plan or manifest['contract']!=contract:
            raise ValueError('existing candidate belongs to another plan/contract')
        result=json.loads((native if native.exists() else evaluation).read_text())
        if result.get('candidate_id')!=plan['id'] or result.get('source_digest')!=manifest['source_digest']:
            raise ValueError('prior evaluation source identity mismatch')
        if result.get('status')=='verified':result=review_native(directory,registry)
        elif result.get('status')=='compile_failed' and (directory/'native_compile.log').exists():
            result['diagnosis']=compile_diagnosis((directory/'native_compile.log').read_text(errors='replace'))
        records.append(result);attempted.append(plan)
    state['historical_evaluations_included']=len(records);save()
    while len(attempted)<max_evaluations and time.monotonic()-started<wall_seconds:
        if joint_report is not None:
            from .joint_extract import choose_joint
            plan=choose_joint(plans,attempted,joint_report)
        else:plan=choose(plans,attempted,records)
        if plan is None:break
        directory=candidate_directory(root,plan)
        decision={'candidate_id':plan['id'],'mode':plan.get('mode','operator_lowering'),'planned_compute_tiles':plan['planned_compute_tiles'],
                  'reason':'baseline' if not attempted else 'cover distinct computation structures and occupied-core counts'}
        if joint_report is not None:
            evaluation=joint_report['evaluations'][plan['id']]
            decision.update(reason='reference or conditional phase priority with reserved unknown exploration',
                phase_status=evaluation['status'],conditional_latency_ns=evaluation.get('conditional_latency_ns'),native_prediction=None)
        if plan.get('gather')=='tree' and any(r.get('diagnosis',{}).get('code')=='placement_failure_observed'
                and r.get('mode')==plan.get('mode') and r.get('row_groups')==plan['row_groups'] for r in records):
            decision['reason']='replace centralized output gather after observed placement failure'
        state['decisions'].append(decision);save()
        if on_progress:on_progress({'event':'selected',**decision})
        if not directory.exists():generate(contract,plan,directory,registry=registry)
        _,m=check_artifact(directory,registry)
        if m['plan']!=plan or m['contract']!=contract:raise ValueError('candidate identity mismatch')
        host=cpu_check(directory,registry=registry)
        if host['status']!='passed':
            record={'candidate_id':plan['id'],'status':'host_check_failed','evidence_kind':'cpu_only',
                    'source_digest':m['source_digest']}
        else:
            remaining=wall_seconds-(time.monotonic()-started)
            if remaining<=0:break
            record=native_run(directory,platform,total_timeout=remaining,registry=registry)
            if record['status']=='verified':record=review_native(directory,registry)
        (directory/'evaluation_result.json').write_text(json.dumps(record,indent=2)+'\n')
        records.append(record);attempted.append(plan);save()
        if on_progress:on_progress({'event':'evaluated','candidate_id':plan['id'],
                                     'status':record['status'],'actual_compute_tiles':record.get('actual_compute_tiles'),
                                     'latency_ns':record.get('latency_ns')})
    state['stop_reason']='evaluation_budget' if len(attempted)>=max_evaluations else (
        'space_exhausted' if len(attempted)==len(plans) else 'wall_time_budget')
    save();return state
