"""Conditional ranking of compiled candidates, grouped by checked comparison."""
from collections import defaultdict

from .compiled_analyzer import evaluate_candidates, seal
from .joint_extract import rank_plans


def rank_candidates(candidates, evaluations, *, metric='batch_complete'):
    if metric not in ('first_output','batch_complete'): raise ValueError('unknown latency objective')
    by_id = {c['id']:c for c in candidates}
    if len(by_id) != len(candidates) or len({c['plan_id'] for c in candidates}) != len(candidates):
        raise ValueError('unique candidate and plan identities required')
    if {r['candidate_id'] for r in evaluations} != set(by_id) or len(evaluations) != len(candidates):
        raise ValueError('one evaluation per candidate required')
    groups = defaultdict(list); unknown = []; refused = []
    for result in evaluations:
        candidate = by_id[result['candidate_id']]
        if result['plan_id'] != candidate['plan_id']: raise ValueError('evaluation plan mismatch')
        if result['status'] != 'conditional_estimate':
            (refused if result['status'] in ('invalid_input','refused') else unknown).append(candidate['id'])
            continue
        cores = result.get('actual_compute_tiles')
        if type(cores) is not int or cores < 1: raise ValueError('actual compiled core count required')
        if not result.get('comparison_key') or not result.get('scenario_id'):
            raise ValueError('checked workload and correlated scenario required')
        groups[(result['comparison_key'],result['scenario_id'])].append((candidate,result))
    rankings = []
    for (comparison,scenario), pairs in sorted(groups.items()):
        plans = [{'id':c['plan_id'],'planned_compute_tiles':r['actual_compute_tiles']} for c,r in pairs]
        results = {r['plan_id']:r for c,r in pairs}
        ranking = rank_plans(plans,results,metric=metric)
        plan_to_candidate = {c['plan_id']:c['id'] for c,r in pairs}
        ranking['candidate_by_plan'] = plan_to_candidate
        ranking['selected_candidate_id'] = plan_to_candidate[ranking['ranked'][0]['plan_id']]
        rankings.append(ranking)
    return seal({'schema':'aie-optimizer-compiled-ranking-v1','metric':metric,'groups':rankings,
        'unknown_candidates':unknown,'refused_candidates':refused,
        'reference_candidate_id':candidates[0]['id'] if candidates else None,
        'cross_group_ranking_performed':False,'native_accuracy_admitted':False,
        'ranking_accuracy_admitted':False,
        'scope':'conditional order within checked compiled-program/input/scenario groups; no cross-compile semantic equivalence or native optimum claim'})


def compiled_campaign(candidates, snapshot, output, *, metric='batch_complete', timeout=300, reuse_prepared=True):
    batch = evaluate_candidates(candidates,snapshot,output,timeout=timeout,reuse_prepared=reuse_prepared)
    ranking = rank_candidates(candidates,batch['evaluations'],metric=metric)
    from .compiled_analyzer import save_new
    from pathlib import Path
    save_new(Path(output)/'ranking.json',ranking)
    return {'evaluations':batch,'ranking':ranking}
