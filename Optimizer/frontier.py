"""Do not mix predicted, differently timed, or numerically unverified points."""
import math


def verified_frontier(records):
    groups={}; refused=[]
    for record in records:
        valid = (record.get('status') == 'verified' and record.get('numerical') == 'EXACT'
                 and record.get('mapping') == 'EXACT'
                 and record.get('evidence_kind') == 'native_simulation'
                 and type(record.get('actual_compute_tiles')) is int
                 and record['actual_compute_tiles'] > 0
                 and type(record.get('latency_ns')) in (int,float)
                 and math.isfinite(record['latency_ns']) and record['latency_ns'] > 0
                 and all(record.get(k) for k in ('comparison_key','candidate_id','source_digest')))
        if not valid:
            refused.append({'candidate_id':record.get('candidate_id'), 'reason':'incomplete native evidence'})
            continue
        groups.setdefault(record['comparison_key'],[]).append(record)
    result={}
    for key,points in groups.items():
        result[key]=[a for a in points if not any(
            b['latency_ns'] <= a['latency_ns'] and b['actual_compute_tiles'] <= a['actual_compute_tiles']
            and (b['latency_ns'] < a['latency_ns'] or b['actual_compute_tiles'] < a['actual_compute_tiles'])
            for b in points)]
        result[key].sort(key=lambda p:(p['actual_compute_tiles'],p['latency_ns'],p['candidate_id']))
    return {'schema':'aie-optimizer-verified-frontier-v1','groups':result,'refused':refused,
            'scope':'non-dominated among evaluated candidates; not a global optimum'}
