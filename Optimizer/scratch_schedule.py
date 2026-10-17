"""Bounded topological scheduling against the generator's real scratch allocator.

Pure local operators only; ADF acquire/release and inter-tile schedules stay in
execution_ir. Every returned order is a complete permutation checked on replay.
"""
from collections import Counter


def allocation_step(active, size):
    offset=0
    for start,length in sorted(active.values()):
        if offset+size<=start:break
        offset=max(offset,start+length)
    return offset


def minimize_scratch(dag, max_states=4096):
    if type(max_states) is not int or max_states<1:raise ValueError('positive scheduling budget required')
    ops={op['out']['base']:op for op in dag['operators']}
    if len(ops)!=len(dag['operators']):raise ValueError('unique SSA operators required')
    deps={n:tuple(v['base'] for v in op['args'] if v['base'] in ops) for n,op in ops.items()}
    uses=Counter(v for values in deps.values() for v in values);uses[dag['output']['base']]+=1
    sizes={n:0 if op.get('output_direct') or 'destination_offset' in op else
           ((op['out']['rows']*op['out']['cols']*(4 if op['out']['dtype']=='int32' else 1)+31)//32*32)
           for n,op in ops.items()}
    def advance(active,left,n):
        active={k:v for k,v in active.items() if left[k]>0}
        offset=allocation_step(active,sizes[n]) if sizes[n] else 0
        if sizes[n]:active[n]=(offset,sizes[n])
        left=left.copy();left.subtract(deps[n])
        return active,left,offset+sizes[n]
    original=list(ops);active={};left=uses.copy();peak=0;done=set()
    for n in original:
        if not set(deps[n])<=done:raise ValueError('input operator DAG must be acyclic and ordered')
        active,left,end=advance(active,left,n);peak=max(peak,end);done.add(n)
    best_peak=peak;best=original;states=0;prunes=0
    stack=[([],set(),{},uses.copy(),0)]
    while stack and states<max_states:
        order,done,active,left,peak=stack.pop();states+=1
        if peak>=best_peak:prunes+=1;continue
        if len(order)==len(ops):best=order;best_peak=peak;continue
        ready=[n for n in ops if n not in done and set(deps[n])<=done]
        for n in reversed(ready):
            nxt,remaining,end=advance(active,left,n)
            stack.append((order+[n],done|{n},nxt,remaining,max(peak,end)))
    return best,{'arena_bytes':best_peak,'states':states,'bound_prunes':prunes,
        'optimality':'exact_for_serial_first_fit_allocator' if not stack else 'best_found_budget_limited'}


def validate_order(dag,order):
    ops={op['out']['base']:op for op in dag['operators']}
    if not isinstance(order,list) or len(order)!=len(ops) or set(order)!=set(ops):
        return False
    done=set()
    for n in order:
        if any(v['base'] in ops and v['base'] not in done for v in ops[n]['args']):return False
        done.add(n)
    return True
