"""Reproducible lowering policy used before comparing structural costs.

The policy is an evaluation projection, not a latency proof. Every rewrite is
guarded and replayable. Keep its resulting execution as an explicit candidate;
other execution alternatives remain available to native exploration.
"""
from .execution_ir import build_execution


def refine_for_cost(expr,spec,target,registry,groups=1,gather='flat'):
    state=build_execution(expr,spec,groups,gather)
    context={'spec':spec,'target':target,'groups':groups};steps=[]
    for rule in ('lower.fuse_quantized_matmul','memory.bind_output_destination','memory.forward_row_concat_destination'):
        proposals=[p for rid,p in registry.execution.proposals(state,context) if rid==rule]
        if proposals:
            state,step=registry.execution.apply(state,context,rule,proposals[0]);steps.append(step)
    # Each strictly improving schedule is tied to one distinct region. Recompute
    # proposals after mutation so a stale order cannot be replayed on another DAG.
    for _ in range(groups):
        proposals=[p for rid,p in registry.execution.proposals(state,context) if rid=='schedule.minimize_scratch']
        if not proposals:break
        state,step=registry.execution.apply(state,context,'schedule.minimize_scratch',proposals[0]);steps.append(step)
    return state,steps
