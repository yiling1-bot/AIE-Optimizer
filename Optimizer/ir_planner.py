"""Replay certified registry derivations and bind expressions to row-owned tiles."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from .builtin_rules import default_registry
from .contracts import identity, validate_contract
from .ir_backend import lower_program
from .planner import gather_regions
from .semantics import expression_id
from .execution_ir import build_execution,validate_execution


def implementation_fingerprint():
    """Invalidate search caches when generation/legality implementations change."""
    names=('ir_planner.py','ir_backend.py','operator_runtime.hpp.in','compute.hpp.in',
           'backend.py','registry.py','builtin_rules.py','semantics.py','contracts.py','fixtures.py',
           'execution_ir.py','execution_rules.py','egraph.py','egraph_rules.py',
           'global_extract.py','egraph_planner.py','scratch_schedule.py','access.py','mechanisms.py',
           'cost_refinement.py','stage_costs.py','compiled_profile.py')
    return identity({n:sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def make_ir_plan(contract, steps=(), groups=1, gather='flat', registry=None, execution_steps=()):
    spec,_=validate_contract(contract)
    registry=registry or default_registry()
    expr=registry.replay(spec,steps)
    return bind_expression(contract,expr,groups,gather,registry,execution_steps,steps=steps)


def bind_expression(contract,expr,groups,gather,registry,execution_steps=(),*,steps=(),certificate=None):
    """Internal lowering after either trace replay or e-class certificate checking."""
    spec,target=validate_contract(contract)
    if type(groups) is not int or groups<1 or groups>min(target.columns,target.max_compute_tiles,8):
        raise ValueError('row group count exceeds target/cap/backend support')
    if spec.m%(4*groups) or any(v%8 for v in (spec.k,spec.hidden,spec.n)):
        raise ValueError('complete equal MMUL microtiles required')
    gather_regions(groups,gather)
    context={'spec':spec,'target':target,'groups':groups}
    execution=registry.execution.replay(build_execution(expr,spec,groups,gather),context,execution_steps)
    validate_execution(execution,context)
    _,lowering=lower_program(expr,spec,groups,execution)
    heaps=[];bounds=[];slots=[]
    for kernel in execution['kernels']:
        regions=[a['region'] for a in kernel['actions'] if a['op']=='compute']
        arena=sum(lowering[g]['arena_bytes'] for g in regions)
        heap=((arena+1024+1023)//1024)*1024;heaps.append(heap)
        bounds.append(heap+2*(sum(p['size'] for p in kernel['inputs'])+kernel['output_size'])+8192+256)
        slots.append({'id':kernel['id'],'tile':kernel['tile'],'group':regions[0] if regions else None,
                      'stages':['L0','L1'] if regions else ['gather']})
    upper=max(bounds)
    if upper>target.tile_memory_bytes:
        raise ValueError('conservative per-tile storage cap exceeded')
    plan={'schema':'aie-optimizer-ir-plan-v2','contract_id':contract['id'],
          'implementation_fingerprint':implementation_fingerprint(),
          'expression':asdict(expr),'expression_id':expression_id(expr),
          'steps':list(steps),'row_groups':groups,'group_rows':spec.m//groups,'gather':gather,
          'execution':execution,'execution_steps':list(execution_steps),'slots':slots,
          'planned_compute_tiles':len(slots),'compiled_compute_tiles':None,'native_legality':'unverified',
          'lowering':lowering,'resources':{'declared_per_tile_memory_upper_bytes':upper,
                                          'kernel_heap_bytes':heaps},
          'scope':'registered tensor and execution rewrites; packed operator DAG; explicit single-rate ownership phases; full-frame ADF interface'}
    if certificate is not None:
        plan['schema']='aie-optimizer-egraph-plan-v3'
        plan['equality_certificate']=certificate
        plan['scope']='certified e-class extraction; bounded shared DAG; existing ADF row deployment and execution registry'
    # JSON-normalize tuples so reloaded manifests compare exactly.
    import json
    plan=json.loads(json.dumps(plan));plan['id']=identity(plan)
    return plan


def validate_ir_plan(contract,plan,registry=None):
    expected=make_ir_plan(contract,plan['steps'],plan['row_groups'],plan['gather'],registry,plan.get('execution_steps',()))
    if plan!=expected:
        raise ValueError('IR plan is not a reproducible registered derivation')


def enumerate_ir_plans(contract,max_candidates=24,max_depth=2,registry=None,max_executions=8,execution_depth=2):
    registry=registry or default_registry();spec,target=validate_contract(contract)
    expressions,report=registry.explore(spec,max_candidates,max_depth)
    plans=[];refused=[];execution_reports=[]
    for entry in expressions:
        for groups in range(1,min(8,target.max_compute_tiles,spec.m//4)+1):
            if spec.m%(4*groups):continue
            for gather in (('flat','tree') if groups>2 else ('flat',)):
                try:
                    seed=build_execution(entry['expression'],spec,groups,gather)
                    context={'spec':spec,'target':target,'groups':groups}
                    variants,execution_report=registry.execution.explore(seed,context,max_executions,execution_depth)
                    execution_reports.append({'expression_id':entry['id'],'groups':groups,'gather':gather,
                                              'variants':len(variants),'report':execution_report})
                    for variant in variants:
                        try:plans.append(make_ir_plan(contract,entry['steps'],groups,gather,registry,variant['steps']))
                        except ValueError as e:
                            refused.append({'expression_id':entry['id'],'groups':groups,'gather':gather,
                                            'execution_rules':[s['rule'] for s in variant['steps']], 'reason':str(e)})
                except ValueError as e:refused.append({'expression_id':entry['id'],'groups':groups,'reason':str(e)})
    return plans,{'expressions':len(expressions),'exploration':report,'lowering_refused':refused,
                  'execution_explorations':execution_reports,
                  'selection_scope':'bounded structural enumeration; no latency predictor or optimality claim'}
