"""Executable coverage for the nine literature-derived specification cards.

Partial capabilities are explicit. No unimplemented mechanism is enabled by
changing a JSON status field or dynamically importing a paper-provided name.
"""
from .registry import Decision


def catalog():
    return [
      {'id':'P5-R01','name':'runtime.config_epoch','status':'specified_not_implemented','capabilities':[],
       'missing':['epoch IR','host/PL configuration distribution','runtime synchronization']},
      {'id':'P5-R02','name':'extract.resource_coupled_modes','status':'partially_implemented',
       'capabilities':['extract.shared_dag','schedule.minimize_scratch'],
       'entrypoints':['Optimizer.global_extract.extract_dags','Optimizer.egraph_planner.enumerate_egraph_plans'],
       'missing':['joint native latency model','arbitrary physical cloning','atomic multi-output fused tasks','joint routing/core schedule']},
      {'id':'P5-R03','name':'placement.route_resource_graph','status':'specified_not_implemented','capabilities':[],
       'missing':['route resource graph','DMA/lock capacities','placement/routing optimizer']},
      {'id':'P5-R04','name':'pipeline.group_iterations','status':'specified_not_implemented','capabilities':[],
       'missing':['iteration/state IR','state forwarding','program capacity profiles']},
      {'id':'P5-R05','name':'memory.weight_residency_epoch','status':'guard_only',
       'capabilities':['guard.weight_residency'],'missing':['preload/refresh code generation','multi-copy epoch protocol']},
      {'id':'P5-R06','name':'lower.register_block_schedule','status':'partially_implemented',
       'capabilities':['lower.tile_accumulator_grid','lower.interchange_output_loops'],
       'missing':['prefetch/double-buffered microkernels','register spill profiles','shape-calibrated cost model']},
      {'id':'P5-R07','name':'semantic.semiring_short_circuit','status':'specified_not_implemented','capabilities':[],
       'missing':['Boolean/sparse IR','semiring proof','backend']},
      {'id':'P5-R08','name':'stream.merge_by_rate','status':'specified_not_implemented','capabilities':[],
       'missing':['multi-rate tokens','PL merger/DWC generation','burst/backpressure proof']},
      {'id':'P5-R09','name':'layout.access_sequence_equivalence','status':'checker_only',
       'capabilities':['check.access_sequence'],'missing':['DMA descriptor generator','layout conversion insertion','multi-channel protocol proof']}]


def run(capability,**kwargs):
    """Closed dispatch; executable handlers, never passive implementation labels."""
    if capability=='extract.shared_dag':
        from .global_extract import extract_dags
        return extract_dags(**kwargs)
    if capability=='schedule.minimize_scratch':
        from .scratch_schedule import minimize_scratch
        return minimize_scratch(**kwargs)
    if capability=='check.access_sequence':
        from .access import check_access
        return check_access(**kwargs)
    if capability=='guard.weight_residency':
        contract=kwargs['contract']
        from .contracts import validate_contract
        validate_contract(contract)
        # contract-v1 explicitly requires distinct weights every frame. Supporting
        # another epoch contract requires its validator and generated refresh path.
        return Decision('DENY','contract-v1 distinct-per-frame weights require refresh; resident-weight backend absent')
    if capability in ('lower.tile_accumulator_grid','lower.interchange_output_loops'):
        from .execution_rules import default_execution_registry
        return default_execution_registry().apply(kwargs['state'],kwargs['context'],capability,kwargs['parameters'])
    raise ValueError('capability is unavailable or not registered: '+str(capability))
