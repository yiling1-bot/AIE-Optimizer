"""Resolve phase service through Analyzer's existing pinned component registry."""
from math import isfinite
from .contracts import identity
from .phase_model import lower_phases,evaluate_graph
from .analyzer_bridge import resolve_program_costs,load_cost_family


def evaluate_phases(contract,plan,binding,registry,selection,*,maximum_events=100000):
    if set(selection)!={'cost_by_component','member_by_group'} or any(type(v) is not dict for v in selection.values()):
        raise ValueError('explicit component cost and correlated member selections required')
    lowered=lower_phases(contract,plan,binding,maximum_events=maximum_events)
    resolution=resolve_program_costs(lowered['components'],registry)
    result={'schema':'aie-optimizer-phase-evaluation-v1','plan_id':plan['id'],'binding_id':binding['id'],
        'comparison_key':lowered['comparison_key'],
        'scenario_id':identity({'registry':registry,'members':selection['member_by_group']}),
        'resolution':resolution,'conditional_latency_ns':None,'latency_ns':None,
        'native_admitted':False,'physical_cores_planned':lowered['physical_cores_planned'],
        'status':resolution['status'],'scope':lowered['scope'],'lowered_graph_id':identity(lowered)}
    if resolution['status']!='reused':return result
    assignment={};chosen=[];unselected=[]
    for row in resolution['components']:
        cid=row['component_id'];options=row['compatible_costs'];cost_id=selection['cost_by_component'].get(cid)
        if cost_id is None and len(options)==1:cost_id=options[0]['cost_id']
        if cost_id is None:unselected.append({'component':cid,'reason':'multiple cost families require explicit selection'});continue
        family=load_cost_family(row,cost_id)
        if family['referenced_documents']:
            raise ValueError('referenced legacy family requires a mechanism-specific adapter; no flattening')
        models=family['inline_models'];group=family['correlation_group']
        member=selection['member_by_group'].get(group)
        if member is None:unselected.append({'component':cid,'reason':'correlated family member must be selected even for one member'});continue
        matching=[m for m in models if m['model_id']==member]
        if len(matching)!=1:raise ValueError('correlation-group member is absent or ambiguous')
        params=matching[0]['parameters']
        if set(params)!={'duration_ps'} or type(params['duration_ps']) is not int or params['duration_ps']<0:
            raise ValueError('phase service requires one nonnegative integer duration_ps, no endpoint timestamps')
        assignment[cid]=params['duration_ps'];chosen.append({'component':cid,'cost_id':cost_id,'group':group,
            'member':member,'origin':family['origin'],'model_discrepancy':family['model_discrepancy']})
    if unselected:result.update(status='needs_model_selection',unselected=unselected);return result
    timing=evaluate_graph(lowered,assignment)
    result.update(status='conditional_estimate',conditional_latency_ns=timing['first_output_ps']/1000,
        conditional_frame_completion_ns=[t/1000 for t in timing['frame_completion_ps']],
        conditional_finite_frame_intervals_ns=[t/1000 for t in timing['finite_frame_intervals_ps']],
        selected_costs=chosen,timing=timing,
        limitations=['declared slot/resource order, not a native DMA arbitration claim',
                     'component cost coverage is not hardware accuracy admission',
                     'no hardware error bound or native optimality guarantee'])
    return result


class PhaseEvaluator:
    """Joint extraction callback; bindings may be compiled-adapter requests later."""
    def __init__(self,contract,registry,selection,bindings=None,binding_factory=None,maximum_events=100000):
        self.contract=contract;self.registry=registry;self.selection=selection
        self.bindings=bindings or {};self.binding_factory=binding_factory;self.maximum_events=maximum_events

    def __call__(self,plan):
        binding=self.bindings.get(plan['id'])
        if binding is None and self.binding_factory is not None:binding=self.binding_factory(plan)
        if binding is None:return {'status':'needs_phase_binding','conditional_latency_ns':None,'latency_ns':None,'native_admitted':False}
        return evaluate_phases(self.contract,plan,binding,self.registry,self.selection,maximum_events=self.maximum_events)


def finite_score(result,metric='first_output'):
    if metric not in ('first_output','batch_complete'):raise ValueError('unknown latency objective')
    if result.get('status')!='conditional_estimate':return None
    if result.get('native_admitted') is not False or result.get('latency_ns') is not None:
        raise ValueError('conditional phase results must not claim native timing')
    score=result.get('conditional_latency_ns') if metric=='first_output' else result['conditional_frame_completion_ns'][-1]
    if type(score) not in (int,float) or not isfinite(score) or score<0:raise ValueError('finite nonnegative phase score required')
    if not result.get('comparison_key') or not result.get('scenario_id'):raise ValueError('workload and correlated scenario identities required')
    return score
