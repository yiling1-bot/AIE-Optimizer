"""Registered rewrites of operator DAGs and buffer ownership phases."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from .contracts import identity
from .execution_ir import validate_execution
from .registry import Decision,Rule


class ExecutionRegistry:
    def __init__(self):self._rules={}

    def register(self,rule):
        if (rule.id in self._rules or not rule.id or type(rule.version) is not int or rule.version<1
            or not rule.identity_argument or not all(callable(getattr(rule,k)) for k in ('match','check','rewrite'))):
            raise ValueError('unique executable execution rule with identity argument required')
        self._rules[rule.id]=rule;return self

    def catalog(self):return [r.metadata() for r in self._rules.values()]

    def proposals(self,state,context):
        for rule in self._rules.values():
            for params in rule.match(state,context):yield rule.id,params

    def apply(self,state,context,rule_id,parameters):
        before=validate_execution(state,context);rule=self._rules[rule_id]
        decision=rule.check(state,context,parameters)
        if decision.status!='ALLOW':raise ValueError(decision.status+': '+decision.reason)
        result=rule.rewrite(deepcopy(state),context,deepcopy(parameters))
        if validate_execution(result,context)!=before:raise ValueError('execution rewrite changed values or output ownership')
        step={'rule':rule.id,'version':rule.version,'parameters':deepcopy(parameters),
              'before':identity(state),'after':identity(result),'condition':asdict(decision)}
        return result,step

    def replay(self,seed,context,steps):
        state=deepcopy(seed);validate_execution(state,context)
        for step in steps:
            if step['rule'] not in self._rules or self._rules[step['rule']].version!=step['version']:
                raise ValueError('unknown execution rule or version')
            state,actual=self.apply(state,context,step['rule'],step['parameters'])
            if actual!=step:raise ValueError('execution certificate differs on replay')
        return state

    def explore(self,seed,context,max_candidates=8,max_depth=2):
        if type(max_candidates) is not int or max_candidates<1 or type(max_depth) is not int or max_depth<0:
            raise ValueError('invalid execution exploration budget')
        entries=[{'state':deepcopy(seed),'steps':[],'depth':0}];seen={identity(seed)};refused=[];edges=[];truncated=False
        for entry in entries:
            if entry['depth']>=max_depth:continue
            for rule,params in self.proposals(entry['state'],context):
                try:state,step=self.apply(entry['state'],context,rule,params)
                except ValueError as e:
                    refused.append({'rule':rule,'parameters':params,'reason':str(e)});continue
                edges.append(step)
                if step['after'] in seen:continue
                if len(entries)>=max_candidates:truncated=True;continue
                seen.add(step['after']);entries.append({'state':state,'steps':entry['steps']+[step],'depth':entry['depth']+1})
        return entries,{'refused':refused,'edges':edges,'budget_reached':truncated}


def fusion_candidates(dag):
    producers={op['out']['base']:op for op in dag['operators']}
    uses=Counter(v['base'] for op in dag['operators'] for v in op['args']);uses[dag['output']['base']]+=1
    def collect(view):
        op=producers.get(view['base'])
        if op is None or view!=op['out'] or uses[view['base']]!=1:return None
        if op['kind']=='matmul':
            a,b=op['args']
            if (a['layout'],b['layout'])!=(4,8) or a['row']%4 or a['col']%8 or b['row']%8 or b['col']%8:return None
            return [op['out']['base']],op['args']
        if op['kind']=='sum':
            parts=[collect(v) for v in op['args']]
            if any(p is None for p in parts):return None
            return [op['out']['base']]+sum((p[0] for p in parts),[]),sum((p[1] for p in parts),[])
        return None
    matches=[]
    for op in dag['operators']:
        if op['kind']!='quantize':continue
        part=collect(op['args'][0])
        if part and sum(v['cols'] for v in part[1][::2])*16384<=2147483647:
            matches.append((op['out']['base'],part[0],part[1]))
    return matches


def _fusion_match(state,context):
    regions=[g for g,d in enumerate(state['regions']) if fusion_candidates(d)]
    return ({'regions':regions},) if regions else ()


def _fusion_check(state,context,p):
    ids=p.get('regions',[])
    if (set(p)!={'regions'} or not ids or any(type(g) is not int for g in ids) or len(ids)!=len(set(ids))
        or any(g<0 or g>=len(state['regions']) or not fusion_candidates(state['regions'][g]) for g in ids)):
        return Decision('DENY','requires private complete MMUL/sum producers below an unchanged quantization boundary')
    return Decision('ALLOW','private aligned producers; complete bounded INT32 reduction; quantize once at original boundary')


def _fusion_rewrite(state,context,p):
    for g in p['regions']:
        dag=state['regions'][g];matches=fusion_candidates(dag)
        remove={n for _,names,_ in matches for n in names};updates={n:args for n,_,args in matches}
        dag['operators']=[{**op,'kind':'fused_qmm','args':updates[op['out']['base']]}
                          if op['out']['base'] in updates else op
                          for op in dag['operators'] if op['out']['base'] not in remove]
    return state


def _deferred_ports(state):
    pairs=[]
    for kernel in state['kernels']:
        compute=[i for i,a in enumerate(kernel['actions']) if a['op']=='compute']
        if not compute:continue
        for p in kernel['inputs']:
            uses=[i for i,a in enumerate(kernel['actions']) if a['op']=='copy' and a['from']==p['name']]
            if p['source'].startswith('kernel:') and not p['async'] and uses and min(uses)>max(compute):
                pairs.append([kernel['id'],p['name']])
    return pairs


def _defer_match(state,context):
    pairs=_deferred_ports(state)
    return ({'ports':pairs},) if pairs else ()


def _defer_check(state,context,p):
    ports=p.get('ports',[]);allowed=_deferred_ports(state)
    if (set(p)!={'ports'} or not ports or any(pair not in allowed for pair in ports)
        or len({tuple(v) for v in ports})!=len(ports)):
        return Decision('DENY','requires a gather-only input whose first use follows independent local compute')
    return Decision('ALLOW','single-rate feed-forward gather; one acquire/release per frame; acquire moves before first read')


def _defer_rewrite(state,context,p):
    for kid,name in p['ports']:
        kernel=next(k for k in state['kernels'] if k['id']==kid)
        next(v for v in kernel['inputs'] if v['name']==name)['async']=True
        actions=[a for a in kernel['actions'] if not (a['op'] in ('acquire','release') and a['port']==name)]
        uses=[i for i,a in enumerate(actions) if a['op']=='copy' and a['from']==name]
        actions.insert(max(uses)+1,{'op':'release','port':name})
        actions.insert(min(uses),{'op':'acquire','port':name});kernel['actions']=actions
    return state


def _extractable(state,context):
    occupied={tuple(k['tile']) for k in state['kernels']};result=[]
    for kernel in state['kernels']:
        computes=[a for a in kernel['actions'] if a['op']=='compute' and a['to']=='own']
        if len(computes)!=1 or not any(p['source'].startswith('kernel:') for p in kernel['inputs']):continue
        g=computes[0]['region'];tile=[kernel['tile'][0],3]
        if tuple(tile) not in occupied and 'compute_'+str(g) not in {k['id'] for k in state['kernels']}:
            result.append({'kernel':kernel['id'],'tile':tile})
    return result


def _extract_match(state,context):return tuple(_extractable(state,context))


def _extract_check(state,context,p):
    if p not in _extractable(state,context):return Decision('DENY','requires independent local output and an unused tile')
    if len(state['kernels'])+1>context['target'].max_compute_tiles:return Decision('DENY','compute tile cap exceeded')
    return Decision('ALLOW','extract local compute onto a new tile; preserve row ownership and one full frame on each edge')


def _extract_rewrite(state,context,p):
    kernel=next(k for k in state['kernels'] if k['id']==p['kernel'])
    compute=next(a for a in kernel['actions'] if a['op']=='compute');g=compute['region']
    size=context['spec'].m//context['groups']*context['spec'].n
    inputs=[v for v in kernel['inputs'] if v['source'].startswith('input:')]
    name='compute_'+str(g)
    actions=[{'op':'acquire','port':v['name']} for v in inputs]+[{'op':'acquire','port':'y'},
             {'op':'compute','region':g,'to':'y'}]+[{'op':'release','port':v['name']} for v in inputs]+[{'op':'release','port':'y'}]
    new={'id':name,'tile':p['tile'],'inputs':inputs,'output_size':size,'actions':actions}
    kernel['inputs']=[v for v in kernel['inputs'] if v not in inputs]+[
        {'name':'local','source':'kernel:'+name,'size':size,'async':False}]
    names={v['name'] for v in inputs}
    kernel['actions']=[a for a in kernel['actions'] if a['op']!='compute'
                       and not (a['op'] in ('acquire','release') and a['port'] in names)]
    kernel['actions'].insert(0,{'op':'acquire','port':'local'})
    for a in kernel['actions']:
        if a['op']=='copy' and a['from']=='own':a['from']='local'
    kernel['actions'].append({'op':'release','port':'local'})
    state['kernels'].insert(state['kernels'].index(kernel),new)
    return state


def _grid_ops(dag,grid):
    return [op for op in dag['operators'] if op['kind']=='fused_qmm' and 'accumulator_grid' not in op
            and op['out']['rows']%(4*grid[0])==0 and op['out']['cols']%(8*grid[1])==0]


def _grid_match(state,context):
    for grid in ([1,2],[2,1],[2,2]):
        regions=[g for g,dag in enumerate(state['regions']) if _grid_ops(dag,grid)]
        if regions:yield {'regions':regions,'grid':grid}


def _grid_check(state,context,p):
    grid=p.get('grid');regions=p.get('regions',[])
    if (set(p)!={'grid','regions'} or not isinstance(grid,list) or grid not in ([1,2],[2,1],[2,2])
        or any(type(v) is not int for v in grid) or not regions or any(type(g) is not int for g in regions)
        or len(set(regions))!=len(regions) or any(g<0 or g>=len(state['regions']) or not _grid_ops(state['regions'][g],grid) for g in regions)):
        return Decision('DENY','requires aligned complete output tiles and at most four independent accumulator microtiles')
    return Decision('ALLOW','each output retains its complete ordered reduction and original Q; share input loads across independent output accumulators')


def _grid_rewrite(state,context,p):
    for g in p['regions']:
        for op in _grid_ops(state['regions'][g],p['grid']):op['accumulator_grid']=p['grid'][:]
    return state


def _direct_output(dag):
    return next((op for op in dag['operators'] if op['out']==dag['output']
                 and op['out']['dtype']=='int8' and not op.get('output_direct')
                 and not any(v['base']==op['out']['base'] for other in dag['operators'] for v in other['args'])),None)


def _output_match(state,context):
    regions=[g for g,dag in enumerate(state['regions']) if _direct_output(dag) is not None]
    return ({'regions':regions},) if regions else ()


def _output_check(state,context,p):
    regions=p.get('regions',[])
    if (set(p)!={'regions'} or not regions or any(type(g) is not int for g in regions)
        or len(set(regions))!=len(regions) or any(g<0 or g>=len(state['regions']) or _direct_output(state['regions'][g]) is None for g in regions)):
        return Decision('DENY','requires a complete private output with no internal readers')
    return Decision('ALLOW','caller owns an exclusive output of the same packed layout; final producer writes it directly; intermediate Q boundaries unchanged')


def _output_rewrite(state,context,p):
    for g in p['regions']:_direct_output(state['regions'][g])['output_direct']=True
    return state


def _loop_match(state,context):
    regions=[g for g,d in enumerate(state['regions']) if any(op['kind']=='fused_qmm' and 'loop_order' not in op for op in d['operators'])]
    return ({'regions':regions,'order':'column_row'},) if regions else ()


def _loop_check(state,context,p):
    regions=p.get('regions',[])
    if (set(p)!={'regions','order'} or p['order']!='column_row' or not isinstance(regions,list) or not regions
        or any(type(g) is not int or not 0<=g<len(state['regions']) for g in regions)
        or len(set(regions))!=len(regions) or any(not any(op['kind']=='fused_qmm' and 'loop_order' not in op
             for op in state['regions'][g]['operators']) for g in regions)):
        return Decision('DENY','requires independent complete output tiles and an unassigned loop order')
    return Decision('ALLOW','interchange independent row/column microtiles; retain every K reduction and Q order')


def _loop_rewrite(state,context,p):
    for g in p['regions']:
        for op in state['regions'][g]['operators']:
            if op['kind']=='fused_qmm' and 'loop_order' not in op:op['loop_order']=p['order']
    return state


def _schedule_match(state,context):
    from .scratch_schedule import minimize_scratch
    for g,dag in enumerate(state['regions']):
        if len(dag['operators'])>24:continue
        order,_=minimize_scratch(dag,max_states=1024)
        if order!=[op['out']['base'] for op in dag['operators']]:yield {'region':g,'order':order}


def _schedule_check(state,context,p):
    from .scratch_schedule import validate_order
    g=p.get('region')
    if (set(p)!={'region','order'} or type(g) is not int or not 0<=g<len(state['regions'])
        or not validate_order(state['regions'][g],p['order'])):
        return Decision('DENY','requires a complete dependency-preserving operator permutation')
    return Decision('ALLOW','pure independent operators may reorder; allocator recomputes all live ranges')


def _schedule_rewrite(state,context,p):
    dag=state['regions'][p['region']];ops={op['out']['base']:op for op in dag['operators']}
    dag['operators']=[ops[n] for n in p['order']];return state


def _row_destination(dag):
    output=next((op for op in dag['operators'] if op['out']==dag['output']),None)
    if (output is None or output['kind']!='concat' or output['axis']!=0 or output.get('concat_forwarded')
        or output['out']['dtype']!='int8'):return None
    producers={op['out']['base']:op for op in dag['operators']}
    uses=Counter(v['base'] for op in dag['operators'] for v in op['args'])
    if any(v['base'] not in producers or producers[v['base']]['kind']!='fused_qmm'
        or producers[v['base']]['out']!=v or uses[v['base']]!=1
        or v['cols']!=output['out']['cols'] or v['rows']%4
        or 'destination_offset' in producers[v['base']] for v in output['args']):return None
    return output


def _row_destination_match(state,context):
    regions=[g for g,dag in enumerate(state['regions']) if _row_destination(dag) is not None]
    return ({'regions':regions},) if regions else ()


def _row_destination_check(state,context,p):
    regions=p.get('regions')
    if (set(p)!={'regions'} or not isinstance(regions,list) or not regions
        or any(type(g) is not int or not 0<=g<len(state['regions']) for g in regions)
        or len(regions)!=len(set(regions)) or any(_row_destination(state['regions'][g]) is None for g in regions)):
        return Decision('DENY','requires final row concat of private fused producers and compatible packed rows')
    return Decision('ALLOW','private producers write disjoint contiguous row slices of caller output; concat copies become identities')


def _row_destination_rewrite(state,context,p):
    for g in p['regions']:
        dag=state['regions'][g];join=_row_destination(dag);producers={op['out']['base']:op for op in dag['operators']};offset=0
        for view in join['args']:
            producers[view['base']]['destination_offset']=offset;offset+=view['rows']*view['cols']
        join['output_direct']=True;join['concat_forwarded']=True
    return state


def default_execution_registry():
    r=ExecutionRegistry()
    r.register(Rule('lower.fuse_quantized_matmul',1,'M04-accumulator-residency',
        ('https://arxiv.org/pdf/2006.01331','https://arxiv.org/html/2504.09688v1',
         'https://arxiv.org/html/2305.18698v1'),
        'Keep private MMUL partials in an exact INT32 accumulator and apply the original saturation only after their complete sum.',
        _fusion_match,_fusion_check,_fusion_rewrite))
    r.register(Rule('schedule.defer_gather_acquire',1,'M10-buffer-ownership-phases',
        ('https://docs.amd.com/r/2024.2-English/ug1603-ai-engine-ml-kernel-graph/Asynchronous-Buffer-Port-Access',),
        'Moving a gather input acquire after independent compute preserves its single consumption and precedes every read.',
        _defer_match,_defer_check,_defer_rewrite))
    r.register(Rule('deploy.extract_local_compute',1,'M05-spatial-temporal-deployment',
        ('https://arxiv.org/html/2401.10417v2','https://arxiv.org/html/2411.17966v1'),
        'Place unchanged local computation in its own producer; the gather consumes that same ordered row region exactly once.',
        _extract_match,_extract_check,_extract_rewrite))
    r.register(Rule('lower.tile_accumulator_grid',1,'M04-multiple-accumulation-chains',
        ('https://arxiv.org/html/2604.19106v1','https://arxiv.org/html/2511.16041v1'),
        'Tile independent output microtiles into a register grid, reuse A/B loads, and preserve each exact reduction and quantization boundary.',
        _grid_match,_grid_check,_grid_rewrite))
    r.register(Rule('memory.bind_output_destination',1,'M06-producer-destination-storage',
        ('https://arxiv.org/abs/2207.04296','https://arxiv.org/html/2510.14871v1'),
        'The sole final producer may target its caller-owned output directly when layout, extent, lifetime and non-aliasing match; no internal reader observes its old storage.',
        _output_match,_output_check,_output_rewrite))
    r.register(Rule('lower.interchange_output_loops',1,'P5-R06-register-block-schedule',
        ('https://arxiv.org/pdf/2302.07594v1','https://arxiv.org/pdf/2504.03083v1'),
        'Disjoint output microtiles have independent accumulators and writes; K order and quantization remain unchanged.',
        _loop_match,_loop_check,_loop_rewrite))
    r.register(Rule('schedule.minimize_scratch',1,'P5-R02-resource-coupled-modes',
        ('https://arxiv.org/pdf/2605.23833v1',),
        'A topological permutation of pure SSA operators preserves values; explicit allocation and final output lifetime are recomputed.',
        _schedule_match,_schedule_check,_schedule_rewrite))
    r.register(Rule('memory.forward_row_concat_destination',1,'M06-producer-destination-storage',
        ('https://arxiv.org/abs/2207.04296',),
        'Private packed row producers write disjoint complete output intervals; their only reader is the final concat, whose copies are identities.',
        _row_destination_match,_row_destination_check,_row_destination_rewrite))
    return r
