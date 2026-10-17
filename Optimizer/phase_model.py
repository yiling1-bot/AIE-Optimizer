"""Multi-input, finite-buffer phase lowering into the existing clocked backend.

Declared full-frame transfer service and explicit resource order are conditional
contracts, not inferred physical DMA/lock semantics. No native timing admission.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import heapq
from .contracts import identity,validate_contract
from .execution_ir import validate_execution
from .analyzer_bridge import (ClockedCausalGraph,CausalRule,CausalParent,DeclaredEvent,
    ClockDomain,diagnose_clocked_graph,SIGNATURE_SCHEMA,dependency_hashes)


def _need(ok,message):
    if not ok:raise ValueError(message)


def _integer(x,minimum=0):
    _need(type(x) is int and x>=minimum,'nonnegative integer picoseconds/count required')
    return x


def _validate_plan(contract,plan):
    spec,target=validate_contract(contract)
    _need(plan.get('id')==identity({k:v for k,v in plan.items() if k!='id'}) and
          plan['contract_id']==contract['id'],'phase plan identity mismatch')
    if plan.get('schema')=='aie-optimizer-ir-plan-v2':
        from .ir_planner import validate_ir_plan
        validate_ir_plan(contract,plan)
    elif plan.get('schema')=='aie-optimizer-egraph-plan-v3':
        from .egraph_planner import validate_egraph_plan
        validate_egraph_plan(contract,plan)
    else:raise ValueError('certified operator or e-class plan required for phase evaluation')
    validate_execution(plan['execution'],{'spec':spec,'target':target,'groups':plan['row_groups']})
    return spec,target


def software_binding(contract,plan,*,frames=4,slots=1):
    """Explicit synthetic round-robin policy; never a compiled slot inference."""
    _validate_plan(contract,plan);_integer(frames,1);_integer(slots,1)
    kernels=plan['execution']['kernels'];buffers={};links={}
    for k in kernels:
        for name,size in [(p['name'],p['size']) for p in k['inputs']]+[('y',k['output_size'])]:
            bid=k['id']+'.'+name
            buffers[bid]={'bytes':size,'slot_ids':[bid+':'+str(i) for i in range(slots)],
                'initial_free_ps':[0]*slots,'uses':[i%slots for i in range(frames)],'alias_of':None}
        for p in k['inputs']:
            src='input:'+p['source'].split(':')[1] if p['source'].startswith('input:') else p['source'].split(':')[1]+'.y'
            lid=k['id']+'.'+p['name']
            links[lid]={'source':src,'destination':lid,'kind':'dma','resource':'channel:'+lid}
    last=plan['execution']['output_kernel']
    links['Y']={'source':last+'.y','destination':'output:Y','kind':'dma','resource':'channel:Y'}
    binding={'schema':'aie-optimizer-phase-binding-v1','plan_id':plan['id'],
        'origin':'software_assumptions','frames':frames,
        'clocks':{'core':{'period_ps':800,'phase_ps':0},'transfer':{'period_ps':5000,'phase_ps':0}},
        'source_requests_ps':{x:[0]*frames for x in ('X','W0','W1')},'buffers':buffers,'links':links,
        'resource_orders':{link['resource']:[{'link':lid,'frame':f} for f in range(frames)] for lid,link in links.items()},
        'policy':'declared full-frame copies; independent channels; synthetic slot rotation; no native claim'}
    binding['id']=identity(binding);return binding


def lower_phases(contract,plan,binding,*,maximum_events=100000):
    """Build dependencies before evaluating any costs; refuse cyclic schedules."""
    spec,target=_validate_plan(contract,plan);_integer(maximum_events,1)
    b=deepcopy(binding)
    _need(set(b)=={'schema','plan_id','origin','frames','clocks','source_requests_ps','buffers','links','resource_orders','policy','id'}
          and b['schema']=='aie-optimizer-phase-binding-v1' and b['id']==identity({k:v for k,v in b.items() if k!='id'})
          and b['plan_id']==plan['id'],'invalid phase binding identity/schema')
    _need(b['origin']=='software_assumptions','compiled phase admission requires a separately validated wrapper/slot adapter')
    frames=_integer(b['frames'],1);kernels=plan['execution']['kernels'];byid={k['id']:k for k in kernels}
    _need(frames*sum(2*len(k['actions']) for k in kernels)<maximum_events,'phase event budget exhausted')
    _need(set(b['clocks'])=={'core','transfer'},'explicit core and transfer clocks required')
    clocks={k:ClockDomain(**v) for k,v in b['clocks'].items()}
    for c in clocks.values():c.validate()
    _need(set(b['source_requests_ps'])=={'X','W0','W1'},'complete declared source schedule required')
    for seq in b['source_requests_ps'].values():
        _need(type(seq) is list and len(seq)==frames,'one declared request per source/frame required')
        for x in seq:_integer(x)
    expected_buffers={k['id']+'.'+n:s for k in kernels
        for n,s in [(p['name'],p['size']) for p in k['inputs']]+[('y',k['output_size'])]}
    _need(set(b['buffers'])==set(expected_buffers),'complete input/output buffer inventory required')
    buffers=b['buffers'];links=b['links'];incoming={};outgoing=defaultdict(list)
    for bid,v in buffers.items():
        _need(set(v)=={'bytes','slot_ids','initial_free_ps','uses','alias_of'} and v['bytes']==expected_buffers[bid],
              'buffer extent or fields mismatch')
        _need(type(v['slot_ids']) is list and v['slot_ids'] and all(type(s) is str and s for s in v['slot_ids'])
              and len(set(v['slot_ids']))==len(v['slot_ids']),'unique explicit slots required')
        _need(type(v['initial_free_ps']) is list and len(v['initial_free_ps'])==len(v['slot_ids'])
              and type(v['uses']) is list and len(v['uses'])==frames,'slot initial state and every use required')
        for x in v['initial_free_ps']:_integer(x)
        for x in v['uses']:_need(type(x) is int and 0<=x<len(v['slot_ids']),'invalid selected slot')
    for lid,link in links.items():
        _need(type(lid) is str and lid and set(link)=={'source','destination','kind','resource'},'invalid link')
        src,dst=link['source'],link['destination']
        _need(link['kind'] in ('dma','shared'),'unsupported transfer mechanism')
        _need(src in buffers or src in ('input:X','input:W0','input:W1'),'unknown source')
        _need(dst in buffers or dst=='output:Y','unknown destination')
        _need(dst not in incoming,'each consumer buffer requires exactly one producer')
        incoming[dst]=lid;outgoing[src].append(lid)
        src_size=buffers[src]['bytes'] if src in buffers else {'input:X':spec.m*spec.k,'input:W0':spec.k*spec.hidden,'input:W1':spec.hidden*spec.n}[src]
        dst_size=buffers[dst]['bytes'] if dst in buffers else spec.m*spec.n
        _need(src_size==dst_size,'whole-frame transfer extent mismatch')
        if link['kind']=='shared':
            _need(src in buffers and dst in buffers and link['resource'] is None and buffers[dst]['alias_of']==src
                  and all(buffers[src][key]==buffers[dst][key] for key in ('slot_ids','initial_free_ps','uses')),
                  'shared transfer requires identical explicitly aliased slots/uses')
        else:_need(type(link['resource']) is str and link['resource'],'DMA requires an explicit resource')
    required_dest={'output:Y'}
    for k in kernels:
        for p in k['inputs']:
            dst=k['id']+'.'+p['name'];required_dest.add(dst)
            wanted='input:'+p['source'].split(':')[1] if p['source'].startswith('input:') else p['source'].split(':')[1]+'.y'
            _need(dst in incoming and links[incoming[dst]]['source']==wanted,'transfer differs from execution dependency')
    _need(set(incoming)==required_dest and links[incoming['output:Y']]['source']==plan['execution']['output_kernel']+'.y',
          'output or consumer connectivity changed')
    for bid,v in buffers.items():
        if v['alias_of'] is not None:
            _need(bid in incoming and links[incoming[bid]]['kind']=='shared','alias must be an explicit shared link')
    owners={}
    for bid,v in buffers.items():
        for slot in v['slot_ids']:
            owner=v['alias_of'] or bid
            _need(slot not in owners or owners[slot]==owner,'undeclared physical slot alias')
            owners[slot]=owner
    expected_transactions={(lid,f) for lid,l in links.items() if l['kind']=='dma' for f in range(frames)}
    seen=set();resource_previous={}
    for resource,order in b['resource_orders'].items():
        _need(type(resource) is str and resource and type(order) is list and order,'explicit nonempty resource order required')
        previous=None
        for row in order:
            _need(set(row)=={'link','frame'},'only declared transfer ordinals allowed')
            lid,f=row['link'],row['frame'];_integer(f)
            _need((lid,f) in expected_transactions and (lid,f) not in seen and links[lid]['resource']==resource,'resource order incomplete, duplicate or mismatched')
            seen.add((lid,f));resource_previous[(lid,f)]=previous;previous=(lid,f)
    _need(seen==expected_transactions,'every DMA episode requires one explicit arbitration position')
    roots={};rules={};components={};metadata={}
    def root(eid,time):roots[eid]=DeclaredEvent(eid,time);return eid
    zero=root('request:reset',0)
    def add(eid,parents,clock='core',component=None,meta=None):
        _need(eid not in rules and eid not in roots,'duplicate event identity')
        _need(len(rules)+len(roots)<maximum_events,'phase event budget exhausted')
        # Combine wait prerequisites before attaching service to its own event.
        parents=tuple(dict.fromkeys(parents))
        rules[eid]=CausalRule(eid,clocks[clock],tuple(CausalParent(p,0,component) for p in parents),eid)
        metadata[eid]=meta or {};return eid
    def cost(kind,operations,interactions):
        facts={'architecture':target.architecture,'clocks':b['clocks'],'units':{'duration':'ps'},
            'boundaries':{'kind':kind,'service':'exclusive active service; waits in causal dependencies'},
            'operations':operations,'interactions':interactions,
            'compile_identity':identity({'generator':plan['implementation_fingerprint'],'toolchain':target.toolchain,
                                         'operations':operations,'interactions':interactions})}
        cid=identity({'mechanism':kind,**facts})
        components[cid]={'schema':SIGNATURE_SCHEMA,'component_id':cid,'mechanism':kind,**facts};return cid
    action_id=lambda kid,f,i:f'actor:{kid}:{f}:{i}'
    acquire={};release={}
    for k in kernels:
        for i,a in enumerate(k['actions']):
            if a['op'] in ('acquire','release'):
                for f in range(frames):(acquire if a['op']=='acquire' else release)[(k['id']+'.'+a['port'],f)]=action_id(k['id'],f,i)
    def transfer_end(lid,f):return f'transfer:{lid}:{f}:done'
    def publication(src,f):
        return source_roots[(src[6:],f)] if src.startswith('input:') else release[(src,f)]
    source_roots={(s,f):root(f'request:{s}:{f}',seq[f]) for s,seq in b['source_requests_ps'].items() for f in range(frames)}
    previous_use={}
    for bid,v in buffers.items():
        last={}
        for f,slot in enumerate(v['uses']):previous_use[(bid,f)]=last.get(slot);last[slot]=f
    def free(bid,f):
        prev=previous_use[(bid,f)];v=buffers[bid]
        if prev is None:return [root(f'initial:{bid}:{v["uses"][f]}',v['initial_free_ps'][v['uses'][f]])]
        if bid.endswith('.y'):
            _need(bool(outgoing[bid]),'unconsumed output')
            return [release[(links[l]['destination'],prev)] if links[l]['kind']=='shared' else transfer_end(l,prev) for l in outgoing[bid]]
        return [release[(bid,prev)]]
    for k in kernels:
        kid=k['id'];kernel_context={'tile':k['tile'],'inputs':k['inputs'],'output_size':k['output_size']}
        start=add('startup:'+kid,[zero],component=cost('actor_startup',{'actions':k['actions']},kernel_context))
        for f in range(frames):
            previous=start if f==0 else action_id(kid,f-1,len(k['actions'])-1)
            for i,a in enumerate(k['actions']):
                eid=action_id(kid,f,i);parents=[previous]
                if a['op']=='acquire':
                    bid=kid+'.'+a['port']
                    if a['port']=='y':parents+=free(bid,f)
                    else:parents.append(transfer_end(incoming[bid],f))
                gate=add(eid+':ready',parents,meta={'actor':kid,'frame':f,'action':a,'phase':'wait_conditions'})
                if a['op']=='compute':
                    ops={'region':plan['execution']['regions'][a['region']],'destination':a['to']};kind='complete_region'
                else:ops=a;kind='wrapper_'+a['op']
                previous=add(eid,[gate],component=cost(kind,ops,kernel_context),meta={'actor':kid,'frame':f,'action':a,'phase':'active_service'})
    for lid,l in links.items():
        for f in range(frames):
            parents=[publication(l['source'],f)]
            if l['destination'] in buffers:parents+=free(l['destination'],f)
            if l['kind']=='shared':
                add(transfer_end(lid,f),parents,meta={'link':lid,'frame':f,'phase':'shared_publication'});continue
            prior=resource_previous[(lid,f)]
            if prior is not None:parents.append(transfer_end(*prior))
            start=add(f'transfer:{lid}:{f}:ready',parents,'transfer',meta={'link':lid,'frame':f,'phase':'transfer_wait'})
            size=buffers[l['source']]['bytes'] if l['source'] in buffers else buffers[l['destination']]['bytes']
            cid=cost('whole_frame_transfer',{'bytes':size,'kind':'dma'},
                     {'link':l,'source_buffer':buffers.get(l['source']), 'destination_buffer':buffers.get(l['destination'])})
            add(transfer_end(lid,f),[start],'transfer',cid,{'link':lid,'frame':f,'phase':'transfer_service'})
    # Topological sorting permits cross-actor and future-listed DMA dependencies.
    indegree={};children=defaultdict(list)
    for eid,r in rules.items():
        indegree[eid]=0
        for p in r.parents:
            _need(p.event_id in roots or p.event_id in rules,'missing phase predecessor')
            if p.event_id in rules:indegree[eid]+=1;children[p.event_id].append(eid)
    ready=[eid for eid,n in indegree.items() if not n];heapq.heapify(ready);ordered=[]
    while ready:
        eid=heapq.heappop(ready);ordered.append(rules[eid])
        for child in children[eid]:
            indegree[child]-=1
            if indegree[child]==0:heapq.heappush(ready,child)
    _need(len(ordered)==len(rules),'cyclic ownership/resource schedule: deadlock or unsupported arbitration')
    graph=ClockedCausalGraph(tuple(roots.values()),tuple(ordered),tuple(components),('binding:'+b['id'],'plan:'+plan['id']))
    comparison=identity({'contract':contract['id'],'frames':frames,'clocks':b['clocks'],
                         'source_requests_ps':b['source_requests_ps'],'metric':'reset_to_complete_output'})
    result={'schema':'aie-optimizer-phase-graph-v1','plan_id':plan['id'],'binding_id':b['id'],'graph':asdict(graph),
        'components':list(components.values()),'event_metadata':metadata,
        'output_events':[transfer_end(incoming['output:Y'],f) for f in range(frames)],
        'comparison_key':comparison,'physical_cores_planned':len(kernels),'native_admitted':False,
        'dependency_hashes':dependency_hashes(),'scope':'conditional full-frame multi-input ownership; explicit slot use and fixed channel order; no inferred native locks/arbiter'}
    result['id']=identity(result);return result


def evaluate_graph(lowered,assignment):
    _need(lowered.get('schema')=='aie-optimizer-phase-graph-v1' and lowered.get('id')==identity({k:v for k,v in lowered.items() if k!='id'})
          and lowered['dependency_hashes']==dependency_hashes(),
          'phase backend changed or invalid lowered graph')
    raw=lowered['graph']
    graph=ClockedCausalGraph(tuple(DeclaredEvent(**e) for e in raw['declared_events']),
        tuple(CausalRule(r['id'],ClockDomain(**r['clock']),tuple(CausalParent(**p) for p in r['parents']),r['observation_semantics']) for r in raw['rules']),
        tuple(raw['cost_names']),tuple(raw['evidence']),raw['scope'])
    result=diagnose_clocked_graph(graph,assignment)
    completions=[result['times_ps'][e] for e in lowered['output_events']]
    result.update(frame_completion_ps=completions,first_output_ps=completions[0],
                  finite_frame_intervals_ps=[b-a for a,b in zip(completions,completions[1:])])
    events={r['id']:r for r in result['events']};seen=set();pending=[lowered['output_events'][0]]
    while pending:
        event=pending.pop()
        if event in seen or event not in events:continue
        seen.add(event);pending.extend(events[event]['critical_parents'])
    result['first_output_critical_events']=[{'id':e,**lowered['event_metadata'][e]} for e in events if e in seen]
    return result
