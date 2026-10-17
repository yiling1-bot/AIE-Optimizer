"""Explicit operator storage and per-invocation buffer ownership.

The checked subset is a feed-forward, single-rate graph with one output per
kernel. This is not a general proof of arbitrary ADF lock-network liveness.
"""
from collections import Counter
import re
from .contracts import identity
from .ir_backend import build_region_dag
from .planner import gather_regions
from .semantics import push_slice


def build_execution(expr, spec, groups, gather):
    children,ranges=gather_regions(groups,gather);gr=spec.m//groups
    regions=[];kernels=[]
    for g in range(groups):
        regions.append(build_region_dag(push_slice(expr,0,g*gr,(g+1)*gr,spec),spec))
        lo,hi=ranges[g]
        ports=[{'name':n,'source':'input:'+s,'size':size,'async':False}
               for n,s,size in [('x','X',spec.m*spec.k),('w0','W0',spec.k*spec.hidden),
                                ('w1','W1',spec.hidden*spec.n)]]
        for other,begin,end in children[g]:
            ports.append({'name':f'y{other}','source':f'kernel:fused_{other}',
                          'size':(end-begin)*gr*spec.n,'async':False})
        actions=[{'op':'acquire','port':p['name']} for p in ports]+[{'op':'acquire','port':'y'}]
        own=bool(children[g] or g==groups-1)
        actions.append({'op':'compute','region':g,'to':'own' if own else 'y'})
        for other,begin,end in children[g]:
            actions.append({'op':'copy','from':f'y{other}','offset':(begin-lo)*gr*spec.n,
                            'size':(end-begin)*gr*spec.n})
        if own:actions.append({'op':'copy','from':'own','offset':(g-lo)*gr*spec.n,'size':gr*spec.n})
        actions += [{'op':'release','port':p['name']} for p in ports]+[{'op':'release','port':'y'}]
        kernels.append({'id':f'fused_{g}','tile':[g,2],'inputs':ports,
                        'output_size':(hi-lo)*gr*spec.n,'actions':actions})
    return {'schema':'aie-optimizer-execution-ir-v1','regions':regions,'kernels':kernels,
            'output_kernel':kernels[-1]['id']}


def _view(v, buffers):
    fields={'base','rows','cols','pitch','dtype','layout','row','col'}
    if set(v)!=fields or v['base'] not in buffers:raise ValueError('unknown or noncanonical view')
    parent=buffers[v['base']]
    if any(type(v[k]) is not int for k in ('rows','cols','pitch','layout','row','col')):
        raise ValueError('integer view dimensions required')
    if min(v['rows'],v['cols'])<=0 or min(v['row'],v['col'])<0:
        raise ValueError('invalid view bounds')
    if (v['row']+v['rows']>parent['rows'] or v['col']+v['cols']>parent['cols']
        or any(v[k]!=parent[k] for k in ('pitch','dtype','layout'))):
        raise ValueError('view exceeds or reinterprets storage')
    value=parent['value']
    if (v['row'],v['col'],v['rows'],v['cols'])!=(0,0,parent['rows'],parent['cols']):
        value=['slice',value,v['row'],v['col'],v['rows'],v['cols']]
    return value


def _sum(terms):
    flat=[]
    for term in terms:
        flat.extend(term[1] if term[0]=='sum' else [term])
    return flat[0] if len(flat)==1 else ['sum',flat]


def dag_signature(dag,spec):
    if set(dag)!={'operators','output'}:raise ValueError('noncanonical operator DAG')
    _validate_destination_slices(dag)
    buffers={name:{'rows':r,'cols':c,'pitch':c,'layout':layout,'dtype':'int8','value':['input',name]}
             for name,r,c,layout in [('x',spec.m,spec.k,4),('w0',spec.k,spec.hidden,8),
                                     ('w1',spec.hidden,spec.n,8)]}
    for op in dag['operators']:
        if set(op)-{'accumulator_grid','output_direct','loop_order','destination_offset','concat_forwarded'}!={'kind','origin','relu','axis','args','out'}:
            raise ValueError('noncanonical operator')
        kind=op['kind'];a=op['args'];o=op['out'];values=[_view(v,buffers) for v in a]
        if 'loop_order' in op and (kind!='fused_qmm' or op['loop_order'] not in ('row_column','column_row')):
            raise ValueError('loop interchange requires independent fused MMUL output microtiles')
        if 'accumulator_grid' in op:
            grid=op['accumulator_grid']
            if (kind!='fused_qmm' or not isinstance(grid,list) or len(grid)!=2
                or any(type(v) is not int or v not in (1,2) for v in grid)
                or any(type(o.get(k)) is not int for k in ('rows','cols'))
                or o['rows']%(4*grid[0]) or o['cols']%(8*grid[1])):
                raise ValueError('invalid accumulator grid or incomplete output tiles')
        if 'output_direct' in op:
            if (op['output_direct'] is not True or o!=dag['output'] or o['dtype']!='int8'
                or any(v['base']==o['base'] for other in dag['operators'] for v in other['args'])):
                raise ValueError('direct output requires an exclusive complete INT8 result')
        if not re.fullmatch(r't\d+',o['base']) or o['base'] in buffers:raise ValueError('invalid SSA buffer')
        if (set(o)!={'base','rows','cols','pitch','dtype','layout','row','col'}
            or any(type(o[k]) is not int for k in ('rows','cols','pitch','layout','row','col'))
            or o['rows']<=0 or o['cols']<=0 or o['rows']%4 or o['cols']%8
            or (o['layout'],o['row'],o['col'],o['pitch'])!=(4,0,0,o['cols'])
            or type(op['relu']) is not bool):raise ValueError('invalid packed operator output')
        shape=lambda v:(v['rows'],v['cols'])
        same=lambda v:shape(v)==shape(o)
        def mm(left,right):
            if (left['dtype']!='int8' or right['dtype']!='int8' or left['cols']!=right['rows']
                or shape(o)!=(left['rows'],right['cols']) or left['cols']%8
                or left['cols']*16384>2147483647):raise ValueError('invalid bounded INT8 MMUL')
        if kind=='matmul':
            if len(a)!=2 or o['dtype']!='int32':raise ValueError('invalid matmul type')
            mm(*a);value=['matmul',*values]
        elif kind=='quantize':
            if len(a)!=1 or a[0]['dtype']!='int32' or o['dtype']!='int8' or not same(a[0]):
                raise ValueError('invalid quantization boundary')
            value=['quantize',op['relu'],values[0]]
        elif kind=='sum':
            if not a or o['dtype']!='int32' or any(v['dtype']!='int32' or not same(v) for v in a):
                raise ValueError('invalid INT32 sum')
            value=_sum(values)
        elif kind=='concat':
            axis=op['axis']
            if not a or type(axis) is not int or axis not in (0,1):raise ValueError('invalid concat axis')
            sizes=[shape(v) for v in a]
            if (any(v['dtype']!=o['dtype'] for v in a) or o['dtype'] not in ('int8','int32')
                or any(s[1-axis]!=shape(o)[1-axis] for s in sizes)
                or sum(s[axis] for s in sizes)!=shape(o)[axis]):raise ValueError('invalid concat shape')
            value=['concat',axis,values]
        elif kind=='fused_qmm':
            if not a or len(a)%2 or o['dtype']!='int8':raise ValueError('invalid fused quantized matmul')
            for left,right in zip(a[::2],a[1::2]):
                mm(left,right)
                if (left['layout']!=4 or right['layout']!=8 or left['row']%4 or left['col']%8
                    or right['row']%8 or right['col']%8):raise ValueError('unaligned fused MMUL view')
            if sum(v['cols'] for v in a[::2])*16384>2147483647:
                raise ValueError('fused partial accumulation may overflow INT32')
            value=['quantize',op['relu'],_sum([['matmul',x,y] for x,y in zip(values[::2],values[1::2])])]
        else:raise ValueError('unsupported execution operator: '+kind)
        buffers[o['base']]={**o,'value':value}
    result=_view(dag['output'],buffers)
    if dag['output']['dtype']!='int8':raise ValueError('INT8 region output required')
    return result


def _validate_destination_slices(dag):
    """Prove private row pieces cover one caller-owned packed output exactly."""
    marked=[op for op in dag['operators'] if 'destination_offset' in op]
    joins=[op for op in dag['operators'] if 'concat_forwarded' in op]
    if not marked and not joins:return
    if len(joins)!=1 or not marked:raise ValueError('destination slices need one complete row assembly')
    join=joins[0];o=join['out']
    if (join['concat_forwarded'] is not True or join['kind']!='concat' or join['axis']!=0
        or o!=dag['output'] or not join.get('output_direct') or o['dtype']!='int8'):
        raise ValueError('destination forwarding requires final packed row concat')
    producers={op['out']['base']:op for op in marked}
    uses=Counter(v['base'] for op in dag['operators'] for v in op['args'])
    if set(producers)!=set(v['base'] for v in join['args']):raise ValueError('destination coverage mismatch')
    offset=0
    for view in join['args']:
        op=producers[view['base']]
        if (op['kind']!='fused_qmm' or view!=op['out'] or view['cols']!=o['cols'] or view['rows']%4
            or uses[view['base']]!=1 or type(op['destination_offset']) is not int
            or op['destination_offset']!=offset or op.get('output_direct')):
            raise ValueError('destination slice is shared, incomplete, overlapping or out of order')
        offset+=view['rows']*view['cols']
    if offset!=o['rows']*o['cols']:raise ValueError('destination slices do not completely cover output')


def validate_execution(state,context):
    spec,target=context['spec'],context['target'];groups=context['groups'];unit=(spec.m//groups)*spec.n
    if (set(state)!={'schema','regions','kernels','output_kernel'}
        or state['schema']!='aie-optimizer-execution-ir-v1' or len(state['regions'])!=groups):
        raise ValueError('invalid execution IR')
    semantic=[dag_signature(dag,spec) for dag in state['regions']]
    if any((d['output']['rows'],d['output']['cols'])!=(spec.m//groups,spec.n) for d in state['regions']):
        raise ValueError('region output shape changed')
    kernels=state['kernels'];ids=[k['id'] for k in kernels];tiles=[tuple(k['tile']) for k in kernels]
    if (len(set(ids))!=len(ids) or len(set(tiles))!=len(tiles) or len(tiles)>target.max_compute_tiles
        or any(len(t)!=2 or any(type(x) is not int for x in t) or not 0<=t[0]<target.columns
               or not 0<=t[1]<target.rows for t in tiles)):
        raise ValueError('invalid kernel placement or compute tile cap')
    outputs={};computed=[];sources={'X':spec.m*spec.k,'W0':spec.k*spec.hidden,'W1':spec.hidden*spec.n}
    for kernel in kernels:
        if (set(kernel)!={'id','tile','inputs','output_size','actions'}
            or not re.fullmatch(r'[A-Za-z_]\w*',kernel['id'])):raise ValueError('invalid kernel')
        ports={p['name']:p for p in kernel['inputs']}
        if len(ports)!=len(kernel['inputs']) or set(ports)&{'y','own'}:raise ValueError('duplicate or reserved port')
        values={}
        for name,p in ports.items():
            if (set(p)!={'name','source','size','async'} or not re.fullmatch(r'[A-Za-z_]\w*',name)
                or type(p['async']) is not bool or type(p['size']) is not int or p['size']<=0 or p['size']%32):
                raise ValueError('invalid buffer port')
            prefix,source=p['source'].split(':',1)
            if prefix=='input':
                if source not in sources or p['size']!=sources[source]:raise ValueError('external input contract')
                values[name]=source
            elif prefix=='kernel':
                if source not in outputs or p['size']!=len(outputs[source])*unit:
                    raise ValueError('cyclic, forward or size-mismatched connection')
                values[name]=outputs[source]
            else:raise ValueError('unsupported source')
        size=kernel['output_size']
        if type(size) is not int or size<=0 or size%unit:raise ValueError('invalid output size')
        out=[None]*(size//unit);held=set();acquired=set();released=set();own=None;body=False;closing=False
        for action in kernel['actions']:
            kind=action['op']
            if kind in ('acquire','release'):
                if set(action)!={'op','port'}:raise ValueError('invalid ownership action')
                name=action['port']
                if name!='y' and name not in ports:raise ValueError('unknown port')
                asynchronous=name!='y' and ports[name]['async']
                if kind=='acquire':
                    if name in acquired or (not asynchronous and body):raise ValueError('invalid acquire order')
                    held.add(name);acquired.add(name)
                else:
                    if name not in held or name in released:raise ValueError('unbalanced release')
                    held.remove(name);released.add(name)
                    if not asynchronous:closing=True
                continue
            if closing:raise ValueError('work after automatic buffer release')
            body=True
            if kind=='compute':
                if set(action)!={'op','region','to'}:raise ValueError('invalid compute action')
                g=action['region']
                if (type(g) is not int or not 0<=g<groups or g in computed or own is not None
                    or any(name not in held or values.get(name)!=src for name,src in [('x','X'),('w0','W0'),('w1','W1')])):
                    raise ValueError('compute ownership or region contract')
                computed.append(g)
                if action['to']=='own':own=[g]
                elif action['to']=='y' and 'y' in held and len(out)==1 and out==[None]:out=[g]
                else:raise ValueError('invalid compute destination')
            elif kind=='copy':
                if set(action)!={'op','from','offset','size'}:raise ValueError('invalid copy action')
                src=action['from'];value=own if src=='own' else values.get(src)
                if (not isinstance(value,list) or (src!='own' and src not in held) or 'y' not in held
                    or type(action['offset']) is not int or type(action['size']) is not int
                    or action['offset']<0 or action['offset']%unit or action['size']!=len(value)*unit):
                    raise ValueError('read before acquire/compute or invalid copy')
                lo=action['offset']//unit;hi=lo+len(value)
                if hi>len(out) or any(v is not None for v in out[lo:hi]):raise ValueError('overlapping output ownership')
                out[lo:hi]=value
            else:raise ValueError('unknown execution action')
        if (held or acquired!=set(ports)|{'y'} or released!=acquired or any(v is None for v in out)):
            raise ValueError('unbalanced token counts or incomplete output')
        outputs[kernel['id']]=out
    if sorted(computed)!=list(range(groups)) or outputs.get(state['output_kernel'])!=list(range(groups)):
        raise ValueError('incomplete or reordered graph output')
    return identity({'region_values':semantic,'output_order':outputs[state['output_kernel']]})


def emit_kernel(kernel,spec,groups):
    """Interpret checked phase actions; no dispatch on a deployment family name."""
    sig='void '+kernel['id']+'('+','.join(('AsyncIn' if p['async'] else 'In')+'& '+p['name']
                                       for p in kernel['inputs'])+',Out& y)'
    lines=[]
    if any(a['op']=='compute' and a['to']=='own' for a in kernel['actions']):
        lines.append('alignas(32) int8 own[opt::GR*opt::N];')
    ports={p['name']:p for p in kernel['inputs']}
    for a in kernel['actions']:
        if a['op'] in ('acquire','release'):
            if a['port'] in ports and ports[a['port']]['async']:
                lines.append(f'{a["port"]}.{a["op"]}();')
        elif a['op']=='compute':
            dst='own' if a['to']=='own' else 'y.data()'
            lines.append(f'opt::run_group_{a["region"]}(x.data(),w0.data(),w1.data(),{dst});')
        elif a['op']=='copy':
            src='own' if a['from']=='own' else a['from']+'.data()'
            lines.append(f'for(unsigned i=0;i<{a["size"]};i+=32) aie::store_v(y.data()+{a["offset"]}+i,aie::load_v<32>({src}+i));')
    return sig,'\n '.join(lines)
