"""Lift a closed subset of compiler-typed array loops to tensor SSA.

Proves full output coverage, signed arithmetic bounds and no loop-carried
array dependence. No helper name is a semantic assertion. Unknown calls or
protocol operations remain outside this mathematical frontend.
"""
from dataclasses import dataclass
from ..contracts import identity
from .ir import GraphBuilder, bounds, tensor_type, fail

META = {'TYPE_REF','NAMESPACE_REF','TEMPLATE_REF'}


def children(n): return [c for c in n['children'] if c['kind'] not in META]


def unwrap(n):
    while n['kind'] in ('UNEXPOSED_EXPR','PAREN_EXPR') and len(children(n))==1:
        n=children(n)[0]
    return n


def items(n): return children(n) if n['kind']=='COMPOUND_STMT' else [n]


def where(n): return f"{n['site']['file']}:{n['site']['line']}:{n['site']['column']}:{n['kind']}"


def integer_type(t, site):
    if t.get('volatile') or not t.get('signed') or t.get('bits') not in (8,16,32,64):
        fail('signed non-volatile integer type required',site,'cpp_type')
    return t['bits']


def array_type(t, site, parameter=False):
    if parameter:
        if t['kind']!='LVALUEREFERENCE':
            fail('fixed-extent array references required; pointer extent/aliasing unknown',site,'cpp_array_extent')
        t=t['pointee']
    shape=[];readonly=t['const']
    while t['kind']=='CONSTANTARRAY':
        if t.get('volatile'):fail('volatile array access unsupported',site)
        shape.append(t['extent']);t=t['element'];readonly=readonly or t['const']
    if len(shape)!=2:fail('this loop lifter requires rank-two arrays',site,'cpp_array_extent')
    return tensor_type(shape,integer_type(t,site)),readonly


@dataclass
class Array:
    typ: dict
    readonly: bool
    value: str | None
    declaration: str


class Lifter:
    def __init__(self, ast, standard):
        self.graph=GraphBuilder();self.functions={f['id']:f for f in ast['functions']}
        self.entry=ast['entry_id'];self.stack=[];self.globals={};self.obligations=[]
        self.modular_narrowing=standard in ('-std=c++20','-std=c++23')
        def reject_thread_state(n):
            if n['kind']=='VAR_DECL' and n.get('tls','NONE')!='NONE':
                fail('thread-local state is not a fresh automatic value',where(n),'cpp_stateful_storage')
            for ch in n['children']:reject_thread_state(ch)
        for n in list(self.functions.values())+ast['declarations']:reject_thread_state(n)
        for f in self.functions.values(): self.function_body(f)
        for n in ast['declarations']:
            if n['kind'] in ('TYPEDEF_DECL','TYPE_ALIAS_DECL'):continue
            if n['kind']=='ENUM_DECL':
                for e in n['children']:
                    if e['kind']=='ENUM_CONSTANT_DECL' and 'constant' in e:self.globals[e['id']]=e['constant']
                continue
            if n['kind']=='VAR_DECL' and n['type']['const'] and not n['type']['volatile'] and 'constant' in n and n['type'].get('signed'):
                init=children(n)
                if len(init)!=1:fail('one pure global constant initializer required',where(n),'cpp_global_effect')
                value=self.global_constant(init[0])
                if value!=n['constant']:fail('global constant evaluation disagrees with compiler',where(n))
                self.globals[n['id']]=value;continue
            fail('global initialization/type outside the closed pure-array subset',where(n),'cpp_global_effect')

    def global_constant(self,n):
        kind=n['kind'];ch=children(n)
        if kind in ('UNEXPOSED_EXPR','PAREN_EXPR','CSTYLE_CAST_EXPR','CXX_STATIC_CAST_EXPR') and len(ch)==1:
            value=self.global_constant(ch[0]);bits=integer_type(n['type'],where(n))
            if not bounds(bits)[0]<=value<=bounds(bits)[1]:fail('global narrowing requires separate handling',where(n),'cpp_global_effect')
            return value
        if kind=='DECL_REF_EXPR' and n['reference']['id'] in self.globals:return self.globals[n['reference']['id']]
        if kind=='INTEGER_LITERAL' and 'constant' in n:return n['constant']
        if kind=='UNARY_OPERATOR' and n['operator'] in ('+','-'):
            value=self.global_constant(ch[0]);value=value if n['operator']=='+' else -value
        elif kind=='BINARY_OPERATOR' and n['operator'] in ('+','-','*','/'):
            a,b=[self.global_constant(c) for c in ch]
            if n['operator']=='+':value=a+b
            elif n['operator']=='-':value=a-b
            elif n['operator']=='*':value=a*b
            elif b and a%b==0:value=a//b
            else:fail('non-exact/undefined constant division',where(n))
        else:fail('global initializer may have side effects',where(n),'cpp_global_effect')
        bits=integer_type(n['type'],where(n))
        if not bounds(bits)[0]<=value<=bounds(bits)[1]:fail('global signed overflow',where(n),'cpp_signed_overflow')
        return value

    def function_body(self,f):
        allowed={'PARM_DECL','COMPOUND_STMT'}|META
        if any(n['kind'] not in allowed for n in f['children']):fail('unhandled function attributes/declarations',where(f))
        bodies=[n for n in f['children'] if n['kind']=='COMPOUND_STMT']
        if len(bodies)!=1:fail('one function definition required',where(f))
        return bodies[0]

    def enter(self,f):
        if f['id'] in self.stack or len(self.stack)>=24:fail('recursive/deep function call',where(f),'recursive_call')
        self.stack.append(f['id'])

    def lift(self, name):
        f=self.functions[self.entry]
        if f['return_type']['kind']!='VOID':fail('entry must return void through explicit array outputs',where(f))
        arrays={};outputs=[];signature=[]
        for pos,p in enumerate(f['parameters']):
            typ,readonly=array_type(p['type'],where(p),True)
            if not readonly and not p['type']['restrict']:
                fail('writable array reference requires __restrict; output/input non-aliasing not established',where(p),'cpp_alias_not_proven')
            value=self.graph.input(p['name'],typ,pos) if readonly else None
            array=Array(typ,readonly,value,p['id']);arrays[p['id']]=array
            signature.append({'position':pos,'name':p['name'],'cpp_type':p['type']['spelling'],
                              'tensor_type':typ,'direction':'input' if readonly else 'output',
                              'input_id':value,'restricted':p['type']['restrict']})
            if not readonly:outputs.append((pos,array))
        if not outputs:fail('entry has no writable array outputs',where(f))
        self.execute(f,arrays)
        if any(a.value is None for _,a in outputs):fail('some output arrays were not fully initialized',where(f))
        program={'schema':'aie-optimizer-tensor-ssa-v1','entry':name,'inputs':self.graph.inputs,'nodes':self.graph.nodes,
            'outputs':[{'position':i,'value':a.value,'type':a.typ,'source_parameter':p} for i,(p,a) in enumerate(outputs)],
            'effects':'writes to restricted output arrays; inputs read-only; locals do not escape','physical_mapping':None}
        program['id']=identity(program)
        return program,{'parameters':signature,'alias_contract':'each restricted writable output is disjoint from read inputs and other outputs',
                        'source_storage':'dense C++ rank-two arrays','original_runtime_interface_preserved':False,
                        'proof_obligations':self.obligations}

    def execute(self,f,arrays):
        self.enter(f)
        try:
            for stmt in children(self.function_body(f)):
                if stmt['kind']=='DECL_STMT':
                    for decl in children(stmt):
                        if decl['kind']!='VAR_DECL' or decl.get('storage') not in ('NONE','AUTO'):
                            fail('local automatic array declaration required',where(decl))
                        typ,readonly=array_type(decl['type'],where(decl))
                        # Literal dimensions appear as AST children; initialization does not.
                        if any(ch['kind'] not in META|{'INTEGER_LITERAL','DECL_REF_EXPR','BINARY_OPERATOR','UNARY_OPERATOR','PAREN_EXPR'} for ch in decl['children']):
                            fail('local arrays must be uninitialized declarations',where(decl))
                        arrays[decl['id']]=Array(typ,readonly,None,decl['id'])
                elif stmt['kind']=='FOR_STMT': self.loop_region(stmt,arrays)
                elif stmt['kind']=='CALL_EXPR':
                    target=self.call_target(stmt)
                    if target['return_type']['kind']!='VOID':fail('unused scalar call unsupported',where(stmt))
                    actual={}
                    if len(target['parameters'])!=len(stmt['arguments']):fail('call arity mismatch',where(stmt))
                    for p,a in zip(target['parameters'],stmt['arguments']):
                        ref=unwrap(a)
                        if ref['kind']!='DECL_REF_EXPR' or ref['reference']['id'] not in arrays:
                            fail('array helper arguments must be whole known arrays',where(a))
                        actual[p['id']]=arrays[ref['reference']['id']]
                        typ,_=array_type(p['type'],where(p),True)
                        if typ!=actual[p['id']].typ:fail('array helper type mismatch',where(stmt))
                    self.execute(target,actual)
                elif stmt['kind']=='RETURN_STMT' and not children(stmt):
                    if stmt is not children(self.function_body(f))[-1]:fail('early return unsupported',where(stmt))
                else:fail('statement has no verified tensor semantics',where(stmt),'unsupported_cpp_statement')
        finally:self.stack.pop()

    def call_target(self,n):
        ref=n.get('reference',{});key=ref.get('definition')
        if key not in self.functions:fail('unresolved/external/API call: '+ref.get('name','?'),where(n),'unsupported_cpp_call')
        return self.functions[key]

    def constant(self,n):
        return self.global_constant(n)

    def loop(self,n):
        ch=children(n)
        if len(ch)!=4 or ch[0]['kind']!='DECL_STMT' or len(children(ch[0]))!=1:
            fail('canonical for(init; bound; step) required',where(n),'cpp_loop')
        decl=children(ch[0])[0];ds=children(decl)
        if decl['kind']!='VAR_DECL' or len(ds)!=1 or self.constant(ds[0])!=0:
            fail('loop must start at zero',where(n),'cpp_loop')
        if decl['type'].get('volatile') or decl['type'].get('bits')!=32:fail('32-bit loop counter required',where(n),'cpp_loop')
        cond=ch[1];cc=children(cond)
        if cond['kind']!='BINARY_OPERATOR' or cond['operator']!='<' or len(cc)!=2 or unwrap(cc[0]).get('reference',{}).get('id')!=decl['id']:
            fail('loop must have a static exclusive upper bound',where(n),'cpp_loop')
        end=self.constant(cc[1]);step=ch[2]
        if not 0<end<2**31-1 or step['kind']!='UNARY_OPERATOR' or step['operator']!='++' or unwrap(children(step)[0]).get('reference',{}).get('id')!=decl['id']:
            fail('unit forward loop with finite safe counter required',where(n),'cpp_loop')
        return decl['id'],end,ch[3]

    def affine(self,n,domains):
        # Preserve casts while checking every intermediate index expression.
        if n['kind'] in ('UNEXPOSED_EXPR','PAREN_EXPR','CSTYLE_CAST_EXPR','CXX_STATIC_CAST_EXPR') and len(children(n))==1:
            result=self.affine(children(n)[0],domains)
            return self.check_index_range(result,n,domains)
        if n['kind']=='DECL_REF_EXPR' and n['reference']['id'] in domains:return ({n['reference']['id']:1},0)
        if n['kind']=='BINARY_OPERATOR' and n['operator'] in ('+','-','*'):
            a,b=[self.affine(ch,domains) for ch in children(n)]
            if n['operator']=='*':
                if a[0] and b[0]:fail('non-affine array index',where(n))
                if not a[0]:a,b=b,a
                return self.check_index_range(({k:v*b[1] for k,v in a[0].items()},a[1]*b[1]),n,domains)
            sign=1 if n['operator']=='+' else -1
            coeff=dict(a[0])
            for k,v in b[0].items():coeff[k]=coeff.get(k,0)+sign*v
            return self.check_index_range(({k:v for k,v in coeff.items() if v},a[1]+sign*b[1]),n,domains)
        return {},self.constant(n)

    def check_index_range(self,result,n,domains):
        coeff,base=result;lo=hi=base
        for axis,multiplier in coeff.items():
            endpoint=multiplier*(domains[axis]-1);lo+=min(0,endpoint);hi+=max(0,endpoint)
        typ=n['type'];bits=typ.get('bits')
        if not bits or bits>64:fail('integer index type required',where(n),'cpp_access')
        lower,upper=bounds(bits) if typ.get('signed') else [0,2**bits-1]
        if lo<lower or hi>upper:fail('array index intermediate overflow/narrowing cannot be excluded',where(n),'cpp_index_overflow')
        return result

    def access(self,n,arrays,domains):
        n=unwrap(n);ix=[]
        while n['kind'] in ('CSTYLE_CAST_EXPR','CXX_STATIC_CAST_EXPR') and len(children(n))==1:
            n=unwrap(children(n)[0])
        while n['kind']=='ARRAY_SUBSCRIPT_EXPR':
            base,index=children(n);ix.append(self.affine(index,domains));n=unwrap(base)
        if n['kind']!='DECL_REF_EXPR' or n['reference']['id'] not in arrays or len(ix)!=2:
            fail('known two-dimensional array access required',where(n),'cpp_access')
        array=arrays[n['reference']['id']];ix.reverse();axes=[]
        for (coeff,offset),extent in zip(ix,array.typ['shape']):
            if offset or len(coeff)!=1 or next(iter(coeff.values()))!=1:
                fail('full direct dimension access required; other affine layouts need a layout adapter',where(n),'cpp_access')
            axis=next(iter(coeff))
            if domains[axis]!=extent:fail('array extent and loop domain mismatch',where(n),'cpp_access')
            axes.append(axis)
        return array,axes

    def origin(self,n):return {'language':'C++','site':where(n),'compiler_type':n['type']['spelling'],'call_stack':list(self.stack)}

    def read_matrix(self,n,arrays,domains,axes,destination):
        # Only widening conversions may surround a matrix operand.
        cursor=n
        while cursor['kind'] in ('UNEXPOSED_EXPR','PAREN_EXPR','CSTYLE_CAST_EXPR','CXX_STATIC_CAST_EXPR'):
            ch=children(cursor)
            if len(ch)!=1:break
            ct=cursor['type'];bt=ch[0]['type']
            if ct.get('bits') and bt.get('bits') and (not ct.get('signed') or not bt.get('signed') or ct['bits']<bt['bits']):
                fail('matrix operand conversion is not signed widening',where(cursor))
            cursor=ch[0]
        array,actual=self.access(cursor,arrays,domains)
        if array is destination:fail('loop-carried in-place array dependence',where(cursor),'cpp_alias_dependence')
        if array.value is None:fail('read of uninitialized array',where(cursor),'uninitialized_value')
        if actual==axes:return array.value
        if actual==axes[::-1]:
            return self.graph.add('transpose',(array.value,),tensor_type(array.typ['shape'][::-1],array.typ['bits']),{'permutation':[1,0]},self.origin(n))
        fail('matrix access does not match output/reduction axes',where(cursor),'cpp_access')

    def loop_region(self,outer,arrays):
        iv,m,inside=self.loop(outer);body=items(inside)
        if len(body)!=1 or body[0]['kind']!='FOR_STMT':fail('two output loops required',where(outer),'cpp_loop')
        jv,n,inside=self.loop(body[0]);stmts=items(inside);domains={iv:m,jv:n}
        if not stmts:fail('empty loop region',where(outer))
        store=stmts[-1];parts=children(store)
        if store['kind']!='BINARY_OPERATOR' or store['operator']!='=' or len(parts)!=2:
            fail('region must finish with one full output store',where(store))
        dest,axes=self.access(parts[0],arrays,domains)
        if dest.readonly or axes[0]==axes[1] or set(axes)!={iv,jv}:fail('output domain not fully and uniquely written',where(store))
        shape=dest.typ['shape'];env={}
        if len(stmts)==3 and stmts[0]['kind']=='DECL_STMT' and stmts[1]['kind']=='FOR_STMT':
            declarations=children(stmts[0])
            if len(declarations)!=1:fail('single accumulator declaration required',where(stmts[0]))
            acc=declarations[0]
            if acc['kind']!='VAR_DECL' or integer_type(acc['type'],where(acc))!=32 or acc.get('storage') not in ('NONE','AUTO'):
                fail('automatic signed i32 accumulator required',where(acc))
            if len(children(acc))!=1:fail('accumulator must have one explicit initializer',where(acc),'uninitialized_value')
            init=self.constant(children(acc)[0]);kv,k,redbody=self.loop(stmts[1]);domains[kv]=k
            red=items(redbody)
            if len(red)!=1:fail('one recurrence statement required',where(stmts[1]))
            red=red[0];rc=children(red)
            if len(rc)!=2 or unwrap(rc[0]).get('reference',{}).get('id')!=acc['id']:
                fail('recurrence must update the declared accumulator',where(red))
            product=rc[1]
            if red['kind']=='BINARY_OPERATOR' and red['operator']=='=':
                plus=unwrap(product);pc=children(plus)
                if plus['kind']!='BINARY_OPERATOR' or plus['operator']!='+' or len(pc)!=2:
                    fail('recurrence must be accumulator plus product',where(red))
                if unwrap(pc[0]).get('reference',{}).get('id')==acc['id']:product=pc[1]
                elif unwrap(pc[1]).get('reference',{}).get('id')==acc['id']:product=pc[0]
                else:fail('recurrence loses previous accumulator',where(red))
            elif red['kind']!='COMPOUND_ASSIGNMENT_OPERATOR' or red['operator']!='+=':
                fail('only additive full reduction supported',where(red))
            product=unwrap(product)
            if product['kind']!='BINARY_OPERATOR' or product['operator']!='*' or integer_type(product['type'],where(product))!=32:
                fail('recurrence requires signed i32 multiplication',where(product))
            lhs,rhs=children(product)
            # Try both commutative operand orientations without mutating the graph.
            la,laxes=self.access(lhs,arrays,domains);ra,raxes=self.access(rhs,arrays,domains)
            expected0,expected1=[axes[0],kv],[kv,axes[1]]
            if set(laxes)==set(expected1) and set(raxes)==set(expected0):lhs,rhs=rhs,lhs
            a=self.read_matrix(lhs,arrays,domains,expected0,dest);b=self.read_matrix(rhs,arrays,domains,expected1,dest)
            initial=self.graph.add('constant',(),tensor_type(shape,32),{'values':[init]},self.origin(acc))
            result=self.graph.add('matmul',(a,b,initial),tensor_type(shape,32),{'promotion':'signed','accumulation':'modular'},self.origin(red))
            if self.graph.values[result]['may_wrap']:fail('C++ signed accumulator overflow cannot be excluded',where(red),'cpp_signed_overflow')
            env[acc['id']]=result
            self.obligations.append({'site':where(red),'proof':'complete affine reduction; every partial i32 sum in range','reduction_extent':k})
        elif len(stmts)!=1:fail('unsupported statements inside output loops',where(outer))
        value=self.expression(parts[1],env,arrays,domains,axes,dest)
        value=self.convert(value,dest.typ['bits'],store)
        if self.graph.values[value]['type']!=dest.typ:fail('output type/coverage mismatch',where(store))
        dest.value=value
        self.obligations.append({'site':where(outer),'proof':'each output element written once; destination absent from all array reads','shape':shape})

    def const_tensor(self,value,shape,bits,n):
        if not bounds(bits)[0]<=value<=bounds(bits)[1]:fail('literal outside signed type range',where(n))
        return self.graph.add('constant',(),tensor_type(shape,bits),{'values':[value]},self.origin(n))

    def convert(self,value,bits,n):
        old=self.graph.values[value];width=old['type']['bits']
        if width==bits:return value
        if bits<width and not self.modular_narrowing and (old['range'][0]<bounds(bits)[0] or old['range'][1]>bounds(bits)[1]):
            fail('pre-C++20 out-of-range signed narrowing is implementation-defined',where(n),'cpp_narrowing')
        return self.graph.add('extend_signed' if bits>width else 'truncate',(value,),tensor_type(old['type']['shape'],bits),{},self.origin(n))

    def expression(self,n,env,arrays,domains,axes,dest):
        kind=n['kind'];ch=children(n);shape=dest.typ['shape']
        if kind in ('UNEXPOSED_EXPR','PAREN_EXPR','CSTYLE_CAST_EXPR','CXX_STATIC_CAST_EXPR') and len(ch)==1:
            value=self.expression(ch[0],env,arrays,domains,axes,dest)
            return self.convert(value,integer_type(n['type'],where(n)),n)
        if kind=='DECL_REF_EXPR':
            key=n['reference']['id']
            if key in env:return env[key]
            if key in self.globals:return self.const_tensor(self.globals[key],shape,integer_type(n['type'],where(n)),n)
            fail('unbound scalar/global read',where(n))
        if kind=='ARRAY_SUBSCRIPT_EXPR':return self.read_matrix(n,arrays,domains,axes,dest)
        if kind in ('INTEGER_LITERAL','CHARACTER_LITERAL') or (kind=='UNARY_OPERATOR' and n['operator'] in ('+','-')):
            return self.const_tensor(self.constant(n),shape,integer_type(n['type'],where(n)),n)
        if kind=='BINARY_OPERATOR' and n['operator'] in ('+','*'):
            bits=integer_type(n['type'],where(n));args=[self.convert(self.expression(v,env,arrays,domains,axes,dest),bits,n) for v in ch]
            value=self.graph.add('add' if n['operator']=='+' else 'mul',args,tensor_type(shape,bits),{},self.origin(n))
            if self.graph.values[value]['may_wrap']:fail('C++ signed arithmetic overflow cannot be excluded',where(n),'cpp_signed_overflow')
            return value
        if kind=='CONDITIONAL_OPERATOR':return self.conditional(n,env,arrays,domains,axes,dest)
        if kind=='CALL_EXPR':
            f=self.call_target(n);self.enter(f)
            try:
                if len(f['parameters'])!=len(n['arguments']):fail('scalar helper arity mismatch',where(n))
                args={p['id']:self.convert(self.expression(a,env,arrays,domains,axes,dest),integer_type(p['type'],where(p)),p) for p,a in zip(f['parameters'],n['arguments'])}
                body=children(self.function_body(f))
                for stmt in body:
                    if stmt['kind']=='DECL_STMT':
                        for decl in children(stmt):
                            if decl['kind']!='VAR_DECL' or decl.get('storage') not in ('NONE','AUTO') or len(children(decl))!=1:fail('unsupported scalar helper local',where(decl))
                            args[decl['id']]=self.convert(self.expression(children(decl)[0],args,arrays,domains,axes,dest),integer_type(decl['type'],where(decl)),decl)
                    elif stmt['kind']=='RETURN_STMT' and stmt is body[-1] and len(children(stmt))==1:
                        value=self.expression(children(stmt)[0],args,arrays,domains,axes,dest)
                        return self.convert(value,integer_type(f['return_type'],where(f)),stmt)
                    else:fail('scalar helper is not a pure expression function',where(stmt))
                fail('scalar helper missing return',where(f))
            finally:self.stack.pop()
        fail('expression has no verified tensor semantics',where(n),'unsupported_cpp_expression')

    def conditional(self,n,env,arrays,domains,axes,dest):
        cond,yes,no=children(n);cond=unwrap(cond)
        if cond['kind']!='BINARY_OPERATOR' or cond['operator'] not in ('<','>','<=','>='):
            fail('only proved scalar min/max conditionals supported',where(n))
        a,b=children(cond);limit=self.constant(b)
        base=self.expression(a,env,arrays,domains,axes,dest)
        y=self.expression(yes,env,arrays,domains,axes,dest);other=self.expression(no,env,arrays,domains,axes,dest)
        yn=self.graph.values[y]
        if yn['range']!=[limit,limit]:fail('conditional true branch must be the compared bound',where(n))
        bits=integer_type(n['type'],where(n));other=self.convert(other,bits,n)
        op='min_signed' if cond['operator'] in ('>','>=') else 'max_signed'
        # Prove that the other branch stays on the correct side of the bound
        # under each predicate interval. This also handles nested clamp syntax.
        initial=self.graph.values[base]['range'];strict=cond['operator'] in ('>','<')
        if op=='min_signed':
            yes_range=[max(initial[0],limit+int(strict)),initial[1]]
            no_range=[initial[0],min(initial[1],limit-int(not strict))]
        else:
            yes_range=[initial[0],min(initial[1],limit-int(strict))]
            no_range=[max(initial[0],limit+int(not strict)),initial[1]]
        def interval(value,replacement):
            if value==base:return replacement
            item=self.graph.values[value]
            if item.get('op') in ('max_signed','min_signed'):
                lohi=[interval(v,replacement) for v in item['operands']]
                fn=max if item['op']=='max_signed' else min
                return [fn(t[0] for t in lohi),fn(t[1] for t in lohi)]
            return item['range']
        yr,nr=interval(other,yes_range),interval(other,no_range)
        yes_ok=yes_range[0]>yes_range[1] or (yr[0]>=limit if op=='min_signed' else yr[1]<=limit)
        no_ok=no_range[0]>no_range[1] or (nr[1]<=limit if op=='min_signed' else nr[0]>=limit)
        if not yes_ok or not no_ok:fail('conditional cannot be proven equivalent to min/max',where(n))
        bound=self.const_tensor(limit,dest.typ['shape'],bits,n)
        return self.graph.add(op,(other,bound),tensor_type(dest.typ['shape'],bits),{},self.origin(n))
