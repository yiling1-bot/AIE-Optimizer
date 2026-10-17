"""Operator-directed lowering, independent of whole-program mode names.

Supported tensor expressions become a CSE DAG with explicit rematerialization
scopes, a serial schedule, liveness-based scratch reuse, and AIE MMUL microtiles.
This is a correctness-first lowering, not a tuned kernel/schedule optimizer.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from .semantics import expression_id, infer, push_slice
from .registry import validate_expression
from .scratch_schedule import allocation_step


@dataclass(frozen=True)
class View:
    base: str
    rows: int
    cols: int
    pitch: int
    dtype: str
    layout: int = 0  # 0=row-major; 4/8=packed row-block height, column-block width=8
    row: int = 0
    col: int = 0

    def get(self, r, c):
        r = f'({r}+{self.row})'; c = f'({c}+{self.col})'
        if self.layout:
            h = self.layout
            index = f'(({r}/{h})*({self.pitch}/8)+({c}/8))*{h*8}+({r}%{h})*8+({c}%8)'
        else:
            index = f'{r}*{self.pitch}+{c}'
        return f'{self.base}[{index}]'

    def descriptor(self):
        kind='V32' if self.dtype=='int32' else 'V8'
        return f'{kind}{{{self.base},{self.rows},{self.cols},{self.pitch},{self.layout},{self.row},{self.col}}}'


def build_region_dag(expr, spec):
    validate_expression(expr, spec)
    cache = {}; ops = []

    def visit(e, scope=(), path=()):
        if e.op == 'rematerialize':
            # A distinct occurrence executes its producer again, while its own
            # internal common subexpressions may still share storage/work.
            return visit(e.args[0], scope+(path,), path+(0,))
        key = (expression_id(e), () if e.op == 'input' else scope)
        if key in cache:
            return cache[key]
        t = infer(e, spec)
        if e.op == 'input':
            result = View({'X':'x','W0':'w0','W1':'w1'}[e.name],
                          t.rows,t.cols,t.cols,t.dtype,4 if e.name=='X' else 8)
        else:
            args = [visit(a, scope, path+(i,)) for i,a in enumerate(e.args)]
            if e.op == 'slice':
                a = args[0]
                result = View(a.base,t.rows,t.cols,a.pitch,a.dtype,a.layout,
                              a.row+(e.begin if e.axis==0 else 0),
                              a.col+(e.begin if e.axis==1 else 0))
            else:
                if e.op not in ('matmul','quantize','sum','concat'):
                    raise ValueError('no lowering for '+e.op)
                if e.op == 'matmul' and (t.rows%4 or t.cols%8 or args[0].cols%8):
                    raise ValueError('expression contains an incomplete 4x8x8 MMUL microtile')
                if e.op == 'quantize' and t.rows*t.cols%32:
                    raise ValueError('quantize lowering requires complete 32-element vectors')
                name = f't{len(ops)}'
                if t.rows%4 or t.cols%8:
                    raise ValueError('materialized operators require complete packed 4x8 tiles')
                result = View(name,t.rows,t.cols,t.cols,t.dtype,4)
                ops.append({'node':e,'args':args,'out':result,
                            'bytes':t.rows*t.cols*(4 if t.dtype=='int32' else 1)})
        cache[key] = result
        return result

    output = visit(expr)
    if output.dtype != 'int8' or output.rows%4 or output.cols%8:
        raise ValueError('packed output requires complete INT8 4x8 tiles')
    return {'operators':[{'kind':op['node'].op, 'origin':expression_id(op['node']),
        'relu':op['node'].relu, 'axis':op['node'].axis,
        'args':[asdict(a) for a in op['args']], 'out':asdict(op['out'])} for op in ops],
        'output':asdict(output)}


def _single_accumulator_helper(op):
    """Specialize only strides/shapes; retain loops and one live MMUL accumulator."""
    args=op['args'];o=op['out']
    key={'rows':o.rows,'cols':o.cols,'relu':op['relu'],'loop_order':op.get('loop_order','row_column'),
         'terms':[(a.cols,a.pitch,b.pitch) for a,b in zip(args[::2],args[1::2])]}
    name='qmm_'+sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:16]
    params=','.join(f'const I8* a{i},const I8* b{i}' for i in range(len(args)//2))
    row=f'for(unsigned rb=0;rb<{o.rows};rb+=4)';col=f'for(unsigned cb=0;cb<{o.cols};cb+=8)'
    outer=f'{col} {row}' if op.get('loop_order')=='column_row' else f'{row} {col}'
    lines=[f'__attribute__((noinline)) inline void {name}({params},I8* out) {{',
           ' '+outer+' {',
           '#ifdef AIE_OPT_HOST','  alignas(32) I32 acc[32];',
           '#else','  using Calc=aie::mmul<4,8,8,int8,int8,acc32>; Calc acc;','#endif']
    for i,(a,b) in enumerate(zip(args[::2],args[1::2])):
        def product(k,initialize):
            ap=f'a{i}+((rb/4)*{a.pitch//8}+({k})/8)*32'
            bp=f'b{i}+((({k})/8)*{b.pitch//8}+cb/8)*64'
            return ['#ifdef AIE_OPT_HOST',f'  multiply({ap},{bp},acc,{str(initialize).lower()});',
                    '#else',f'  acc.{"mul" if initialize else "mac"}(aie::load_v<32>({ap}),aie::load_v<64>({bp}));','#endif']
        if i==0:lines.extend(product('0',True))
        start=8 if i==0 else 0
        if a.cols>start:
            lines.append(f'  for(unsigned kb={start};kb<{a.cols};kb+=8) {{')
            lines.extend(product('kb',False));lines.append('  }')
    dst=f'out+((rb/4)*{o.cols//8}+cb/8)*32'
    lines+=['#ifdef AIE_OPT_HOST',f'  quantize(acc,{dst},{str(op["relu"]).lower()});','#else',
            '  auto value=acc.to_accum().template to_vector<int8>(0);']
    if op['relu']:lines.append('  value=aie::max(value,int8(0));')
    lines += [f'  aie::store_v({dst},value);','#endif',' }','}']
    pointers=[]
    for v in args:
        offset=((v.row//v.layout)*(v.pitch//8)+v.col//8)*(v.layout*8)
        pointers.append(v.base+(f'+{offset}' if offset else ''))
    return name,'\n'.join(lines),f'  {name}({",".join(pointers)},{o.base});'


def fused_helper(op):
    grid=op.get('accumulator_grid',[1,1])
    if grid==[1,1]:return _single_accumulator_helper(op)
    rows,cols=grid;args=op['args'];o=op['out']
    key={'rows':o.rows,'cols':o.cols,'relu':op['relu'],'grid':grid,'loop_order':op.get('loop_order','row_column'),
         'terms':[(a.cols,a.pitch,b.pitch) for a,b in zip(args[::2],args[1::2])]}
    name='qmm_'+sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:16]
    params=','.join(f'const I8* a{i},const I8* b{i}' for i in range(len(args)//2))
    chains=[(r,c) for r in range(rows) for c in range(cols)]
    row=f'for(unsigned rb=0;rb<{o.rows};rb+={4*rows})';col=f'for(unsigned cb=0;cb<{o.cols};cb+={8*cols})'
    outer=f'{col} {row}' if op.get('loop_order')=='column_row' else f'{row} {col}'
    lines=[f'__attribute__((noinline)) inline void {name}({params},I8* out) {{',
           ' '+outer+' {',
           '#ifdef AIE_OPT_HOST']
    lines += [f'  alignas(32) I32 acc{r}{c}[32];' for r,c in chains]
    lines += ['#else','  using Calc=aie::mmul<4,8,8,int8,int8,acc32>;']
    lines += [f'  Calc acc{r}{c};' for r,c in chains]+['#endif']
    for i,(a,b) in enumerate(zip(args[::2],args[1::2])):
        def product(k,initialize):
            ap=[f'a{i}+((rb/4+{r})*{a.pitch//8}+({k})/8)*32' for r in range(rows)]
            bp=[f'b{i}+((({k})/8)*{b.pitch//8}+cb/8+{c})*64' for c in range(cols)]
            code=['  {','#ifdef AIE_OPT_HOST']
            code += [f'  multiply({ap[r]},{bp[c]},acc{r}{c},{str(initialize).lower()});' for r,c in chains]
            code += ['#else']+[f'  auto av{r}=aie::load_v<32>({ap[r]});' for r in range(rows)]
            code += [f'  auto bv{c}=aie::load_v<64>({bp[c]});' for c in range(cols)]
            code += [f'  acc{r}{c}.{ "mul" if initialize else "mac"}(av{r},bv{c});' for r,c in chains]
            return code+['#endif','  }']
        if i==0:lines.extend(product('0',True))
        start=8 if i==0 else 0
        if a.cols>start:
            lines.append(f'  for(unsigned kb={start};kb<{a.cols};kb+=8) {{')
            lines.extend(product('kb',False));lines.append('  }')
    for r,c in chains:
        dst=f'out+((rb/4+{r})*{o.cols//8}+cb/8+{c})*32'
        lines += ['  {','#ifdef AIE_OPT_HOST',f'  quantize(acc{r}{c},{dst},{str(op["relu"]).lower()});',
                  '#else',f'  auto value=acc{r}{c}.to_accum().template to_vector<int8>(0);']
        if op['relu']:lines.append('  value=aie::max(value,int8(0));')
        lines += [f'  aie::store_v({dst},value);','#endif','  }']
    lines += [' }','}']
    pointers=[]
    for v in args:
        offset=((v.row//v.layout)*(v.pitch//8)+v.col//8)*(v.layout*8)
        pointers.append(v.base+(f'+{offset}' if offset else ''))
    return name,'\n'.join(lines),f'  {name}({",".join(pointers)},{o.base});'


def emit_region(dag, function, helpers):
    ops=[{**op,'args':[View(**a) for a in op['args']], 'out':View(**op['out'])}
         for op in dag['operators']]
    output=View(**dag['output'])
    for op in ops:
        o=op['out'];op['bytes']=o.rows*o.cols*(4 if o.dtype=='int32' else 1)
        if op.get('output_direct') or 'destination_offset' in op:op['bytes']=0
    last = {op['out'].base:i for i,op in enumerate(ops)}
    for i,op in enumerate(ops):
        for a in op['args']:
            if a.base in last:
                last[a.base] = max(last[a.base], i)
    if output.base in last:
        last[output.base] = len(ops)
    active = {}; peak = 0
    for i,op in enumerate(ops):
        active = {n:v for n,v in active.items() if last[n]>=i}
        if op.get('output_direct') or 'destination_offset' in op:
            op['offset']=None
            continue
        size = (op['bytes']+31)//32*32; offset = allocation_step(active,size)
        active[op['out'].base] = (offset,size)
        op['offset'] = offset; peak = max(peak,offset+size)
    lines = [f'inline void {function}(const I8* x,const I8* w0,const I8* w1,I8* y) {{',
             f'  alignas(32) static I8 arena[{max(32,peak)}];']
    macs = 0
    for op in ops:
        kind,a,o = op['kind'],op['args'],op['out']
        cpp = 'I32' if o.dtype=='int32' else 'I8'
        if 'destination_offset' in op:lines.append(f'  auto* {o.base}=y+{op["destination_offset"]};')
        elif op.get('output_direct'):lines.append(f'  auto* {o.base}=y;')
        else:lines.append(f'  auto* {o.base}=reinterpret_cast<{cpp}*>(arena+{op["offset"]});')
        if kind == 'matmul':
            macs += o.rows*a[0].cols*o.cols
            lines.append(f'  op_mm({a[0].descriptor()},{a[1].descriptor()},{o.base});')
        elif kind == 'quantize':
            # Gather allows quantization of sliced/concatenated regions without
            # assuming their parent storage has the same row pitch.
            lines.append(f'  op_q({a[0].descriptor()},{o.base},{str(op["relu"]).lower()});')
        elif kind == 'fused_qmm':
            macs+=sum(o.rows*v.cols*o.cols for v in a[::2])
            name,helper,call=fused_helper(op);helpers[name]=helper;lines.append(call)
        elif kind == 'sum':
            lines.append(f'  op_copy32({a[0].descriptor()},{o.base},{o.cols},0,0);')
            for v in a[1:]:lines.append(f'  op_add32({v.descriptor()},{o.base});')
        elif kind == 'concat':
            if op.get('concat_forwarded'):continue
            offset = 0
            for v in a:
                r = offset if op['axis']==0 else 0
                c = offset if op['axis']==1 else 0
                kind='32' if v.dtype=='int32' else '8'
                lines.append(f'  op_copy{kind}({v.descriptor()},{o.base},{o.cols},{r},{c});')
                offset += v.rows if op['axis']==0 else v.cols
    direct=any(op.get('output_direct') for op in ops)
    if not direct:lines.append(f'  op_pack8({output.descriptor()},y);')
    lines.append('}')
    return '\n'.join(lines), {'scheduled_operators':sum(not op.get('concat_forwarded') for op in ops),'scheduled_macs':macs,
        'arena_bytes':peak, 'temporary_microtile_bytes':224,
        'code_generation':'operator_dag_with_registered_execution_rules_v3',
        'fused_quantized_regions':sum(op['kind']=='fused_qmm' for op in ops),
        'direct_output':direct,
        'forwarded_output_slices':sum('destination_offset' in op for op in ops),
        'maximum_live_accumulator_microtiles':max([r*c for op in ops if op['kind']=='fused_qmm'
                                                  for r,c in [op.get('accumulator_grid',[1,1])]] or [1]),
        'allocation_scope':'serial lowered DAG scratch, excluding stack, compiler spills and ADF buffers',
        'storage':[{'node':op['out'].base,'offset':op['offset'],'bytes':op['bytes'],
                    'last_use':last[op['out'].base]} for op in ops]}


def lower_region(expr, spec, function):
    helpers={};code,report=emit_region(build_region_dag(expr,spec),function,helpers)
    return '\n'.join(helpers.values())+'\n'+code,report


def lower_program(expr, spec, groups, execution=None):
    code=[]; reports=[];helpers={}
    for g in range(groups):
        region=push_slice(expr,0,g*(spec.m//groups),(g+1)*(spec.m//groups),spec)
        dag=build_region_dag(region,spec) if execution is None else execution['regions'][g]
        source,report=emit_region(dag,f'run_group_{g}',helpers)
        code.append(source); reports.append(report)
    # Reuse only the already tested microtile primitive, not the mode templates.
    template=Path(__file__).with_name('compute.hpp.in').read_text(encoding='utf-8')
    primitives=template[template.index('inline void multiply'):template.index('inline void hidden_tile')]
    header='''#pragma once
#include <cstdint>
#ifdef AIE_OPT_HOST
#include <algorithm>
#else
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#endif
namespace opt {
using I8=int8_t; using I32=int32_t;
'''+f'constexpr unsigned M={spec.m},K={spec.k},H={spec.hidden},N={spec.n},GR={spec.m//groups};\n'
    dispatch='inline void execute(const I8* x,const I8* w0,const I8* w1,unsigned g,I8* y) {\n'
    dispatch+='\n'.join(f'  if(g=={g}) run_group_{g}(x,w0,w1,y);' for g in range(groups))+'\n}\n}\n'
    runtime=Path(__file__).with_name('operator_runtime.hpp.in').read_text(encoding='utf-8')
    return header+primitives+runtime+'\n'.join(helpers.values())+'\n'+'\n'.join(code)+'\n'+dispatch,reports
