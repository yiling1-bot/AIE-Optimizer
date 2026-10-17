"""Emit fresh, standalone ADF programs from supported composed plans."""
from hashlib import sha256
import json
from pathlib import Path
from .contracts import identity, validate_contract
from .planner import validate_plan, analyzer_request, gather_regions
from .fixtures import make_frames


def generate(contract,plan,destination,*,frames=4,seed=20261014,registry=None):
    validate_plan(contract,plan,registry); spec,target=validate_contract(contract)
    is_ir=plan['schema'] in ('aie-optimizer-ir-plan-v2','aie-optimizer-egraph-plan-v3')
    dest=Path(destination)
    if dest.exists(): raise FileExistsError('generation requires a fresh destination')
    dest.mkdir(parents=True); (dest/'data').mkdir()
    fixtures=make_frames(spec,frames,seed)
    def write(name,value): (dest/name).write_text(value,encoding='utf-8',newline='\n')
    template=Path(__file__).with_name('compute.hpp.in').read_text(encoding='utf-8')
    for key,val in {'M':spec.m,'K':spec.k,'H':spec.hidden,'N':spec.n,
                    'GR':plan['group_rows'],'GROUPS':plan['row_groups'],
                    'RELU':str(spec.hidden_relu).lower()}.items():
        template=template.replace('@'+key+'@',str(val))
    if is_ir:
        from .ir_backend import lower_program
        from .registry import decode_expr
        template,lowering=lower_program(decode_expr(plan['expression']),spec,plan['row_groups'],plan['execution'])
        write('lowering.json',json.dumps(lowering,indent=2))
        write('execution.json',json.dumps(plan['execution'],indent=2))
    write('compute.hpp',template)
    for name in ('X','W0','W1'):
        values=sum((f[name] for f in fixtures),[])
        write('data/'+name+'.txt',''.join(' '.join(map(str,values[i:i+16]))+'\n'
                                        for i in range(0,len(values),16)))
    write('golden.json',json.dumps([f['Y'] for f in fixtures]))
    fixture_header='#pragma once\n#include <cstdint>\n'
    for name in ('X','W0','W1','Y'):
        fixture_header+=f'alignas(32) const int8_t fixture_{name}[{frames}][{len(fixtures[0][name])}]={{'
        fixture_header+=','.join('{'+','.join(map(str,f[name]))+'}' for f in fixtures)+'};\n'
    write('fixture.hpp',fixture_header)
    mode='execute' if is_ir else plan['mode']; groups=plan['row_groups']; gr=plan['group_rows']
    gather_children,gather_ranges=gather_regions(groups,plan.get('gather','flat'))
    if mode=='materialized':
        computation='alignas(32) int8_t h[opt::GR*opt::H]; opt::layer0(x,w0,g,h); opt::layer1(h,w1,y);'
    else: computation=f'opt::{mode}(x,w0,w1,g,y);'
    write('cpu_check.cpp',f'''#define AIE_OPT_HOST
#include "compute.hpp"
#include "fixture.hpp"
#include <iostream>
int main() {{
  unsigned mismatches=0;
  for(unsigned f=0;f<{frames};++f) for(unsigned g=0;g<{groups};++g) {{
    const auto *x=fixture_X[f], *w0=fixture_W0[f], *w1=fixture_W1[f];
    alignas(32) int8_t y[opt::GR*opt::N];
    {computation}
    for(unsigned i=0;i<opt::GR*opt::N;++i) mismatches+=(y[i]!=fixture_Y[f][g*opt::GR*opt::N+i]);
  }}
  std::cout<<mismatches<<"\\n"; return mismatches?1:0;
}}
''')
    signatures=[]; bodies=[]; nodes=[]; connections=[]
    def node(name,signature,body,inputs,output_size,tile):
        idx=len(nodes); signatures.append(signature+';')
        bodies.append(signature+' {\n aie::set_saturation(aie::saturation_mode::saturate);\n'+body+'\n}\n')
        nodes.append((name,idx,tile))
        for port,(source,size) in enumerate(inputs):
            connections.append(f'adf::connect<>({source},k[{idx}].in[{port}]); adf::dimensions(k[{idx}].in[{port}])={{{size}}};')
        connections.append(f'adf::dimensions(k[{idx}].out[0])={{{output_size}}};')
        return idx
    producers=[]
    if is_ir:
        from .execution_ir import emit_kernel
        indices={}
        for kernel in plan['execution']['kernels']:
            signature,body=emit_kernel(kernel,spec,groups)
            inputs=[]
            for port in kernel['inputs']:
                kind,source=port['source'].split(':',1)
                origin={'X':'x.out[0]','W0':'w0.out[0]','W1':'w1.out[0]'}[source] if kind=='input' else f'k[{indices[source]}].out[0]'
                inputs.append((origin,port['size']))
            indices[kernel['id']]=node(kernel['id'],signature,body,inputs,kernel['output_size'],kernel['tile'])
        producers=[indices[plan['execution']['output_kernel']]]
    for g in range(0 if is_ir else groups):
        if mode=='materialized':
            i=node(f'l0_{g}',f'void l0_{g}(In& x,In& w,Out& h)',
                   f' opt::layer0(x.data(),w.data(),{g},h.data());',
                   [('x.out[0]',spec.m*spec.k),('w0.out[0]',spec.k*spec.hidden)],gr*spec.hidden,(g,2))
            inp=[(f'k[{i}].out[0]',gr*spec.hidden),('w1.out[0]',spec.hidden*spec.n)]
            sig=f'void l1_{g}(In& h,In& w'
            body='opt::layer1(h.data(),w.data(),own);'
            name=f'l1_{g}'; tile=(g,3)
        else:
            inp=[('x.out[0]',spec.m*spec.k),('w0.out[0]',spec.k*spec.hidden),('w1.out[0]',spec.hidden*spec.n)]
            sig=f'void fused_{g}(In& x,In& w0,In& w1'
            body=f'opt::{mode}(x.data(),w0.data(),w1.data(),{g},own);'
            if is_ir: body=f'opt::run_group_{g}(x.data(),w0.data(),w1.data(),own);'
            name=f'fused_{g}'; tile=(g,2)
        children=gather_children[g];outbegin,outend=gather_ranges[g]
        if children or g==groups-1:
            for other,begin,end in children:
                idx=producers[other]
                sig+=f',In& y{other}'; inp.append((f'k[{idx}].out[0]',(end-begin)*gr*spec.n))
            sig+=',Out& y)'
            outsize=(outend-outbegin)*gr*spec.n
            body='alignas(32) int8 own[opt::GR*opt::N];\n '+body
            for other,begin,end in children:
                body+=f'\n for(unsigned i=0;i<{end-begin}*opt::GR*opt::N;i+=32) aie::store_v(y.data()+{begin-outbegin}*opt::GR*opt::N+i,aie::load_v<32>(y{other}.data()+i));'
            body+=f'\n for(unsigned i=0;i<opt::GR*opt::N;i+=32) aie::store_v(y.data()+{g-outbegin}*opt::GR*opt::N+i,aie::load_v<32>(own+i));'
        else:
            sig+=',Out& y)'; outsize=gr*spec.n
            body='auto* own=y.data();\n '+body
        idx=node(name,sig,body,inp,outsize,tile); producers.append(idx)
    header='#pragma once\n#include <adf.h>\nusing In=adf::input_buffer<int8>;\nusing AsyncIn=adf::input_async_buffer<int8>;\nusing Out=adf::output_buffer<int8>;\n'
    write('kernels.hpp',header+'\n'.join(signatures)+'\n')
    write('kernels.cpp','#include "kernels.hpp"\n#include "compute.hpp"\n'+''.join(bodies))
    graph=['#pragma once\n#include <adf.h>\n#include "kernels.hpp"\nclass OptGraph:public adf::graph {\npublic:\n',
           f'adf::kernel k[{len(nodes)}]; adf::input_plio x,w0,w1; adf::output_plio y;\n OptGraph() {{\n']
    for var,name in [('x','X'),('w0','W0'),('w1','W1')]:
        graph.append(f'{var}=adf::input_plio::create("{name}",adf::plio_128_bits,"data/{name}.txt");\n')
    graph.append('y=adf::output_plio::create("Y",adf::plio_128_bits,"out_y.txt");\n')
    for name,i,tile in nodes:
        graph.append(f'k[{i}]=adf::kernel::create({name}); adf::source(k[{i}])="kernels.cpp"; '
                     f'adf::runtime<adf::ratio>(k[{i}])=1.0; adf::stack_size(k[{i}])=8192; '
                     f'adf::location<adf::kernel>(k[{i}])=adf::tile({tile[0]},{tile[1]});\n')
        if is_ir:
            graph.append(f'adf::heap_size(k[{i}])={plan["resources"]["kernel_heap_bytes"][i]};\n')
    graph.extend(c+'\n' for c in connections)
    graph.append(f'adf::connect<>(k[{producers[-1]}].out[0],y.in[0]);\n }}\n}};\n')
    write('graph.hpp',''.join(graph))
    write('TopGraph.cpp',f'#include "graph.hpp"\nOptGraph graph;\nint main() {{graph.init();graph.run({frames});graph.end();return 0;}}\n')
    write('analyzer_request.json',json.dumps(analyzer_request(contract,plan,registry),indent=2))
    if contract['source'].get('kind') in ('verified_mlir_tensor_import_v1','verified_cpp_tensor_import_v1'):
        imported=contract['source']['import']
        if imported['schema']=='aie-optimizer-mlir-import-v1':
            write('imported_source.mlir',imported['source']['text'])
        else:
            from .importing.cpp_capture import materialize
            materialize(imported['source'],(dest/'source_snapshot').resolve())
        write('source_import.json',json.dumps(imported,indent=2)+'\n')
        write('interface_adapter.json',json.dumps(contract['source']['admission'],indent=2)+'\n')
    hashes={str(p.relative_to(dest)).replace('\\','/'):sha256(p.read_bytes()).hexdigest()
            for p in sorted(dest.rglob('*')) if p.is_file()}
    manifest={'schema':'aie-optimizer-artifact-v1','contract':contract,'plan':plan,
              'frames':frames,'seed':seed,'files':hashes,'source_digest':identity(hashes),
              'kernel_bindings':[{'function':n,'index':i,'tile':list(t)} for n,i,t in nodes],
              'numerical':'unverified','native':'unverified'}
    generator_files=('backend.py','compute.hpp.in','contracts.py','planner.py','semantics.py','fixtures.py')
    if is_ir: generator_files+=('registry.py','builtin_rules.py','ir_backend.py','ir_planner.py','operator_runtime.hpp.in',
                              'execution_ir.py','execution_rules.py','scratch_schedule.py','stage_costs.py')
    if plan['schema']=='aie-optimizer-egraph-plan-v3':
        generator_files+=('egraph.py','egraph_rules.py','global_extract.py','egraph_planner.py','access.py','mechanisms.py','cost_refinement.py')
    if contract['source'].get('kind')=='verified_mlir_tensor_import_v1':
        generator_files+=('importing/mlir_frontend.py','importing/mlir_compat.py','importing/ir.py','importing/admission.py','importing/frontends.py')
    if contract['source'].get('kind')=='verified_cpp_tensor_import_v1':
        generator_files+=('importing/cpp_frontend.py','importing/cpp_capture.py','importing/cpp_lift.py','importing/ir.py','importing/admission.py','importing/frontends.py')
    manifest['generator_files']={n:sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
        for n in generator_files}
    write('manifest.json',json.dumps(manifest,indent=2)+'\n')
    return manifest
