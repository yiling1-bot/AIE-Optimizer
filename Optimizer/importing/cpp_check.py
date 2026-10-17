"""Compile the captured original C++ and compare exact values with tensor SSA.

Uses a separate host compiler, never the lifted algorithm as its source oracle.
The generated harness allocates distinct arrays satisfying the restrict domain.
"""
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

from ..contracts import identity
from .cpp_capture import materialize
from .cpp_frontend import validate_import
from .ir import bounds,evaluate,fail


def original_cpp_outputs(artifact,batches,compiler='g++'):
    program=validate_import(artifact)
    if not batches or len(batches)>64:fail('C++ check batches must be 1..64')
    if sum(prod(a['type']['shape']) for a in program['inputs'])*len(batches)>300000:
        fail('C++ check literal input budget exceeded',code='resource_limit')
    # Validate shape and domains before compiling any source.
    expected=[evaluate(program,inputs) for inputs in batches]
    executable=shutil.which(compiler)
    if not executable:fail('host C++ compiler unavailable',code='missing_dependency')
    parameters=artifact['source_interface']['parameters']
    with tempfile.TemporaryDirectory(prefix='aie-cpp-check-') as directory:
        root=Path(directory).resolve();main=materialize(artifact['source'],root)
        if any(c in artifact['source']['main'] for c in ('"','\n','\r','\\')):
            fail('source path cannot be represented as a portable include',code='source_closure')
        lines=['#include <iostream>',f'#include "{artifact["source"]["main"]}"','int main() {']
        for batch in batches:
            lines.append('{');actual=[]
            for param in parameters:
                pos=param['position'];typ=param['tensor_type'];m,n=typ['shape'];bits=typ['bits']
                # Preserve the original leaf spelling to avoid int/long ABI ambiguity.
                entry=next(f for f in artifact['compiler_ast']['functions'] if f['id']==artifact['compiler_ast']['entry_id'])
                leaf=entry['parameters'][pos]['type']['pointee']
                while leaf['kind']=='CONSTANTARRAY':leaf=leaf['element']
                scalar=leaf['canonical'].replace('const ','')
                data=batch[param['input_id']] if param['direction']=='input' else [0]*(m*n)
                rows=['{'+','.join(str(v) for v in data[i:i+n])+'}' for i in range(0,m*n,n)]
                lines.append(f'{scalar} argument_{pos}[{m}][{n}]={{'+','.join(rows)+'};')
                lines.append(f'static_assert(sizeof(argument_{pos}[0][0])*8=={bits},"compiler integer width mismatch");')
                actual.append(f'argument_{pos}')
            lines.append(artifact['program']['entry']+'('+','.join(actual)+');')
            for param in parameters:
                if param['direction']=='output':
                    m,n=param['tensor_type']['shape'];pos=param['position']
                    lines.append(f'for(int i=0;i<{m};++i) for(int j=0;j<{n};++j) std::cout << (long long)argument_{pos}[i][j] << " ";')
            lines.append('std::cout << "\\n"; }')
        lines.append('return 0; }')
        harness='\n'.join(lines)+'\n';(root/'import_check.cpp').write_text(harness,encoding='utf-8',newline='\n')
        command=[executable,*artifact['source']['command']['effective_flags'],'import_check.cpp','-o','import_check.exe']
        try:
            build=subprocess.run(command,cwd=root,text=True,capture_output=True,timeout=60)
            if build.returncode:fail(build.stderr[-6000:],code='cpp_check_compile')
            run=subprocess.run([str(root/'import_check.exe')],cwd=root,text=True,capture_output=True,timeout=30)
            if run.returncode:fail(run.stderr[-2000:],code='cpp_check_run')
        except subprocess.TimeoutExpired as exc:fail(str(exc),code='cpp_check_timeout')
        try:flat=[int(v) for v in run.stdout.split()]
        except ValueError:fail('non-integer output from original function',code='cpp_check_output')
        total=sum(len(o) for batch in expected for o in batch)
        if len(flat)!=total:fail('original function output count mismatch',code='cpp_check_output')
        outputs=[];start=0
        for batch in expected:
            current=[]
            for out in batch:current.append(flat[start:start+len(out)]);start+=len(out)
            outputs.append(current)
        version=subprocess.run([executable,'--version'],capture_output=True,text=True,timeout=10).stdout.splitlines()[0]
        evidence={'compiler':version,'compiler_path':executable,'effective_source_flags':artifact['source']['command']['effective_flags'],
                  'harness_sha256':sha256(harness.encode()).hexdigest(),'stdout_sha256':sha256(run.stdout.encode()).hexdigest(),
                  'compile_exit':build.returncode,'run_exit':run.returncode,'compiler_diagnostics':build.stderr[-6000:],
                  'scope':'independent host compiler, distinct arrays; no AIE execution/latency claim'}
        return outputs,evidence


def check_cpp_import(artifact,cases=8,seed=20261018,compiler='g++'):
    program=validate_import(artifact)
    if type(cases) is not int or not 1<=cases<=64:fail('check cases must be 1..64')
    rng=random.Random(seed);batches=[]
    for case in range(cases):
        inputs={}
        for arg in program['inputs']:
            lo,hi=bounds(arg['type']['bits']);pool=[0] if case==0 else [lo,hi,-1,0,1] if case%2 else [-3,-1,0,1,2,4]
            inputs[arg['id']]=[rng.choice(pool) for _ in range(prod(arg['type']['shape']))]
        batches.append(inputs)
    actual,evidence=original_cpp_outputs(artifact,batches,compiler)
    expected=[evaluate(program,b) for b in batches]
    if actual!=expected:fail('original compiled C++ differs from imported tensor semantics',code='semantic_mismatch')
    return {'schema':'aie-optimizer-cpp-import-check-v1','import_id':artifact['id'],'status':'passed','cases':cases,'seed':seed,
        'values_checked':sum(len(o) for b in actual for o in b),'input_identity':identity(batches),'output_identity':identity(actual),
        'checker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'original_execution':evidence}
