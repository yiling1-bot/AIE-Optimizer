"""Reproducible CPU/semantic coverage witnesses; never native speed evidence."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Optimizer.contracts import Target
from Optimizer.importing.ir import GraphBuilder, tensor_type
from Optimizer.importing.mlir_frontend import import_mlir
from Optimizer.importing.cpp_frontend import import_cpp
from Optimizer.program_ir import make_program, program_from_builder, program_from_import
from Optimizer.program_search import enumerate_egraph_plans
from Optimizer.program_egraph import check_extraction, equality_identity
from Optimizer.program_backend import generate, cpu_check
from Optimizer.program_planner import implementation_identity


def matrix_chain():
    g = GraphBuilder(); typ = lambda s:tensor_type(s,16)
    a=g.input('a',typ([8,2]),0); b=g.input('b',typ([2,8]),1); c=g.input('c',typ([8,2]),2)
    z=g.add('constant',[],typ([8,8]),{'values':[0]}, {})
    ab=g.add('matmul',[a,b,z],typ([8,8]),{}, {})
    zz=g.add('constant',[],typ([8,2]),{'values':[0]}, {})
    return program_from_builder(g,[g.add('matmul',[ab,c,zz],typ([8,2]),{}, {})])


def shared_factor():
    g=GraphBuilder(); typ=lambda s:tensor_type(s,16)
    a=g.input('a',typ([2,3]),0); b=g.input('b',typ([3,2]),1); c=g.input('c',typ([3,2]),2)
    z=g.add('constant',[],typ([2,2]),{'values':[7]}, {})
    zz=g.add('constant',[],typ([2,2]),{'values':[-3]}, {})
    ab=g.add('matmul',[a,b,z],typ([2,2]),{}, {})
    ac=g.add('matmul',[a,c,zz],typ([2,2]),{}, {})
    y=g.add('add',[ab,ac],typ([2,2]),{}, {})
    return program_from_builder(g,[y,ab])


def facts(plan):
    p=plan['program']; vals={v['id']:v for v in p['inputs']+p['nodes']}
    return {'operators':dict(Counter(n['op'] for n in p['nodes'])),
        'integer_macs_not_time':sum(n['type']['shape'][0]*n['type']['shape'][1]*
            vals[n['operands'][0]]['type']['shape'][1] for n in p['nodes'] if n['op']=='matmul'),
        'planned_cores':plan['planned_compute_tiles'],'outputs':len(p['outputs'])}


def run(destination, frames=4):
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False)
    temporary=out/'temporary';temporary.mkdir();tempfile.tempdir=str(temporary)
    target=Target(max_compute_tiles=3)
    fixtures=ROOT/'tests/fixtures/optimizer_program'
    cases=[]
    for name,path in [('three_layer',fixtures/'three_layer.mlir'),('fir',fixtures/'fir.mlir'),
                      ('branched',ROOT/'tests/fixtures/optimizer_import/branched.mlir')]:
        cases.append((name,program_from_import(import_mlir(path),target)))
    cpp=fixtures/'residual.cpp'
    cases.append(('cpp_residual',program_from_import(import_cpp(cpp,'residual',project_root=cpp.parent),target)))
    cases.extend([('matrix_chain',make_program(matrix_chain(),target)),('shared_factor',make_program(shared_factor(),target))])
    records=[]
    def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    for name,contract in cases:
        case=out/name;case.mkdir();save(case/'original-contract.json',contract)
        entries,report=enumerate_egraph_plans(contract,max_rounds=1,max_applications=8,max_terms=400,
            max_states=20000,max_dags=8,max_plans=24,max_plans_per_dag=8,max_implementations=2)
        save(case/'space.json',{'schema':'aie-optimizer-certified-program-space-v1','candidates':entries,'report':report})
        picks=[('reference',entries[0])]
        original_ops=facts(entries[0]['plan'])['operators']
        structural=next((e for e in entries if facts(e['plan'])['operators']!=original_ops),None)
        if structural:picks.append(('structural',structural))
        # Exercise algebraic families when this source admits them. This is a
        # coverage selector, not a time ranking or an application dispatch rule.
        for rid in ('tensor.matmul.associate','tensor.matmul.factor_left'):
            dags={s['dag_id'] for s in report['diversity_seeds'] if s['rule_id']==rid and s['admitted']}
            candidate=next((e for e in entries if e['dag_id'] in dags and
                all(e['plan']['id']!=p['plan']['id'] for _,p in picks)),None)
            if candidate:picks.append((rid.rsplit('.',1)[-1],candidate))
        multi=next((e for e in entries if e['plan']['planned_compute_tiles']>1 and
            all(e['plan']['id']!=p['plan']['id'] for _,p in picks)),None)
        if multi:picks.append(('multi_core',multi))
        checks=[]
        for label,entry in picks:
            source=entry['contract']['source'];certificate=source.get('certificate')
            if certificate:
                if check_extraction(certificate,contract['program'])!=entry['contract']['program']:
                    raise ValueError('certificate replay differs')
            root=case/label;manifest=generate(entry['contract'],entry['plan'],root,frames=frames)
            result=cpu_check(root)
            if result['status']!='passed':raise ValueError('generated CPU check failed')
            checks.append({'label':label,'plan_id':entry['plan']['id'],'facts':facts(entry['plan']),
                'certificate_id':certificate['id'] if certificate else None,
                'source_digest':manifest['source_digest'],'cpu':result})
            print(json.dumps({'application':name,'label':label,'status':result['status']}),flush=True)
        records.append({'application':name,'original_contract_id':contract['id'],'candidates':len(entries),
            'dag_assignments':report['retained_dag_assignments'],
            'unknown_candidates':len(report['unknown_plan_ids']),'checks':checks})
    result={'schema':'aie-optimizer-program-egraph-witness-v1','status':'passed',
        'implementation':implementation_identity(),'equality_implementation':equality_identity(),
        'applications':records,'generated_projects':sum(len(r['checks']) for r in records),
        'output_checks':sum(c['cpu']['outputs_checked'] for r in records for c in r['checks']),
        'source_oracle_projects':sum(c['cpu']['original_source_checked'] for r in records for c in r['checks']),
        'frames':frames,'new_native_compilations':0,'new_Simulator_runs':0,'new_cost_fits':0,
        'native_performance_claim':False,'scope':'synthetic and imported-source CPU semantic coverage; no device timing or ranking proof'}
    save(out/'report.json',result)
    files={p.relative_to(out).as_posix():sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob('*')) if p.is_file() and 'temporary' not in p.relative_to(out).parts}
    save(out/'files.json',files)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=4);args=parser.parse_args()
    result=run(args.output,args.frames)
    print(json.dumps({k:v for k,v in result.items() if k!='applications'}))
