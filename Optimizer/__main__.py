import argparse
import json
from pathlib import Path
from .contracts import make_contract, import_reviewed_k4, Target
from .planner import enumerate_plans, make_plan
from .backend import generate
from .runner import cpu_check, native_run, review_native
from .frontier import verified_frontier
from .search import campaign


def main(argv=None):
    p=argparse.ArgumentParser(description='Generate and verify variable-core AIE implementations')
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('contract');c.add_argument('--source',type=Path);c.add_argument('--max-cores',type=int,default=16);c.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('space');s.add_argument('contract',type=Path);s.add_argument('--output',type=Path,required=True)
    ir=sub.add_parser('ir-space');ir.add_argument('contract',type=Path);ir.add_argument('--output',type=Path,required=True)
    ir.add_argument('--max-candidates',type=int,default=24);ir.add_argument('--max-depth',type=int,default=2)
    ir.add_argument('--max-executions',type=int,default=8);ir.add_argument('--execution-depth',type=int,default=2)
    eg=sub.add_parser('egraph-space');eg.add_argument('contract',type=Path);eg.add_argument('--output',type=Path,required=True)
    eg.add_argument('--max-rounds',type=int,default=2);eg.add_argument('--max-terms',type=int,default=600)
    eg.add_argument('--max-applications',type=int,default=32);eg.add_argument('--max-states',type=int,default=10000)
    eg.add_argument('--keep',type=int,default=6);eg.add_argument('--max-executions',type=int,default=8)
    eg.add_argument('--execution-depth',type=int,default=2)
    ir=sub.add_parser('ir-generate');ir.add_argument('contract',type=Path);ir.add_argument('plan',type=Path)
    ir.add_argument('--output',type=Path,required=True);ir.add_argument('--frames',type=int,default=4)
    r=sub.add_parser('rules');r.add_argument('--output',type=Path,required=True)
    g=sub.add_parser('generate');g.add_argument('contract',type=Path);g.add_argument('--mode',required=True);g.add_argument('--groups',type=int,required=True);g.add_argument('--frames',type=int,default=4);g.add_argument('--output',type=Path,required=True)
    g.add_argument('--gather',choices=('flat','tree'),default='flat')
    c=sub.add_parser('cpu');c.add_argument('artifact',type=Path);c.add_argument('--compiler',default='g++')
    n=sub.add_parser('native');n.add_argument('artifact',type=Path);n.add_argument('--platform',required=True)
    r=sub.add_parser('review');r.add_argument('artifact',type=Path)
    s=sub.add_parser('search');s.add_argument('contract',type=Path);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--platform',required=True);s.add_argument('--max-evaluations',type=int,default=6);s.add_argument('--wall-seconds',type=float,default=1200)
    s.add_argument('--registered',action='store_true');s.add_argument('--max-candidates',type=int,default=24);s.add_argument('--max-depth',type=int,default=2)
    s.add_argument('--max-executions',type=int,default=8);s.add_argument('--execution-depth',type=int,default=2)
    s.add_argument('--egraph',action='store_true');s.add_argument('--max-states',type=int,default=10000)
    s.add_argument('--joint-evaluation',type=Path);s.add_argument('--allow-conditional-ranking',action='store_true')
    f=sub.add_parser('frontier');f.add_argument('results',type=Path,nargs='+');f.add_argument('--output',type=Path,required=True)
    cp=sub.add_parser('compiled-profile');cp.add_argument('artifact',type=Path);cp.add_argument('--output',type=Path,required=True)
    fp=sub.add_parser('function-profile');fp.add_argument('artifact',type=Path);fp.add_argument('simulation',type=Path);fp.add_argument('--output',type=Path,required=True)
    pi=sub.add_parser('phase-inventory');pi.add_argument('artifact',type=Path);pi.add_argument('--output',type=Path,required=True)
    pb=sub.add_parser('phase-binding');pb.add_argument('contract',type=Path);pb.add_argument('plan',type=Path)
    pb.add_argument('--software-assumptions',action='store_true',required=True);pb.add_argument('--frames',type=int,default=4);pb.add_argument('--slots',type=int,default=1);pb.add_argument('--output',type=Path,required=True)
    pe=sub.add_parser('phase-evaluate');pe.add_argument('contract',type=Path);pe.add_argument('plan',type=Path)
    pe.add_argument('binding',type=Path);pe.add_argument('registry',type=Path);pe.add_argument('selection',type=Path);pe.add_argument('--output',type=Path,required=True)
    js=sub.add_parser('joint-space');js.add_argument('contract',type=Path);js.add_argument('--evaluation',type=Path)
    js.add_argument('--max-rounds',type=int,default=2);js.add_argument('--max-terms',type=int,default=600)
    js.add_argument('--max-applications',type=int,default=32);js.add_argument('--max-states',type=int,default=10000)
    js.add_argument('--max-plans',type=int,default=256);js.add_argument('--max-executions',type=int,default=8)
    js.add_argument('--execution-depth',type=int,default=2);js.add_argument('--metric',choices=('first_output','batch_complete'),default='first_output')
    js.add_argument('--output',type=Path,required=True)
    im=sub.add_parser('import-mlir',help='Import a pure static integer tensor function from MLIR')
    im.add_argument('source',type=Path);im.add_argument('--entry');im.add_argument('--output',type=Path,required=True)
    cp=sub.add_parser('import-cpp',help='Import a closed compiler-typed dense-array C++ function')
    cp.add_argument('source',type=Path);cp.add_argument('--entry',required=True);cp.add_argument('--project-root',type=Path)
    cp.add_argument('--compile-db',type=Path);cp.add_argument('--command-index',type=int)
    cp.add_argument('--flag',action='append',default=[]);cp.add_argument('--output',type=Path,required=True)
    ic=sub.add_parser('import-contract',help='Admit a verified imported function to the existing backend')
    ic.add_argument('artifact',type=Path);ic.add_argument('--max-cores',type=int,default=16);ic.add_argument('--output',type=Path,required=True)
    ck=sub.add_parser('import-check',help='Compare source-region execution with imported tensor SSA')
    ck.add_argument('artifact',type=Path);ck.add_argument('--cases',type=int,default=8);ck.add_argument('--seed',type=int,default=20261018);ck.add_argument('--output',type=Path,required=True)
    ex=sub.add_parser('import-export',help='Round-trip the verified source MLIR, retaining its signature')
    ex.add_argument('artifact',type=Path);ex.add_argument('--output',type=Path,required=True)
    ac=sub.add_parser('import-artifact-check',help='Replay source MLIR on exactly the generated packed inputs')
    ac.add_argument('artifact',type=Path);ac.add_argument('--output',type=Path,required=True)
    na=sub.add_parser('native-ast-inventory',help='Audit target-compiler AST evidence without claiming semantic import')
    na.add_argument('capture',type=Path);na.add_argument('--source-artifact',type=Path,required=True);na.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    if args.command.startswith('import-'):
        from .importing.ir import ImportFailure
        from .importing.mlir_frontend import import_mlir,export_mlir
        from .importing.admission import contract_from_import
        from .importing.check import check_import,check_generated_import
        code=0
        try:
            if args.command=='import-mlir':result=import_mlir(args.source,args.entry)
            elif args.command=='import-cpp':
                from .importing.cpp_frontend import import_cpp
                result=import_cpp(args.source,args.entry,project_root=args.project_root,args=args.flag,
                                  database=args.compile_db,command_index=args.command_index)
            elif args.command=='import-contract':result=contract_from_import(read(args.artifact),Target(max_compute_tiles=args.max_cores))
            elif args.command=='import-check':result=check_import(read(args.artifact),args.cases,args.seed)
            elif args.command=='import-artifact-check':result=check_generated_import(args.artifact)
            else:
                imported=read(args.artifact)
                if imported.get('schema')=='aie-optimizer-cpp-import-v1':
                    from .importing.cpp_frontend import validate_import
                    from .importing.cpp_capture import materialize
                    validate_import(imported);args.output.mkdir(exist_ok=False,parents=True)
                    materialize(imported['source'],args.output.resolve())
                    print(json.dumps({'output':str(args.output.resolve()),'status':'exported_source_closure'}));return 0
                text=export_mlir(read(args.artifact))
                with args.output.open('x',encoding='utf-8') as handle:handle.write(text)
                print(json.dumps({'output':str(args.output.resolve()),'status':'exported'}));return 0
        except ImportFailure as exc:
            result={'schema':'aie-optimizer-import-diagnostic-v1','status':'rejected','diagnostic':exc.diagnostic()};code=2
        with args.output.open('x',encoding='utf-8') as handle:handle.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'output':str(args.output.resolve()),'status':result.get('status','created')}));return code
    if args.command=='native-ast-inventory':
        from .importing.native_ast import inspect_native_ast
        result=inspect_native_ast(args.capture,args.source_artifact)
    elif args.command=='contract':
        target=Target(max_compute_tiles=args.max_cores)
        result=import_reviewed_k4(args.source,target) if args.source else make_contract(target=target)
    elif args.command=='space':
        plans,refused=enumerate_plans(read(args.contract));result={'plans':plans,'refused':refused}
    elif args.command=='ir-space':
        from .ir_planner import enumerate_ir_plans
        plans,report=enumerate_ir_plans(read(args.contract),args.max_candidates,args.max_depth,
                                      max_executions=args.max_executions,execution_depth=args.execution_depth)
        result={'plans':plans,'report':report}
    elif args.command=='egraph-space':
        from .egraph_planner import enumerate_egraph_plans
        plans,report=enumerate_egraph_plans(read(args.contract),max_rounds=args.max_rounds,max_terms=args.max_terms,
            max_applications=args.max_applications,max_states=args.max_states,keep=args.keep,
            max_executions=args.max_executions,execution_depth=args.execution_depth)
        result={'plans':plans,'report':report}
    elif args.command=='rules':
        from .egraph_rules import default_equality_registry,default_patterns
        registry=default_equality_registry()
        from .mechanisms import catalog
        result={'semantic_rules':registry.catalog(),'execution_rules':registry.execution.catalog(),
                'egraph_pattern_rule_ids':list(default_patterns()),'literature_mechanisms':catalog()}
    elif args.command=='ir-generate':
        result=generate(read(args.contract),read(args.plan),args.output,frames=args.frames)
        print(json.dumps({'artifact':str(args.output.resolve()),'candidate_id':result['plan']['id']}));return 0
    elif args.command=='generate':
        contract=read(args.contract);plan=make_plan(contract,args.mode,args.groups,args.gather)
        result=generate(contract,plan,args.output,frames=args.frames)
        print(json.dumps({'artifact':str(args.output.resolve()),'candidate_id':plan['id'],
                          'planned_compute_tiles':plan['planned_compute_tiles']}));return 0
    elif args.command=='cpu':
        result=cpu_check(args.artifact,args.compiler);print(json.dumps(result));return result['status']!='passed'
    elif args.command=='native':
        result=native_run(args.artifact,args.platform);print(json.dumps(result));return result['status']!='verified'
    elif args.command=='review':
        print(json.dumps(review_native(args.artifact)));return 0
    elif args.command=='search':
        result=campaign(read(args.contract),args.output,args.platform,max_evaluations=args.max_evaluations,
                        wall_seconds=args.wall_seconds,on_progress=lambda r:print(json.dumps(r),flush=True),
                        registered=args.registered,max_candidates=args.max_candidates,max_depth=args.max_depth,
                        max_executions=args.max_executions,execution_depth=args.execution_depth,
                        egraph=args.egraph,max_states=args.max_states,
                        joint_evaluation=read(args.joint_evaluation) if args.joint_evaluation else None,
                        allow_conditional_ranking=args.allow_conditional_ranking)
        print(json.dumps({'stop_reason':result['stop_reason'],'evaluated':len(result['records']),
                          'output':str(args.output.resolve()/'campaign.json')}));return 0
    elif args.command=='compiled-profile':
        from .compiled_profile import read_compiled_profile
        result=read_compiled_profile(args.artifact)
    elif args.command=='function-profile':
        from .native_profile import audit_function_profiles
        result=audit_function_profiles(args.artifact,args.simulation)
    elif args.command=='phase-inventory':
        from .compiled_phase_inventory import inspect_phase_inventory
        result=inspect_phase_inventory(args.artifact)
    elif args.command=='phase-binding':
        from .phase_model import software_binding
        result=software_binding(read(args.contract),read(args.plan),frames=args.frames,slots=args.slots)
    elif args.command=='phase-evaluate':
        from .phase_costs import evaluate_phases
        result=evaluate_phases(read(args.contract),read(args.plan),read(args.binding),read(args.registry),read(args.selection))
    elif args.command=='joint-space':
        from .joint_extract import enumerate_joint_plans
        from .phase_costs import PhaseEvaluator
        contract=read(args.contract);evaluator=None
        if args.evaluation:
            config=read(args.evaluation)
            if set(config)!={'registry','selection','bindings'}:raise ValueError('joint evaluation requires registry, selection and plan-specific bindings')
            evaluator=PhaseEvaluator(contract,config['registry'],config['selection'],bindings=config['bindings'])
        plans,report=enumerate_joint_plans(contract,evaluator=evaluator,max_rounds=args.max_rounds,max_terms=args.max_terms,
            max_applications=args.max_applications,max_states=args.max_states,max_plans=args.max_plans,
            max_executions=args.max_executions,execution_depth=args.execution_depth,metric=args.metric)
        result={'plans':plans,'report':report}
    else: result=verified_frontier([read(x) for x in args.results])
    with args.output.open('x',encoding='utf-8') as handle: handle.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output.resolve())}));return 0


if __name__=='__main__':raise SystemExit(main())
