"""Private isolated worker; the selected external Model package owns imports."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Optimizer.compiled_analyzer import (validate_snapshot, validate_candidate, strict_json,
    read_ref, save_new, identity)


def comparison_facts(candidate, request, raw, parse_commands):
    if (request['mode'] != 'predict' or raw.get('graph',{}).get('architecture') != 'AIE1' or
            raw.get('status') not in ('reused','conditional_prediction')): return {}
    sources = {}
    for name,ref in sorted(request['source_csv'].items()):
        data = read_ref(ref)
        if candidate['comparison_policy'] == 'exact_inputs':
            sources[name] = {'bytes':len(data),'sha256':sha256(data).hexdigest()}; continue
        parsed = parse_commands(data.decode('utf-8-sig'),stall_extra_ticks=0)
        if any(c['kind'] != 'word' for c in parsed['commands']):
            raise ValueError('dense packet interleaving policy refuses STALL commands')
        packets, current = {}, []
        for command in parsed['commands']:
            if command['header'] and current: raise ValueError('nested packet header')
            current.append((command['word32'],command['last']))
            if command['last']:
                packets.setdefault(str(command['packet_id']),[]).append(identity(current)); current=[]
        if current: raise ValueError('partial packet in comparison contract')
        sources[name] = packets
    facts = {'compile_identity':raw.get('graph',{}).get('compile_identity'),
        'binding_sha256':request['program_binding']['sha256'],
        'policy':candidate['comparison_policy'],'source_sequence_fingerprint':identity(sources),
        'scope':'same compiled program; dense policy preserves packet order within each source/destination'}
    return {**facts,'comparison_key':identity(facts),
            'scenario_id':identity({'registry_sha256':request['cost_registry']['sha256'],
                                    'model_selection':request['model_selection']})}


def main():
    job = strict_json(Path(sys.argv[1]).read_bytes()); snapshot = validate_snapshot(job['snapshot'])
    root = Path(snapshot['root']).resolve(); output = Path(job['output']).resolve()
    sys.path.insert(0,str(root))
    from Model.Timing import generic_analyzer as api
    from Model.Timing.cascade_backpressure_transport import parse_commands
    if not Path(api.__file__).resolve().is_relative_to(root): raise ValueError('external Analyzer import was shadowed')
    if job.get('operation') == 'prepare':
        from Model.Timing.compiled_program_extraction import extract_compiled_program
        from Model.Timing.compiled_request_preparation import prepare_request
        extraction = strict_json(read_ref(job['extraction_ref']))
        template = strict_json(read_ref(job['preparation_ref']))
        if template.get('metadata_ref') is not None or template['compiler_inputs'] != extraction['compiler_inputs']:
            raise ValueError('unbound metadata and identical compiler facts required')
        extracted = extract_compiled_program(extraction,output/'extracted')
        prepared_spec = dict(template,metadata_ref=extracted['metadata_ref'])
        save_new(output/'preparation-specification.json',prepared_spec)
        prepared_result = prepare_request(prepared_spec,output/'prepared')
        validate_snapshot(snapshot);read_ref(job['extraction_ref']);read_ref(job['preparation_ref'])
        save_new(output/'preparation.json',{'extracted':extracted,'prepared':prepared_result})
        return 0
    prepared, memo = {}, {}
    for i,candidate in enumerate(job['candidates']):
        started = time.perf_counter()
        try:
            validate_snapshot(snapshot); validate_candidate(candidate)
            request = strict_json(read_ref(candidate['request_ref']))
            key = identity(request); reused = key in memo; hot = False
            if reused:
                record = memo[key]; raw = strict_json(read_ref(record['result_ref']))
                for ref in raw.get('input_refs',[]): read_ref(ref)
                result_ref = record['result_ref']
            else:
                base = identity({k:request[k] for k in ('schema','compiler_inputs','program_binding','cost_registry')})
                if job['reuse_prepared'] and base in prepared:
                    raw = prepared[base].analyze(request); hot = True
                else:
                    raw = api.analyze_request(request)
                    # Prepare only when a later distinct query can reuse this context.
                    later = job['candidates'][i+1:]
                    if job['reuse_prepared'] and raw.get('status') in ('reused','conditional_prediction') and later:
                        if len(prepared) >= 2: prepared.pop(next(iter(prepared)))
                        prepared[base] = api.PreparedAnalyzer(request)
                validate_snapshot(snapshot); read_ref(candidate['request_ref'])
                result_ref = save_new(output/f'raw-{i:03d}.json',raw,compact=True)
                memo[key] = {'result_ref':result_ref}
            facts = comparison_facts(candidate,request,raw,parse_commands)
            for ref in raw.get('input_refs',[]):read_ref(ref)
            record = {'status':'completed','candidate_id':candidate['id'],'result_ref':result_ref,
                      'comparison_facts':facts,'seconds':time.perf_counter()-started,
                      'query_reused':reused,'prepared_reused':hot}
        except (ValueError,TypeError,KeyError,OSError,IndexError) as error:
            record = {'status':'invalid_input','candidate_id':candidate['id'],'reason':str(error),
                      'seconds':time.perf_counter()-started}
        save_new(output/f'query-{i:03d}.json',record)
        print(json.dumps({'candidate_id':candidate['id'],'status':record['status']}),flush=True)
    validate_snapshot(snapshot)
    return 0


if __name__ == '__main__': raise SystemExit(main())
