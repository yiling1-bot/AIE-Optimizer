"""Prove that a complete imported function fits the existing two-layer backend.

This is a separate capability gate, not the frontend's accepted language.
All signature inputs/outputs must be accounted for. No subgraph is silently
substituted for the whole source function.
"""
from dataclasses import asdict
from ..contracts import make_contract, Target
from ..semantics import Spec, build, expression_id
from .ir import ImportFailure, fail


def admit_two_layer(program):
    values = {x['id']:x for x in program['inputs']+program['nodes']}
    inputs = {x['id'] for x in program['inputs']}
    def constant(value):
        node = values[value]
        if node.get('op') == 'constant':
            vals = node['attributes']['values']
            return vals[0] if len(set(vals)) == 1 else None
        if node.get('op') == 'splat': return constant(node['operands'][0])
        return None
    def clamp(value):
        node = values[value]
        if node.get('op') != 'truncate' or node['type']['bits'] != 8:
            fail('each layer must end in explicit clamp followed by i32-to-i8 truncation')
        base = values[node['operands'][0]]
        if base['type']['bits'] != 32: fail('quantization input must be i32')
        found = {}
        for _ in range(2):
            op = base.get('op')
            if op not in ('min_signed','max_signed') or op in found:
                fail('saturation bounds must be explicit min/max operations')
            a,b = base['operands']; ca,cb = constant(a),constant(b)
            if (ca is None) == (cb is None): fail('clamp needs one uniform constant bound')
            found[op] = ca if ca is not None else cb
            base = values[b if ca is not None else a]
        if found.get('min_signed') != 127 or found.get('max_signed') not in (-128,0):
            fail('backend supports only signed sat8, optionally followed by hidden ReLU')
        return base, found['max_signed'] == 0
    def matmul(node):
        if node.get('op') != 'matmul' or node['type']['bits'] != 32 or node['may_wrap']:
            fail('each layer requires a full non-overflowing signed i32 matrix product')
        a,b,init = node['operands']
        if constant(init) != 0: fail('nonzero accumulator/bias is not supported by current backend')
        def unwrap(value):
            old = values[value]
            if old.get('op') == 'extend_signed':
                value = old['operands'][0]; old = values[value]
            if old['type']['bits'] != 8: fail('matrix product operands must originate in i8 values')
            return value
        return unwrap(a),unwrap(b)
    try:
        if len(program['outputs']) != 1: fail('current backend requires one complete output')
        second, final_relu = clamp(program['outputs'][0]['value'])
        if final_relu: fail('final-output ReLU is not supported by current backend')
        h,w1 = matmul(second)
        first, hidden_relu = clamp(h)
        x,w0 = matmul(first)
        mapping = {'X':x,'W0':w0,'W1':w1}
        if len(set(mapping.values())) != 3 or set(mapping.values()) != inputs:
            fail('whole function must have exactly three distinct i8 tensor inputs')
        for value in mapping.values():
            if len(values[value]['type']['shape']) != 2: fail('backend requires rank-two inputs')
        m,k = values[x]['type']['shape']; k0,hid = values[w0]['type']['shape']; h1,n = values[w1]['type']['shape']
        if k != k0 or hid != h1: fail('layer shapes do not compose')
        spec = Spec(m,k,hid,n,hidden_relu); spec.validate()
        if m%4 or any(d%8 for d in (k,hid,n)):
            fail('current backend requires complete 4x8x8 microtiles')
        return {'status':'supported','backend':'two_layer_int8_adf_v1','spec':asdict(spec),
                'input_binding':mapping,'output_binding':{'Y':program['outputs'][0]['value']},
                'canonical_expression_id':expression_id(build(spec)['materialized_layers'].expression),
                'proof':'structural replay: zero-init signed GEMM, exact clamp, complete signature',
                'interface_adapter':{'source':'logical tensor indices','destination':'three full-frame packed PLIOs',
                    'packing':{'X':[4,8],'W0':[8,8],'W1':[8,8],'Y':[4,8]},
                    'source_signature_retained_in_import':True,'original_runtime_interface_preserved':False}}
    except (ImportFailure,ValueError) as exc:
        return {'status':'unsupported','backend':'two_layer_int8_adf_v1','reason':str(exc),
                'imported_program_preserved':True}


def contract_from_import(artifact, target=Target()):
    from .frontends import validate_import,source_kind
    program = validate_import(artifact); admission = admit_two_layer(program)
    if admission['status'] != 'supported': fail(admission['reason'], code='backend_unsupported')
    return make_contract(Spec(**admission['spec']),target,
        {'kind':source_kind(artifact),'import':artifact,'admission':admission})


def validate_contract_source(source, spec):
    from .frontends import validate_import,source_kind
    if set(source) != {'kind','import','admission'}: fail('unexpected MLIR contract source fields', code='invalid_import')
    program = validate_import(source['import']); admission = admit_two_layer(program)
    if source['kind']!=source_kind(source['import']) or admission['status'] != 'supported' or source['admission'] != admission or asdict(spec) != admission['spec']:
        fail('contract is not the admitted complete source function', code='invalid_import')
