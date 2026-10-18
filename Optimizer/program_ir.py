"""Whole-function tensor SSA contracts, independent of layer count and names.

The frontend IR is the semantic authority. This path never projects a function
onto Spec(m,k,hidden,n); the complete signature and every live output survive.
"""
from copy import deepcopy
from dataclasses import asdict
from math import prod

from .contracts import Target, identity
from .importing.ir import GraphBuilder, tensor_type


OPERATORS = {
    'constant': (0, {'values'}), 'empty': (0, set()),
    'splat': (1, set()), 'extend_signed': (1, set()), 'truncate': (1, set()),
    'add': (2, set()), 'mul': (2, set()),
    'min_signed': (2, set()), 'max_signed': (2, set()),
    'matmul': (3, {'promotion', 'accumulation'}), 'transpose': (1, {'permutation'}),
    'slice': (1, {'offsets', 'strides'}), 'concat': (None, {'axis'}),
}


def seal(value):
    return {**value, 'id': identity(value)}


def program_from_builder(builder, outputs, entry='declared'):
    return seal({'schema': 'aie-optimizer-tensor-ssa-v1', 'entry': entry,
        'inputs': deepcopy(builder.inputs), 'nodes': deepcopy(builder.nodes),
        'outputs': [{'position': i, 'value': v, 'type': builder.values[v]['type']}
                    for i, v in enumerate(outputs)],
        'effects': 'pure_tensor_function', 'physical_mapping': None})


def validate_program(program):
    """Replay types, initialization, integer ranges and SSA edges, not just hash."""
    if set(program) != {'schema', 'entry', 'inputs', 'nodes', 'outputs', 'effects', 'physical_mapping', 'id'}:
        raise ValueError('exact whole-program SSA fields required')
    if (program['schema'] != 'aie-optimizer-tensor-ssa-v1' or
            program['effects'] != 'pure_tensor_function' or program['physical_mapping'] is not None or
            not isinstance(program['entry'], str) or not program['entry'] or not program['outputs']):
        raise ValueError('pure static tensor function with complete outputs required')
    if len(program['nodes']) > 10000 or len(program['inputs']) + len(program['outputs']) > 64:
        raise ValueError('program size budget exceeded')
    graph = GraphBuilder()
    for i, arg in enumerate(program['inputs']):
        typ = arg['type']
        if typ != tensor_type(typ['shape'], typ['bits']): raise ValueError('noncanonical input type')
        graph.input(arg['name'], typ, i)
        if graph.inputs[-1] != arg: raise ValueError('input signature/range replay differs')
    for node in program['nodes']:
        typ, op, args, attrs = node['type'], node['op'], node['operands'], node['attributes']
        if typ != tensor_type(typ['shape'], typ['bits']): raise ValueError('noncanonical node type')
        if op not in OPERATORS: raise ValueError('unregistered tensor operator: ' + op)
        arity, fields = OPERATORS[op]
        if op == 'matmul' and not attrs: fields = set()
        if (arity is not None and len(args) != arity) or set(attrs) != fields:
            raise ValueError('operator arity/attributes mismatch: ' + op)
        if op == 'matmul' and attrs and attrs != {'promotion': 'signed', 'accumulation': 'modular'}:
            raise ValueError('only signed promotion and modular accumulation are defined')
        if any(v not in graph.values for v in args): raise ValueError('forward or missing SSA use')
        if op == 'constant' and (not isinstance(attrs['values'], list) or not attrs['values'] or
                                 any(type(v) is not int for v in attrs['values'])):
            raise ValueError('literal integer constants required')
        if op in ('slice', 'transpose') and any(type(v) is not int for seq in attrs.values() for v in seq):
            raise ValueError('literal integer indices required')
        if op == 'concat' and type(attrs['axis']) is not int: raise ValueError('literal concat axis required')
        graph.add(op, args, typ, deepcopy(attrs), deepcopy(node['origin']))
        if graph.nodes[-1] != node: raise ValueError('SSA/type/range replay differs at ' + node['id'])
    outputs = []
    for i, out in enumerate(program['outputs']):
        v = out['value']
        if v not in graph.values or not graph.values[v]['initialized']:
            raise ValueError('uninitialized/missing output')
        if out != {'position': i, 'value': v, 'type': graph.values[v]['type']}:
            raise ValueError('output signature mismatch')
        outputs.append(v)
    if program_from_builder(graph, outputs, program['entry']) != program:
        raise ValueError('program identity differs from semantic replay')
    return program


def make_program(program, target=Target(), source=None):
    validate_program(program); target.validate()
    # i64 remains importable. Lowering refuses it until exact device support exists.
    return seal({'schema': 'aie-optimizer-program-v1', 'program': deepcopy(program),
        'target': asdict(target), 'source': deepcopy(source) if source else {'kind': 'declared_tensor_ssa'},
        'interface': {'layout': 'logical_row_major', 'frame_alignment_bytes': 16,
                      'inputs': 'one_typed_PLIO_per_signature_input',
                      'outputs': 'one_typed_PLIO_per_signature_result',
                      'input_policy': 'distinct_immutable_values_per_frame',
                      'original_runtime_interface_preserved': False}})


def program_from_import(artifact, target=Target()):
    from .importing.frontends import validate_import
    program, binding = normalize_imported_program(validate_import(artifact))
    return make_program(program, target,
                        {'kind': 'verified_source_import', 'artifact': artifact, 'input_binding': binding})


def normalize_imported_program(program):
    """Preserve the complete function while separating source ABI from SSA.

    C++ argument positions include writable outputs. Dense SSA input IDs and
    pure result values therefore need an explicit, invertible source binding.
    The source frontend must have proved its restrict/effect domain first.
    """
    if program['effects'] not in ('pure_tensor_function',
            'writes to restricted output arrays; inputs read-only; locals do not escape'):
        raise ValueError('source effects lack a tensor-function adapter')
    graph = GraphBuilder(); mapped = {}; binding = {}
    for i, arg in enumerate(program['inputs']):
        v = graph.input(arg['name'], arg['type'], i)
        mapped[arg['id']] = v; binding[v] = arg['id']
    for n in program['nodes']:
        mapped[n['id']] = graph.add(n['op'], [mapped[v] for v in n['operands']], n['type'],
                                    deepcopy(n['attributes']), deepcopy(n['origin']))
    return program_from_builder(graph, [mapped[o['value']] for o in program['outputs']], program['entry']), binding


def validate_contract(contract):
    if contract.get('schema') != 'aie-optimizer-program-v1': raise ValueError('program contract schema')
    target = Target(**contract['target'])
    expected = make_program(contract['program'], target, contract['source'])
    if contract != expected: raise ValueError('program contract identity/interface mismatch')
    source = contract['source']
    if source['kind'] == 'verified_source_import':
        from .importing.frontends import validate_import
        original = validate_import(source['artifact'])
        normalized, binding = normalize_imported_program(original)
        if set(source) != {'kind', 'artifact', 'input_binding'} or normalized != contract['program'] or source['input_binding'] != binding:
            raise ValueError('whole source function differs from program')
    elif source != {'kind': 'declared_tensor_ssa'}:
        raise ValueError('unknown source provenance')
    return contract['program'], target


def live_nodes(program):
    values = {n['id']: n for n in program['nodes']}
    needed = {o['value'] for o in program['outputs']}
    for node in reversed(program['nodes']):
        if node['id'] in needed: needed.update(node['operands'])
    return [n for n in program['nodes'] if n['id'] in needed]


def elements(typ):
    return prod(typ['shape'])


def transfer_elements(typ):
    byte_count = elements(typ) * typ['bits'] // 8
    return ((byte_count + 15) // 16 * 16) // (typ['bits'] // 8)
