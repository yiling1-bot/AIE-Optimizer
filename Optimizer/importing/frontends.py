"""Dispatch by versioned artifact schema; keep optional dependencies lazy."""
from .ir import fail


def validate_import(artifact):
    if artifact.get('schema')=='aie-optimizer-mlir-import-v1':
        from .mlir_frontend import validate_import as validate
    elif artifact.get('schema')=='aie-optimizer-cpp-import-v1':
        from .cpp_frontend import validate_import as validate
    else:fail('unsupported source frontend artifact',code='invalid_import')
    return validate(artifact)


def source_kind(artifact):
    return {'aie-optimizer-mlir-import-v1':'verified_mlir_tensor_import_v1',
            'aie-optimizer-cpp-import-v1':'verified_cpp_tensor_import_v1'}[artifact['schema']]
