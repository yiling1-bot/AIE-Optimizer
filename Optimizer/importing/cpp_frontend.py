"""C++ source closure -> compiler AST -> proved array-loop tensor semantics."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from ..contracts import identity
from .cpp_capture import capture_cpp,replay_capture,compiler_identity
from .cpp_lift import Lifter
from .admission import admit_two_layer
from .ir import fail


def implementation_identity():
    names=('cpp_frontend.py','cpp_capture.py','cpp_lift.py','ir.py','admission.py','frontends.py')
    return identity({n:sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def build_import(closure,ast):
    standard=next(f for f in closure['command']['effective_flags'] if f.startswith('-std='))
    program,interface=Lifter(ast,standard).lift(closure['entry'])
    result={'schema':'aie-optimizer-cpp-import-v1','status':'imported','source':closure,
        'frontend':{**compiler_identity(),'implementation':implementation_identity()},
        'compiler_ast':ast,'program':program,'source_interface':interface,
        'backend_admission':admit_two_layer(program),
        'scope':'closed dense-array C++ subset under declared restrict and arithmetic semantics; host Clang, no AIE target/API lowering'}
    result['id']=identity(result)
    return result


def import_cpp(path,entry,**kwargs):
    closure,ast,_=capture_cpp(path,entry,**kwargs)
    return build_import(closure,ast)


@lru_cache(maxsize=16)
def _replay(source_json,implementation,compiler):
    closure=json.loads(source_json)
    return json.dumps(build_import(closure,replay_capture(closure)),sort_keys=True)


def validate_import(artifact):
    if artifact.get('schema')!='aie-optimizer-cpp-import-v1':fail('not a C++ import artifact',code='invalid_import')
    expected=json.loads(_replay(json.dumps(artifact['source'],sort_keys=True),implementation_identity(),identity(compiler_identity())))
    if artifact!=expected:fail('C++ import fails source, AST and semantic replay',code='invalid_import')
    return expected['program']
