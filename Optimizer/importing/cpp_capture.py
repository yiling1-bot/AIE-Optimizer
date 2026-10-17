"""Capture compiler-typed C++ ASTs with a closed, replayable source closure.

Compile database commands are data, never shell commands. The supported flag
set excludes plugins, response files and output actions. Semantic import uses
project-contained includes; inventory may also describe external SDK headers.
"""
from ctypes import c_int, c_void_p, c_longlong
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

from ..contracts import identity
from .ir import ImportFailure, fail

VERSION = '18.1.1'


@lru_cache(maxsize=1)
def clang():
    try:
        from clang import cindex as c
        if version('libclang') != VERSION: fail('libclang 18.1.1 required',code='parser_version')
    except ImportError as exc:
        raise ImportFailure('missing_dependency','install Optimizer/requirements-cpp-import.txt') from exc
    lib = c.conf.lib
    bindings = (
        ('clang_getCursorBinaryOperatorKind',c_int,[c.Cursor]),
        ('clang_getCursorUnaryOperatorKind',c_int,[c.Cursor]),
        ('clang_getBinaryOperatorKindSpelling',c._CXString,[c_int]),
        ('clang_getUnaryOperatorKindSpelling',c._CXString,[c_int]),
        ('clang_Cursor_Evaluate',c_void_p,[c.Cursor]),
        ('clang_EvalResult_getKind',c_int,[c_void_p]),
        ('clang_EvalResult_getAsLongLong',c_longlong,[c_void_p]),
        ('clang_EvalResult_dispose',None,[c_void_p]),
        ('clang_getClangVersion',c._CXString,[]),
    )
    for name,result,args in bindings:
        fn = getattr(lib,name); fn.restype = result; fn.argtypes = args
    return c


@lru_cache(maxsize=1)
def compiler_identity():
    c = clang(); lib = Path(c.Config.library_path)/('libclang.dll' if __import__('os').name=='nt' else 'libclang.so')
    return {'package':'libclang','version':VERSION,
            'clang_version':c._CXString.from_result(c.conf.lib.clang_getClangVersion()),
            'library_sha256':sha256(lib.read_bytes()).hexdigest()}


def type_info(typ):
    t = typ.get_canonical(); kind = t.kind.name
    result = {'spelling':typ.spelling,'canonical':t.spelling,'kind':kind,
              'const':t.is_const_qualified(),'volatile':t.is_volatile_qualified(),
              'restrict':t.is_restrict_qualified()}
    if kind in ('LVALUEREFERENCE','RVALUEREFERENCE','POINTER'):
        result['pointee'] = type_info(t.get_pointee())
    elif kind == 'CONSTANTARRAY':
        result['extent'] = t.get_array_size(); result['element'] = type_info(t.get_array_element_type())
    elif kind in ('SCHAR','UCHAR','CHAR_S','CHAR_U','SHORT','USHORT','INT','UINT','LONG','ULONG','LONGLONG','ULONGLONG','BOOL'):
        result['bits'] = t.get_size()*8
        result['signed'] = kind in ('SCHAR','CHAR_S','SHORT','INT','LONG','LONGLONG')
    return result


def qualified(cursor):
    parts = [cursor.spelling]; p = cursor.semantic_parent
    while p and p.kind.name != 'TRANSLATION_UNIT':
        if p.spelling: parts.append(p.spelling)
        p = p.semantic_parent
    return '::'.join(reversed(parts))


def relative(path, root):
    try: return Path(path).resolve().relative_to(root).as_posix()
    except ValueError: return str(Path(path).resolve())


def site(cursor, root):
    loc = cursor.location
    return {'file':relative(loc.file.name,root) if loc.file else None,
            'line':loc.line,'column':loc.column,'offset':loc.offset}


def decl_id(cursor, root):
    loc = site(cursor,root)
    return f"{loc['file']}:{loc['offset']}:{cursor.kind.name}:{cursor.spelling}"


def evaluate_constant(cursor):
    c = clang(); value = c.conf.lib.clang_Cursor_Evaluate(cursor)
    if not value: return None
    try:
        if c.conf.lib.clang_EvalResult_getKind(value) == 1:
            return int(c.conf.lib.clang_EvalResult_getAsLongLong(value))
    finally:
        c.conf.lib.clang_EvalResult_dispose(value)
    return None


def node(cursor, root, budget):
    budget[0] += 1
    if budget[0] > 100000: fail('AST node budget exceeded',code='resource_limit')
    c = clang(); kind = cursor.kind.name
    result = {'kind':kind,'name':cursor.spelling,'type':type_info(cursor.type),
              'site':site(cursor,root),'id':decl_id(cursor,root)}
    if kind in ('VAR_DECL','FUNCTION_DECL'):
        result['storage'] = cursor.storage_class.name
    if kind=='VAR_DECL':result['tls']=cursor.tls_kind.name
    if kind in ('BINARY_OPERATOR','COMPOUND_ASSIGNMENT_OPERATOR','UNARY_OPERATOR'):
        which = 'Unary' if kind=='UNARY_OPERATOR' else 'Binary'
        enum = getattr(c.conf.lib,'clang_getCursor'+which+'OperatorKind')(cursor)
        result['operator'] = c._CXString.from_result(getattr(c.conf.lib,'clang_get'+which+'OperatorKindSpelling')(enum))
    if kind in ('INTEGER_LITERAL','CHARACTER_LITERAL','CXX_BOOL_LITERAL_EXPR','DECL_REF_EXPR','VAR_DECL','ENUM_CONSTANT_DECL'):
        value = evaluate_constant(cursor)
        if value is not None: result['constant'] = value
    ref = cursor.referenced
    if ref and kind in ('DECL_REF_EXPR','CALL_EXPR','MEMBER_REF_EXPR'):
        result['reference'] = {'id':decl_id(ref,root),'name':qualified(ref),'type':type_info(ref.type),
                               'kind':ref.kind.name,'site':site(ref,root)}
        if kind == 'CALL_EXPR':
            definition = ref.get_definition()
            result['reference']['definition'] = decl_id(definition,root) if definition else None
    result['children'] = [node(ch,root,budget) for ch in cursor.get_children()]
    if kind == 'CALL_EXPR':
        result['arguments'] = [node(ch,root,budget) for ch in cursor.get_arguments()]
    return result


def normalize_flags(args, source, root, directory):
    """Keep semantic flags, strip only recorded build outputs, reject ambiguity."""
    source,root,directory = Path(source).resolve(),Path(root).resolve(),Path(directory).resolve()
    normalized=[]; removed=[]; i=0
    paths = ('-I','-isystem','-iquote','-include')
    while i < len(args):
        arg = args[i]; i += 1
        if not isinstance(arg,str) or '\x00' in arg: fail('invalid compiler argument',code='compile_flags')
        if arg in ('-c','-S','-E','-g','-MMD','-MD','-MP','--driver-mode=g++'):
            removed.append(arg); continue
        if arg in ('-o','-MF','-MT','-MQ'):
            if i == len(args): fail('missing output flag value',code='compile_flags')
            removed.extend((arg,args[i])); i+=1; continue
        candidate=Path(arg)
        if not arg.startswith('-') and (directory/candidate).resolve()==source:
            removed.append(arg); continue
        path_flag = next((f for f in paths if arg==f or (f=='-I' and arg.startswith('-I'))),None)
        if path_flag:
            value=arg[len(path_flag):]
            if not value:
                if i==len(args): fail('missing include path',code='compile_flags')
                value=args[i];i+=1
            path=(directory/value).resolve()
            if not path.is_relative_to(root): fail('semantic imports require include paths inside project root',str(path),'source_closure')
            normalized.extend((path_flag,path.relative_to(root).as_posix() or '.'));continue
        if arg in ('-D','-U'):
            if i==len(args): fail('missing macro value',code='compile_flags')
            arg += args[i];i+=1
        if arg.startswith(('-D','-U')) and len(arg)>2:
            normalized.append(arg);continue
        if arg in ('-std=c++14','-std=c++17','-std=c++20','-std=c++23','-fno-exceptions','-fno-rtti',
                   '-nostdinc','-nostdinc++','-O0','-O1','-O2','-O3','-Os','-Og','-Wall','-Wextra','-Werror'):
            normalized.append(arg);continue
        if arg=='-x' and i<len(args) and args[i]=='c++':i+=1;continue
        fail('unsupported compiler flag (not silently discarded): '+arg,code='compile_flags')
    if not any(s.startswith('-std=') for s in normalized): normalized.insert(0,'-std=c++20')
    if sum(s.startswith('-std=') for s in normalized)!=1: fail('one C++ standard required',code='compile_flags')
    return normalized,removed


def read_command(source, root, database=None, args=(), command_index=None):
    source,root=Path(source).resolve(),Path(root).resolve()
    original=list(args); directory=root; selected=None
    if database:
        if args: fail('use a compile database or explicit flags, not both',code='compile_flags')
        path=Path(database).resolve(); path=path.parent if path.is_file() else path
        raw=json.loads((path/'compile_commands.json').read_text(encoding='utf-8'))
        exact=[entry for entry in raw if (Path(entry['directory'])/entry['file']).resolve()==source]
        if not exact:fail('no exact source entry in compilation database; inferred commands refused',code='compile_database')
        c=clang()
        commands=list(c.CompilationDatabase.fromDirectory(str(path)).getCompileCommands(str(source)) or [])
        if not commands: fail('no compilation command for source',code='compile_database')
        if len(commands)>1 and command_index is None: fail('select one compilation command explicitly',code='compile_database')
        selected=0 if command_index is None else command_index
        if not 0<=selected<len(commands): fail('compilation command index out of range',code='compile_database')
        command=commands[selected]; original=list(command.arguments); directory=Path(command.directory)
        flags=original[1:]
        if not any(f.startswith('-std=') for f in flags):
            fail('compilation database must state the C++ standard; compiler defaults are not inferred',code='compile_database')
    else: flags=original
    effective,removed=normalize_flags(flags,source,root,directory)
    return {'original_arguments':original,'original_directory':str(directory),'database_entry':selected,
            'effective_flags':effective,'removed_build_only_arguments':removed,
            'frontend':'libclang; original compiler command is not executed'}


def parse_file(source, root, flags):
    c=clang(); effective=['-x','c++']; i=0
    while i<len(flags):
        arg=flags[i];i+=1;effective.append(arg)
        if arg in ('-I','-isystem','-iquote','-include'):
            path=(root/flags[i]).resolve();i+=1
            if not path.is_relative_to(root): fail('include escapes source snapshot',code='source_closure')
            effective.append(str(path))
    try:tu=c.Index.create().parse(str(source),args=effective,options=c.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    except Exception as exc: raise ImportFailure('cpp_parse',str(exc),str(source)) from exc
    diagnostics=[{'severity':d.severity,'message':d.spelling,'line':d.location.line} for d in tu.diagnostics]
    if any(d['severity']>=3 for d in diagnostics):fail(json.dumps(diagnostics),str(source),'cpp_parse')
    return tu,diagnostics


def describe(tu, root, entry):
    declarations=[];functions=[];macros=[];budget=[0]
    def visit(cursor):
        loc=cursor.location
        if loc.file and not Path(loc.file.name).resolve().is_relative_to(root):return
        kind=cursor.kind.name
        if kind=='FUNCTION_DECL' and cursor.is_definition():
            item=node(cursor,root,budget);item['qualified_name']=qualified(cursor)
            item['parameters']=[node(a,root,budget) for a in cursor.get_arguments()]
            item['return_type']=type_info(cursor.result_type)
            functions.append(item);return
        if kind=='MACRO_DEFINITION' and loc.file:
            macros.append({'name':cursor.spelling,'tokens':[t.spelling for t in cursor.get_tokens()],'site':site(cursor,root)});return
        if kind in ('VAR_DECL','TYPEDEF_DECL','TYPE_ALIAS_DECL','ENUM_DECL','STRUCT_DECL','CLASS_DECL','FUNCTION_TEMPLATE'):
            declarations.append(node(cursor,root,budget));return
        if kind in ('TRANSLATION_UNIT','NAMESPACE','LINKAGE_SPEC'):
            for ch in cursor.get_children():visit(ch)
        elif kind not in ('FUNCTION_DECL','INCLUSION_DIRECTIVE','MACRO_INSTANTIATION','STATIC_ASSERT','USING_DIRECTIVE','USING_DECLARATION') and loc.file:
            declarations.append(node(cursor,root,budget))
    visit(tu.cursor)
    entries=[f for f in functions if f['qualified_name']==entry]
    if len(entries)!=1: fail('entry must identify exactly one defined non-template free function',entry,'cpp_entry')
    return {'functions':functions,'declarations':declarations,'macros':macros,'entry_id':entries[0]['id']}


def capture_cpp(source, entry, *, project_root=None, args=(), database=None, command_index=None):
    source=Path(source).resolve();root=Path(project_root).resolve() if project_root else source.parent
    if not source.is_relative_to(root): fail('source must be inside project root',code='source_closure')
    command=read_command(source,root,database,args,command_index)
    tu,diagnostics=parse_file(source,root,command['effective_flags'])
    paths={source}|{Path(i.include.name).resolve() for i in tu.get_includes()}
    if any(not p.is_relative_to(root) for p in paths):
        fail('implicit/system headers outside project root need an explicit SDK capture adapter',code='source_closure')
    if len(paths)>128 or sum(p.stat().st_size for p in paths)>8_000_000: fail('source closure budget exceeded',code='resource_limit')
    files={p.relative_to(root).as_posix():p.read_bytes().decode('utf-8') for p in sorted(paths)}
    closure={'main':source.relative_to(root).as_posix(),'entry':entry,'files':files,
             'file_sha256':{n:sha256(s.encode('utf-8')).hexdigest() for n,s in files.items()},
             'original_root':str(root),'command':command}
    original=describe(tu,root,entry)
    replayed=replay_capture(closure)
    if original!=replayed:fail('source closure did not reproduce initial AST',code='source_changed')
    return closure,replayed,diagnostics


def materialize(closure, root):
    if len(closure['files'])>128 or sum(len(s.encode('utf-8')) for s in closure['files'].values())>8_000_000:
        fail('source snapshot budget exceeded',code='resource_limit')
    expected={n:sha256(s.encode('utf-8')).hexdigest() for n,s in closure['files'].items()}
    if expected!=closure['file_sha256']:fail('source closure hash mismatch',code='invalid_import')
    for name,text in closure['files'].items():
        path=(root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or '..' in Path(name).parts:
            fail('snapshot path escapes root',code='invalid_import')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(text.encode('utf-8'))
    if closure['main'] not in closure['files']:fail('main source absent',code='invalid_import')
    return root/closure['main']


def replay_capture(closure):
    with tempfile.TemporaryDirectory(prefix='aie-cpp-import-') as directory:
        root=Path(directory).resolve();source=materialize(closure,root)
        flags=closure['command']['effective_flags']
        normalized,_=normalize_flags(flags,source,root,root)
        if flags!=normalized:fail('compiler flags are not normalized',code='invalid_import')
        tu,_=parse_file(source,root,flags)
        loaded={Path(i.include.name).resolve() for i in tu.get_includes()}
        if any(not p.is_relative_to(root) or p.relative_to(root).as_posix() not in closure['files'] for p in loaded):
            fail('replay read a file outside the captured closure',code='source_closure')
        return describe(tu,root,closure['entry'])
