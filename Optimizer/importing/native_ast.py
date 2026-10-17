"""Audit filtered AST dumps from the real target compiler, without lifting APIs.

Driver success is insufficient: require JSON AST objects and matching original
kernel sources. Calls are typed lexical evidence, not a dynamic protocol trace.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
from .ir import fail
from ..contracts import identity


def parse_stream(text):
    decoder=json.JSONDecoder();result=[];end=0
    for match in re.finditer(r'(?m)^\{\s*\n\s*"id"\s*:',text):
        if match.start()<end:continue
        try:item,end=decoder.raw_decode(text,match.start())
        except json.JSONDecodeError as exc:fail(str(exc),code='native_ast_parse')
        if not isinstance(item,dict) or 'kind' not in item:fail('invalid AST root',code='native_ast_parse')
        result.append(item)
    if not result:fail('successful driver produced no JSON AST evidence',code='native_ast_empty')
    return result


def summarize(roots):
    functions=[];seen=set()
    def references(n):
        refs=[]
        if n.get('kind')=='DeclRefExpr' and 'referencedDecl' in n:
            refs.append({'kind':'direct','declaration':n['referencedDecl'],'expression_type':n.get('type')})
        if n.get('kind')=='MemberExpr':
            refs.append({'kind':'member','name':n.get('name'),'declaration_id':n.get('referencedMemberDecl'),
                         'expression_type':n.get('type'),'object_types':[c.get('type') for c in n.get('inner',[])]})
        for child in n.get('inner',[]):refs.extend(references(child))
        return refs
    def function(n):
        if n.get('id') in seen:return
        seen.add(n.get('id'));calls=[];loops=[]
        def body(v,ancestors):
            kind=v.get('kind');current=ancestors
            if kind in ('ForStmt','WhileStmt','IfStmt'):
                item={'kind':kind,'range':v.get('range'),'id':v.get('id')};loops.append(item);current=ancestors+[item]
            if kind in ('CallExpr','CXXMemberCallExpr','CXXOperatorCallExpr'):
                inners=v.get('inner',[])
                callee=references(inners[0]) if inners else []
                calls.append({'lexical_index':len(calls),'kind':kind,'range':v.get('range'),
                    'callee':callee,'callee_resolution':'declaration_reference_present' if callee else 'unresolved_in_filtered_dump',
                    'result_type':v.get('type'),'argument_types':[x.get('type') for x in inners[1:]],
                    'control_ancestors':current,'semantic_status':'unmodeled; declaration/body contract required'})
            for child in v.get('inner',[]):body(child,current)
        for child in n.get('inner',[]):
            if child.get('kind')=='CompoundStmt':body(child,[])
        functions.append({'name':n.get('name'),'mangled_name':n.get('mangledName'),'type':n.get('type'),
            'location':n.get('loc'),'parameters':[{'name':x.get('name'),'type':x.get('type')} for x in n.get('inner',[]) if x.get('kind')=='ParmVarDecl'],
            'calls':calls,'control':loops})
    def visit(n):
        if n.get('kind') in ('FunctionDecl','CXXMethodDecl') and any(c.get('kind')=='CompoundStmt' for c in n.get('inner',[])):
            function(n)
        else:
            for c in n.get('inner',[]):visit(c)
    for root in roots:visit(root)
    if not functions:fail('no defined functions in filtered AST',code='native_ast_empty')
    return functions


def inspect_native_ast(directory,source_artifact):
    root=Path(directory).resolve();source=Path(source_artifact).resolve()
    meta=json.loads((root/'capture.json').read_text(encoding='utf-8'))
    if meta.get('schema')!='aie-optimizer-native-ast-capture-v1':fail('unknown native AST capture schema')
    if '-D__AIE_ARCH__=20' not in meta['original_command']:fail('AIE-ML compiler target not evidenced')
    for name,digest in meta['source_files'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or sha256(path.read_bytes()).hexdigest()!=digest:fail('captured source hash mismatch',name,'native_ast_source')
    streams=[];rejected=[]
    for record in meta['records']:
        name=record['label'];path=(root/(name+'.json-stream')).resolve()
        if not path.is_relative_to(root):fail('capture stream path escapes root')
        if record['returncode']!=0:
            rejected.append({'stream':name,'reason':'driver failed','returncode':record['returncode']});continue
        try:
            raw=path.read_bytes();objects=parse_stream(raw.decode('utf-8'));functions=summarize(objects)
            streams.append({'stream':name,'sha256':sha256(raw).hexdigest(),'root_objects':len(objects),'functions':functions})
        except ValueError as exc:rejected.append({'stream':name,'reason':str(exc)})
    if not streams:fail('no usable source-bound native AST streams',code='native_ast_empty')
    result={'schema':'aie-optimizer-native-ast-inventory-v1','status':'typed_inventory_only',
        'capture_sha256':sha256((root/'capture.json').read_bytes()).hexdigest(),'source_files':meta['source_files'],
        'streams':streams,'rejected_streams':rejected,'architecture':'AIE-ML',
        'semantic_admission':False,'dynamic_protocol_recovered':False,'latency_ns':None,
        'scope':'real compiler resolved source types/calls; lexical nesting only; no API semantic equivalence or runtime ordering proof'}
    result['id']=identity(result);return result
