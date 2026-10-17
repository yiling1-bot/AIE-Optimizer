from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from Optimizer.importing.ir import ImportFailure
from Optimizer.importing.native_ast import inspect_native_ast,parse_stream,summarize


class NativeASTTests(unittest.TestCase):
    def function(self):
        return {'id':'0x01','kind':'FunctionDecl','name':'fused_0','type':{'qualType':'void (Buffer &)'},
          'inner':[{'kind':'CompoundStmt','inner':[{'kind':'CXXMemberCallExpr','type':{'qualType':'void'},
            'inner':[{'kind':'MemberExpr','name':'acquire','referencedMemberDecl':'0x02',
                      'inner':[{'kind':'DeclRefExpr','type':{'qualType':'Buffer'},'referencedDecl':{'name':'input','kind':'ParmVarDecl'}}]}]}]}]}

    def test_success_log_without_ast_is_not_admitted(self):
        with self.assertRaisesRegex(ImportFailure,'no JSON AST'):parse_stream('Compilation finished successfully (0 errors)')

    def test_typed_call_inventory_does_not_claim_dynamic_semantics(self):
        data='compiler banner\n'+json.dumps(self.function(),indent=2)+'\ncompiler timings\n'
        f=summarize(parse_stream(data))[0]
        self.assertEqual(f['calls'][0]['callee'][0]['name'],'acquire')
        self.assertIn('unmodeled',f['calls'][0]['semantic_status'])

    def test_source_hash_binding_and_explicit_unadmitted_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'kernels.cpp').write_bytes(b'fixture')
            capture={'schema':'aie-optimizer-native-ast-capture-v1','original_command':['-D__AIE_ARCH__=20'],
                'source_files':{'kernels.cpp':sha256(b'fixture').hexdigest()},'records':[{'label':'kernels','returncode':0}]}
            (root/'capture.json').write_text(json.dumps(capture));(root/'kernels.json-stream').write_text(json.dumps(self.function(),indent=2))
            result=inspect_native_ast(root,source)
            self.assertFalse(result['semantic_admission']);self.assertFalse(result['dynamic_protocol_recovered']);self.assertIsNone(result['latency_ns'])
            (source/'kernels.cpp').write_bytes(b'changed')
            with self.assertRaisesRegex(ImportFailure,'source hash'):inspect_native_ast(root,source)
