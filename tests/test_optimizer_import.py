import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from Optimizer.contracts import identity, validate_contract
from Optimizer.importing.ir import ImportFailure, evaluate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/optimizer_import'
SOURCE = (FIXTURES/'two_layer.mlir').read_text(encoding='utf-8')


@unittest.skipUnless(importlib.util.find_spec('xdsl'), 'optional requirements-import.txt not installed')
class ImportTests(unittest.TestCase):
    def imp(self, text=SOURCE, entry=None):
        from Optimizer.importing.mlir_frontend import import_mlir_text
        return import_mlir_text(text, entry)

    def check(self, artifact, cases=4):
        from Optimizer.importing.check import check_import
        return check_import(artifact,cases)

    def test_non_square_signature_mapping_and_three_independent_evaluations(self):
        artifact = self.imp(); admitted = artifact['backend_admission']
        self.assertEqual(admitted['status'],'supported')
        self.assertEqual(admitted['spec'],{'m':8,'k':16,'hidden':24,'n':16,'hidden_relu':False})
        self.assertEqual(admitted['input_binding'],{'X':'arg1','W0':'arg2','W1':'arg0'})
        self.assertEqual(self.check(artifact,8)['values_checked'],1024)

    def test_names_comments_and_generic_printing_do_not_define_semantics(self):
        from Optimizer.importing.mlir_frontend import export_mlir, parse_module
        from xdsl.printer import Printer
        from io import StringIO
        text = SOURCE.replace('@network','@arbitrary').replace('%samples','%foo').replace('%first','%bar')
        a = self.imp(text)
        self.assertEqual(a['backend_admission']['spec'],self.imp()['backend_admission']['spec'])
        round_trip = self.imp(export_mlir(a))
        self.assertEqual(round_trip['backend_admission'],a['backend_admission'])
        out = StringIO(); Printer(stream=out,print_generic_format=True).print_op(parse_module(text))
        generic = self.imp(out.getvalue())
        self.assertEqual(generic['backend_admission'],a['backend_admission'])
        self.check(generic,2)

    def test_hidden_relu_retained(self):
        a = self.imp(SOURCE.replace('%lo = arith.constant -128','%lo = arith.constant 0'))
        self.assertTrue(a['backend_admission']['spec']['hidden_relu'])
        self.check(a)

    def test_bare_truncation_is_imported_but_not_saturation(self):
        a = self.imp(SOURCE.replace('arith.trunci %upper :','arith.trunci %a :'))
        self.assertEqual(a['backend_admission']['status'],'unsupported')
        self.check(a)

    def test_bias_initial_value_is_preserved_and_backend_rejected(self):
        a = self.imp(SOURCE.replace('%zero = arith.constant 0','%zero = arith.constant 1'))
        self.assertIn('nonzero accumulator',a['backend_admission']['reason'])
        self.check(a)

    def test_extra_signature_input_is_not_silently_discarded(self):
        a = self.imp(SOURCE.replace('@network(', '@network(%spare: tensor<8x16xi8>, '))
        self.assertEqual(a['backend_admission']['status'],'unsupported')
        self.check(a,2)

    def test_branches_calls_multiple_outputs_transpose_and_strided_slices(self):
        a = self.imp((FIXTURES/'branched.mlir').read_text(encoding='utf-8'))
        self.assertEqual(a['backend_admission']['status'],'unsupported')
        self.assertEqual(len(a['program']['outputs']),2)
        self.check(a,8)
        outputs = evaluate(a['program'],{'arg0':list(range(12))})
        self.assertEqual(outputs[1],[1,3,2,4,5,7,6,8,9,11,10,12])

    def test_arbitrary_layer_count_preserved_not_forced_to_two_layers(self):
        # Compose the same pure helper twice: four GEMMs, three tensor arguments.
        text = SOURCE.replace('func.func @network','func.func private @network')
        text = text[:text.rfind('}')]+'''func.func @four(%w: tensor<24x16xi8>, %x: tensor<8x16xi8>, %v: tensor<16x24xi8>) -> tensor<8x16xi8> {
          %a = func.call @network(%w,%x,%v) : (tensor<24x16xi8>,tensor<8x16xi8>,tensor<16x24xi8>) -> tensor<8x16xi8>
          %b = func.call @network(%w,%a,%v) : (tensor<24x16xi8>,tensor<8x16xi8>,tensor<16x24xi8>) -> tensor<8x16xi8>
          return %b : tensor<8x16xi8>
        }}'''
        a = self.imp(text)
        self.assertEqual(sum(n['op']=='matmul' for n in a['program']['nodes']),4)
        self.assertEqual(a['backend_admission']['status'],'unsupported')
        self.check(a,3)

    def test_named_matmul_signed_widening(self):
        text = '''module { func.func @product(%x: tensor<2x3xi8>, %w: tensor<3x2xi8>) -> tensor<2x2xi32> {
          %z = arith.constant dense<0> : tensor<2x2xi32>
          %y = linalg.matmul ins(%x,%w : tensor<2x3xi8>,tensor<3x2xi8>) outs(%z : tensor<2x2xi32>) -> tensor<2x2xi32>
          return %y : tensor<2x2xi32>
        }}'''
        a = self.imp(text); self.check(a)
        self.assertEqual(evaluate(a['program'],{'arg0':[-128]*6,'arg1':[-128]*6}),[[49152]*4])

    def test_contraction_permuted_loop_order_and_transposed_input(self):
        text = '''#a = affine_map<(k,j,i)->(k,i)>
        #b = affine_map<(k,j,i)->(k,j)>
        #c = affine_map<(k,j,i)->(i,j)>
        module {func.func @p(%x: tensor<3x2xi8>, %w: tensor<3x4xi8>) -> tensor<2x4xi32> {
          %z = arith.constant dense<0> : tensor<2x4xi32>
          %r = linalg.generic {indexing_maps=[#a,#b,#c], iterator_types=["reduction","parallel","parallel"]}
          ins(%x,%w : tensor<3x2xi8>,tensor<3x4xi8>) outs(%z : tensor<2x4xi32>) {
          ^bb0(%a: i8, %b: i8, %c: i32):
            %aa = arith.extsi %a : i8 to i32
            %bb = arith.extsi %b : i8 to i32
            %m = arith.muli %bb,%aa : i32
            %s = arith.addi %m,%c : i32
            linalg.yield %s : i32
          } -> tensor<2x4xi32>
          return %r : tensor<2x4xi32>
        }}'''
        a = self.imp(text)
        self.assertIn('transpose',[n['op'] for n in a['program']['nodes']])
        self.check(a)

    def test_modular_integer_semantics_and_input_domain(self):
        a = self.imp('''module {func.func @wrap(%x: tensor<1x2xi8>) -> tensor<1x2xi8> {
          %y = arith.addi %x,%x : tensor<1x2xi8>
          return %y : tensor<1x2xi8>
        }}''')
        self.assertEqual(evaluate(a['program'],{'arg0':[127,-128]}),[[-2,0]])
        with self.assertRaisesRegex(ImportFailure,'domain'): evaluate(a['program'],{'arg0':[128,0]})

    def test_uninitialized_accumulator_is_rejected(self):
        text = SOURCE.replace('outs(%init0 : tensor<8x24xi32>)','outs(%scratch0 : tensor<8x24xi32>)')
        with self.assertRaisesRegex(ImportFailure,'uninitialized'): self.imp(text)

    def test_uninitialized_pointwise_destination_read_is_rejected(self):
        text = SOURCE.replace('linalg.yield %q : i8','linalg.yield %unused : i8')
        with self.assertRaisesRegex(ImportFailure,'uninitialized'): self.imp(text)

    def test_unknown_effectful_operation_and_non_tensor_memory_rejected(self):
        text = '''module {func.func @effect(%x: memref<4xi8>) { memref.dealloc %x : memref<4xi8> return }}'''
        with self.assertRaises(ImportFailure): self.imp(text)
        text = '''module {func.func private @unknown(tensor<2x2xi8>) -> tensor<2x2xi8>
          func.func @f(%x: tensor<2x2xi8>) -> tensor<2x2xi8> {
          %y = func.call @unknown(%x) : (tensor<2x2xi8>) -> tensor<2x2xi8>
          return %y : tensor<2x2xi8> }}'''
        with self.assertRaisesRegex(ImportFailure,'defined single-block'): self.imp(text)

    def test_dynamic_shape_and_encoded_layout_rejected(self):
        with self.assertRaises(ImportFailure): self.imp(SOURCE.replace('tensor<8x16xi8>','tensor<?x16xi8>'))
        text = '''module {func.func @f(%x: tensor<2x2xi8, "packed">) -> tensor<2x2xi8, "packed"> {return %x : tensor<2x2xi8, "packed">}}'''
        with self.assertRaisesRegex(ImportFailure,'layout'): self.imp(text)

    def test_poison_flags_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'poison'):
            self.imp(SOURCE.replace('arith.muli %xx, %ww :','arith.muli %xx, %ww overflow<nsw> :'))

    def test_unknown_semantic_attributes_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'unhandled'):
            self.imp(SOURCE.replace('arith.muli %xx, %ww :','arith.muli %xx, %ww {custom_semantics = true} :'))

    def test_wrong_reduction_body_is_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'product'):
            self.imp(SOURCE.replace('arith.addi %a, %p :','arith.addi %a, %a :'))

    def test_changed_named_fill_body_is_rejected(self):
        from Optimizer.importing.mlir_frontend import parse_module
        from xdsl.printer import Printer
        from io import StringIO
        module = parse_module(SOURCE)
        fill = next(op for op in module.walk() if op.name=='linalg.fill')
        fill.body.block.last_op.operands = [fill.body.block.args[1]]
        out = StringIO(); Printer(stream=out,print_generic_format=True).print_op(module)
        with self.assertRaisesRegex(ImportFailure,'non-copy'): self.imp(out.getvalue())

    def test_narrow_multiply_is_not_promoted_into_wide_product(self):
        text = SOURCE.replace('%p = arith.muli %xx, %ww : i32',
                              '%narrow = arith.muli %x, %w : i8\n        %p = arith.extsi %narrow : i8 to i32')
        with self.assertRaisesRegex(ImportFailure,'product'): self.imp(text)

    def test_source_operation_diagnostic_and_location_are_retained(self):
        with self.assertRaises(ImportFailure) as caught:
            self.imp(SOURCE.replace('arith.muli %xx, %ww :','arith.subi %xx, %ww :'))
        self.assertIn('@network/op',caught.exception.diagnostic()['site'])
        a = self.imp(SOURCE.replace('%zero = arith.constant 0 : i32','%zero = arith.constant 0 : i32 loc("source.cpp":12:3)'))
        self.assertIn('source.cpp',a['program']['nodes'][0]['origin']['location'])

    def test_out_of_bounds_slice_rejected(self):
        text = (FIXTURES/'branched.mlir').read_text(encoding='utf-8').replace('%a[0, 1][3, 2]','%a[0, 3][3, 2]')
        with self.assertRaisesRegex(ImportFailure,'slice'): self.imp(text)

    def test_recursion_and_ambiguous_entry_rejected(self):
        text = '''module {func.func @r(%x: tensor<1x1xi8>) -> tensor<1x1xi8> {
          %y = func.call @r(%x) : (tensor<1x1xi8>) -> tensor<1x1xi8>
          return %y : tensor<1x1xi8> }}'''
        with self.assertRaisesRegex(ImportFailure,'recursive'): self.imp(text)
        text = SOURCE.replace('  func.func @network','  func.func @another() { return }\n  func.func @network')
        with self.assertRaisesRegex(ImportFailure,'entry'): self.imp(text)
        self.assertEqual(self.imp(text,'network')['backend_admission']['status'],'supported')

    def test_source_and_ir_tampering_rejected_even_with_new_outer_hash(self):
        from Optimizer.importing.mlir_frontend import validate_import
        a = self.imp()
        for kind in ('source','graph','admission'):
            bad = copy.deepcopy(a)
            if kind == 'source': bad['source']['text'] = bad['source']['text'].replace('constant -128','constant -127')
            elif kind == 'graph': bad['program']['nodes'][0]['attributes']['values'] = [42]
            else: bad['backend_admission']['input_binding']['X'] = 'arg0'
            bad['id'] = identity({k:v for k,v in bad.items() if k != 'id'})
            with self.assertRaisesRegex(ImportFailure,'replay'): validate_import(bad)

    def test_source_hash_preserves_actual_crlf_bytes(self):
        from hashlib import sha256
        from Optimizer.importing.mlir_frontend import import_mlir
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'windows.mlir'; raw = SOURCE.replace('\n','\r\n').encode('utf-8')
            path.write_bytes(raw); a = import_mlir(path)
            self.assertEqual(a['source']['sha256'],sha256(raw).hexdigest())
            self.assertEqual(a['source']['text'].encode('utf-8'),raw)
            path.write_bytes(b'\xff')
            with self.assertRaisesRegex(ImportFailure,'source_encoding'): import_mlir(path)

    def test_contract_replays_import_and_proves_complete_source(self):
        from Optimizer.importing.admission import contract_from_import
        a = self.imp(); c = contract_from_import(a); validate_contract(c)
        bad = copy.deepcopy(c); bad['spec']['hidden_relu'] = True
        # Build a fully consistent declared contract, then attach incompatible source.
        from Optimizer.contracts import make_contract
        from Optimizer.semantics import Spec
        bad = make_contract(Spec(**bad['spec']),source=bad['source'])
        with self.assertRaisesRegex(ImportFailure,'complete source'): validate_contract(bad)
        b = self.imp(SOURCE.replace('%zero = arith.constant 0','%zero = arith.constant 1'))
        with self.assertRaisesRegex(ImportFailure,'backend_unsupported'): contract_from_import(b)

    def test_import_to_egraph_codegen_and_cpu(self):
        from Optimizer.importing.admission import contract_from_import
        from Optimizer.egraph_planner import enumerate_egraph_plans,validate_egraph_plan
        from Optimizer.backend import generate
        from Optimizer.runner import cpu_check
        a = self.imp(); c = contract_from_import(a)
        plans,report = enumerate_egraph_plans(c,max_rounds=1,max_terms=48,max_applications=3,max_states=64,
                                            keep=2,max_executions=2,execution_depth=1)
        self.assertTrue(plans)
        plan = next(p for p in plans if p['planned_compute_tiles']==1)
        validate_egraph_plan(c,plan)
        compiler = shutil.which('g++')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'candidate'; generate(c,plan,out,frames=2)
            if compiler: self.assertEqual(cpu_check(out,compiler)['status'],'passed')
            from Optimizer.importing.check import check_generated_import
            self.assertEqual(check_generated_import(out)['values_checked'],256)

    def test_cli_import_check_export_contract_and_rejection(self):
        from Optimizer.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); artifact = root/'import.json'
            self.assertEqual(main(['import-mlir',str(FIXTURES/'two_layer.mlir'),'--output',str(artifact)]),0)
            self.assertEqual(main(['import-check',str(artifact),'--cases','2','--output',str(root/'check.json')]),0)
            self.assertEqual(main(['import-export',str(artifact),'--output',str(root/'roundtrip.mlir')]),0)
            self.assertEqual(main(['import-contract',str(artifact),'--output',str(root/'contract.json')]),0)
            src = root/'bad.mlir'; src.write_text('module { totally_not_mlir }',encoding='utf-8')
            self.assertEqual(main(['import-mlir',str(src),'--output',str(root/'rejected.json')]),2)
            self.assertEqual(json.loads((root/'rejected.json').read_text())['status'],'rejected')


if __name__ == '__main__': unittest.main()
