import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from Optimizer.importing.ir import ImportFailure,evaluate
from Optimizer.contracts import identity

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures/optimizer_cpp'
SOURCE=(FIXTURES/'two_layer.cpp').read_text(encoding='utf-8')
HEADER=(FIXTURES/'dimensions.hpp').read_text(encoding='utf-8')


@unittest.skipUnless(importlib.util.find_spec('clang'),'optional libclang dependency not installed')
class CppImportTests(unittest.TestCase):
    def imp(self,text=SOURCE,header=HEADER,args=()):
        from Optimizer.importing.cpp_frontend import import_cpp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'two_layer.cpp').write_bytes(text.encode('utf-8'));(root/'dimensions.hpp').write_bytes(header.encode('utf-8'))
            return import_cpp(root/'two_layer.cpp','network',args=args)

    def check(self,a,cases=3):
        from Optimizer.importing.cpp_check import check_cpp_import
        return check_cpp_import(a,cases)

    def test_real_compiler_ast_shapes_calls_order_and_original_cpp(self):
        a=self.imp();b=a['backend_admission']
        self.assertEqual(b['status'],'supported');self.assertEqual(b['spec']['hidden'],24)
        self.assertEqual(b['input_binding'],{'X':'arg1','W0':'arg3','W1':'arg0'})
        self.assertEqual(a['program']['outputs'][0]['source_parameter'],2)
        self.assertIn('dimensions.hpp',a['source']['file_sha256'])
        self.assertEqual(self.check(a,8)['values_checked'],1024)

    def test_macro_and_constant_dimensions_are_evaluated_by_compiler(self):
        a=self.imp(args=['-DIMPORT_ROWS=4','-std=c++17'])
        self.assertEqual(a['backend_admission']['spec']['m'],4)
        self.check(a)

    def test_source_replay_does_not_require_original_files(self):
        from Optimizer.importing.cpp_frontend import validate_import
        a=self.imp(SOURCE.replace('\n','\r\n'))
        self.assertEqual(validate_import(a),a['program'])
        self.assertIn('\r\n',a['source']['files']['two_layer.cpp'])
        self.check(a,2)

    def test_names_do_not_supply_semantics(self):
        a=self.imp(SOURCE.replace('limit_value','not_a_quantizer_name').replace('first_stage','something_else').replace('accumulator','qwerty'))
        self.assertEqual(a['backend_admission']['status'],'supported');self.check(a)

    def test_no_restrict_refuses_unknown_alias(self):
        with self.assertRaisesRegex(ImportFailure,'cpp_alias_not_proven'):self.imp(SOURCE.replace('& __restrict output','& output'))

    def test_pointers_refuse_unknown_extents(self):
        text='void network(const signed char* x,signed char* __restrict y) { y[0]=x[0]; }'
        with self.assertRaisesRegex(ImportFailure,'cpp_array_extent'):self.imp(text)

    def test_uninitialized_array_is_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'uninitialized'):
            self.imp(SOURCE.replace('  first_stage(input, first, temporary);',''))

    def test_uninitialized_scalar_accumulator_is_a_structured_rejection(self):
        with self.assertRaisesRegex(ImportFailure,'explicit initializer'):
            self.imp(SOURCE.replace('int total = 0;','int total;'))

    def test_in_place_array_dependence_is_rejected(self):
        text=SOURCE.replace('temporary[row][reduction] * second[reduction][column]',
                            'output[row][reduction] * second[reduction][column]')
        with self.assertRaisesRegex(ImportFailure,'extent|in-place'):self.imp(text)

    def test_nonzero_initial_accumulator_is_retained_but_not_admitted(self):
        a=self.imp(SOURCE.replace('int total = 0','int total = 1'))
        self.assertIn('nonzero accumulator',a['backend_admission']['reason']);self.check(a)

    def test_cxx20_narrowing_is_not_saturation(self):
        text=SOURCE.replace('return (I8)(value > 127 ? 127 : value < -128 ? -128 : value);','return (I8)value;')
        a=self.imp(text);self.assertEqual(a['backend_admission']['status'],'unsupported');self.check(a)
        with self.assertRaisesRegex(ImportFailure,'implementation-defined'):self.imp(text,args=['-std=c++17'])

    def test_signed_overflow_and_narrow_product_are_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'signed.*overflow'):
            self.imp(SOURCE.replace('int total = 0','int total = 2147483647'))
        with self.assertRaisesRegex(ImportFailure,'multiplication'):
            self.imp(SOURCE.replace('temporary[row][reduction] * second[reduction][column]',
                '(I8)(temporary[row][reduction] * second[reduction][column])'))

    def test_wrong_conditional_cannot_be_called_a_clamp(self):
        text=SOURCE.replace('value > 127 ? 127 :','value > 127 ? 126 :')
        with self.assertRaisesRegex(ImportFailure,'bound'):self.imp(text)

    def test_volatile_and_global_side_effects_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'volatile'):
            self.imp(SOURCE.replace('int total = 0','volatile int total = 0'))
        with self.assertRaisesRegex(ImportFailure,'global'):
            self.imp('int state=0;\n'+SOURCE)

    def test_thread_local_accumulator_or_counter_is_not_treated_as_fresh(self):
        for text in (SOURCE.replace('int total = 0','thread_local int total = 0'),
                     SOURCE.replace('for (int row = 0','for (thread_local int row = 0')):
            with self.subTest(text=text[:20]),self.assertRaisesRegex(ImportFailure,'thread-local'):self.imp(text)

    def test_unknown_calls_and_dynamic_control_flow_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'pure expression|unsupported|verified'):
            self.imp(SOURCE.replace('  return (I8)', '  if(value==0) return 0;\n  return (I8)'))
        text=SOURCE.replace('#include "dimensions.hpp"','#include "dimensions.hpp"\nI8 unknown(int);').replace('limit_value(total)','unknown(total)')
        with self.assertRaisesRegex(ImportFailure,'external/API'):self.imp(text)

    def test_non_unit_and_partial_output_domains_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'loop'):self.imp(SOURCE.replace('++row','row+=2'))
        with self.assertRaisesRegex(ImportFailure,'extent'):self.imp(SOURCE.replace('row < M','row < M-1'))

    def test_affine_simplification_cannot_hide_cpp_overflow(self):
        text=SOURCE.replace('samples[row][reduction]','samples[row + 2147483647 - 2147483647][reduction]')
        with self.assertRaisesRegex(ImportFailure,'index.*overflow'):self.imp(text)

    def test_loop_bound_cast_is_not_silently_erased(self):
        with self.assertRaisesRegex(ImportFailure,'narrowing'):
            self.imp(SOURCE.replace('reduction < H','reduction < (signed char)(H+256)'))

    def test_global_initializer_call_and_file_asm_rejected(self):
        with self.assertRaisesRegex(ImportFailure,'global'):
            self.imp('int external(); const int sneaky=(external(),0);\n'+SOURCE)
        with self.assertRaisesRegex(ImportFailure,'global'):
            self.imp('asm("");\n'+SOURCE)

    def test_compile_database_argument_and_command_forms(self):
        from Optimizer.importing.cpp_frontend import import_cpp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'two_layer.cpp';src.write_text(SOURCE);(root/'dimensions.hpp').write_text(HEADER)
            entry={'directory':str(root),'file':str(src),'arguments':['clang++','-std=c++20','-DIMPORT_ROWS=4','-c',str(src),'-o','unused.o']}
            db=root/'compile_commands.json';db.write_text(json.dumps([entry]))
            a=import_cpp(src,'network',database=db)
            self.assertEqual(a['backend_admission']['spec']['m'],4);self.assertFalse((root/'unused.o').exists())
            del entry['arguments'];entry['command']='clang++ -std=c++17 -DIMPORT_ROWS=8 -c two_layer.cpp -o unused.o'
            db.write_text(json.dumps([entry]));a=import_cpp(src,'network',database=db)
            self.assertEqual(a['backend_admission']['spec']['m'],8)

    def test_plugin_response_and_unhandled_flags_rejected(self):
        for flag in ('@flags.rsp','-Xclang','-fplugin=evil','-fwrapv','-march=native'):
            with self.subTest(flag=flag),self.assertRaisesRegex(ImportFailure,'compile_flags'):self.imp(args=[flag])

    def test_compile_database_never_infers_flags_from_another_file(self):
        from Optimizer.importing.cpp_frontend import import_cpp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'two_layer.cpp';src.write_text(SOURCE);(root/'dimensions.hpp').write_text(HEADER)
            db=root/'compile_commands.json';db.write_text(json.dumps([{'directory':str(root),'file':'different.cpp','arguments':['clang++','-c','different.cpp']}]))
            with self.assertRaisesRegex(ImportFailure,'exact source'):import_cpp(src,'network',database=db)

    def test_compile_database_does_not_change_the_implicit_language_standard(self):
        from Optimizer.importing.cpp_frontend import import_cpp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'two_layer.cpp';src.write_text(SOURCE);(root/'dimensions.hpp').write_text(HEADER)
            db=root/'compile_commands.json';db.write_text(json.dumps([{'directory':str(root),'file':str(src),'arguments':['clang++','-c',str(src)]}]))
            with self.assertRaisesRegex(ImportFailure,'C\\+\\+ standard'):import_cpp(src,'network',database=db)

    def test_source_ast_and_graph_tampering_replayed(self):
        from Optimizer.importing.cpp_frontend import validate_import
        from hashlib import sha256
        a=self.imp()
        for kind in ('source','ast','graph'):
            bad=copy.deepcopy(a)
            if kind=='source':
                bad['source']['files']['dimensions.hpp']=HEADER.replace('K = 16','K = 8')
                bad['source']['file_sha256']['dimensions.hpp']=sha256(bad['source']['files']['dimensions.hpp'].encode()).hexdigest()
            elif kind=='ast':bad['compiler_ast']['functions'][0]['name']='fake'
            else:bad['program']['nodes'][0]['attributes']['values']=[9]
            bad['id']=identity({k:v for k,v in bad.items() if k!='id'})
            with self.assertRaises(ImportFailure):validate_import(bad)

    def test_snapshot_path_escape_rejected(self):
        from Optimizer.importing.cpp_capture import materialize
        from hashlib import sha256
        c={'main':'../x.cpp','files':{'../x.cpp':'oops'},'file_sha256':{'../x.cpp':sha256(b'oops').hexdigest()}}
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ImportFailure,'escapes'):materialize(c,Path(tmp))

    def test_pointwise_multi_output_source_preserved(self):
        text='''void network(const signed char (&x)[4][8], signed char (& __restrict y)[4][8], int (& __restrict z)[8][4]) {
          for(int i=0;i<4;++i) for(int j=0;j<8;++j) y[i][j]=x[i][j];
          for(int j=0;j<8;++j) for(int i=0;i<4;++i) z[j][i]=(int)x[i][j]*2;
        }'''
        a=self.imp(text);self.assertEqual(len(a['program']['outputs']),2)
        self.assertEqual(a['backend_admission']['status'],'unsupported');self.check(a)

    def test_closure_and_codegen_cli_round_trip(self):
        from Optimizer.__main__ import main
        from Optimizer.importing.admission import contract_from_import
        from Optimizer.ir_planner import make_ir_plan
        from Optimizer.backend import generate
        from Optimizer.runner import cpu_check
        from Optimizer.importing.check import check_generated_import
        a=self.imp();c=contract_from_import(a)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifact=root/'import.json';artifact.write_text(json.dumps(a))
            self.assertEqual(main(['import-export',str(artifact),'--output',str(root/'exported')]),0)
            self.assertEqual((root/'exported/dimensions.hpp').read_bytes(),HEADER.encode())
            out=root/'candidate';generate(c,make_ir_plan(c,groups=2),out,frames=2)
            self.assertEqual(cpu_check(out,shutil.which('g++'))['status'],'passed')
            self.assertEqual(check_generated_import(out)['values_checked'],256)


if __name__=='__main__':unittest.main()
