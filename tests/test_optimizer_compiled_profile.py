"""Synthetic report fixtures test the inspector, not compiler performance."""
import copy,json,tempfile,unittest
from pathlib import Path
from Optimizer.backend import generate
from Optimizer.contracts import make_contract
from Optimizer.ir_planner import make_ir_plan
from Optimizer.compiled_profile import read_compiled_profile


class CompiledProfileTests(unittest.TestCase):
    def fixture(self,root):
        contract=make_contract();plan=make_ir_plan(contract)
        generate(contract,plan,root)
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        config={'aiearch':'aie2','aie_freq':1250000000,'pl_ip_block':[]}
        for name in ('X','W0','W1','Y'):
            config['pl_ip_block'].append({'name':name,'pl_freq':contract['target']['plio_hz'],
                'arguments':['out_y.txt' if name=='Y' else 'data/'+name+'.txt','int8'],
                'axi_stream':[{'direction':'me_to_pl' if name=='Y' else 'pl_to_me','bus_width':128}]})
        report={'blockInstances':{},'blockTypes':{},'mapping':{'blockInstanceMapping':{}}}
        for i,b in enumerate(manifest['kernel_bindings']):
            key=str(i);col,row=b['tile'];tile=f'{col}_{row}'
            report['blockInstances'][key]={'blockType':key}
            report['blockTypes'][key]={'type':'function','function':b['function']}
            report['mapping']['blockInstanceMapping'][key]={'coreInfo':{'column':col,'row':row}}
            p=root/'Work/aie'/tile/'Release';p.mkdir(parents=True)
            (p/(tile+'.map')).write_text("Memory map for memory 'PM':\nUsed = 1234\n",encoding='utf-8')
            (p/(tile+'.calltree')).write_text('__AIE_ARCH__=20\nMaximum stack size : 128\n 32 128 0 1 256 0 synthetic_function\n',encoding='utf-8')
        self.save(root/'Work/config/scsim_config.json',{'SimulationConfig':config})
        self.save(root/'Work/reports/compiler_report.json',report)
        return manifest,config,report

    def save(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')

    def test_compiler_evidence_does_not_consume_runtime_or_invent_cycles(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'candidate';m,c,r=self.fixture(root)
            first=read_compiled_profile(root)
            (root/'native_result.json').write_text('not valid JSON: must not read runtime measurements',encoding='utf-8')
            (root/'out_y.txt').write_text('T 999999999 ns',encoding='utf-8')
            self.assertEqual(first,read_compiled_profile(root))
            self.assertFalse(first['runtime_timing_used']);self.assertIsNone(first['latency_ns'])
            self.assertEqual(first['profiles'][0]['program_bytes'],1234)
            self.assertEqual(first['profiles'][0]['maximum_stack_bytes'],128)
            self.assertIsNone(first['profiles'][0]['compute_cycles'])
            self.assertIsNone(first['profiles'][0]['spill_bytes'])
            self.assertTrue(first['compiler_evidence_hashes'])

    def test_source_architecture_mapping_and_plio_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'candidate';m,c,r=self.fixture(root)
            path=root/next(iter(m['files']));raw=path.read_bytes();path.write_bytes(raw+b'changed')
            with self.assertRaisesRegex(ValueError,'source hash'):read_compiled_profile(root)
            path.write_bytes(raw)
            wrong=copy.deepcopy(c);wrong['aiearch']='aie1'
            self.save(root/'Work/config/scsim_config.json',{'SimulationConfig':wrong})
            with self.assertRaisesRegex(ValueError,'architecture'):read_compiled_profile(root)
            wrong=copy.deepcopy(c);wrong['pl_ip_block'][0]['axi_stream'][0]['bus_width']=64
            self.save(root/'Work/config/scsim_config.json',{'SimulationConfig':wrong})
            with self.assertRaisesRegex(ValueError,'PLIO'):read_compiled_profile(root)
            self.save(root/'Work/config/scsim_config.json',{'SimulationConfig':c})
            r['mapping']['blockInstanceMapping']['0']['coreInfo']['column']+=1
            self.save(root/'Work/reports/compiler_report.json',r)
            with self.assertRaisesRegex(ValueError,'mapping differs'):read_compiled_profile(root)


if __name__=='__main__':unittest.main()
