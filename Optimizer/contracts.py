"""Contracts are independent of a particular deployment's core count."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from .semantics import Spec


def identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Target:
    architecture: str = 'AIE-ML'
    toolchain: str = 'Vitis-2024.2'
    columns: int = 38
    rows: int = 8
    max_compute_tiles: int = 16
    tile_memory_bytes: int = 65536
    plio_hz: int = 200000000

    def validate(self):
        if self.architecture != 'AIE-ML':
            raise ValueError('this initial native backend requires AIE-ML')
        if self.toolchain != 'Vitis-2024.2':
            raise ValueError('this initial native backend requires Vitis 2024.2')
        if any(type(v) is not int or v <= 0 for k, v in asdict(self).items()
               if k not in ('architecture', 'toolchain')):
            raise ValueError('positive integer resource limits required')
        if self.rows < 4 or self.max_compute_tiles > self.columns*self.rows:
            raise ValueError('invalid physical region or compute cap')


def make_contract(spec=Spec(hidden_relu=False), target=Target(), source=None):
    spec.validate(); target.validate()
    contract = {'schema': 'aie-optimizer-contract-v1', 'spec': asdict(spec),
        'target': asdict(target), 'source': source or {'kind': 'declared_tensor_semantics'},
        'numerics': {'inputs': 'int8', 'accumulator': 'int32', 'shift': 0,
                     'Q0': 'sat8_relu' if spec.hidden_relu else 'sat8', 'Q1': 'sat8'},
        'interface': {'id': 'three_full_frame_plio128_v1',
            'inputs': ['X', 'W0', 'W1'], 'outputs': ['Y'], 'weights': 'distinct_per_frame',
            'source_schedule': 'all three source files available at simulation start',
            'original_interface_preserved': False,
            'scope': 'normalized comparison campaign; not timing-comparable to the original 13-PLIO graph'},
        'semantic_goal': 'Y=Q1(Q0(X@W0)@W1)'}
    contract['id'] = identity(contract)
    return contract


def validate_contract(contract):
    if contract.get('schema') != 'aie-optimizer-contract-v1':
        raise ValueError('unsupported contract schema')
    if contract.get('id') != identity({k:v for k,v in contract.items() if k != 'id'}):
        raise ValueError('contract identity mismatch')
    spec, target = Spec(**contract['spec']), Target(**contract['target'])
    expected = make_contract(spec, target, contract['source'])
    if contract != expected:
        raise ValueError('contract semantics or interface not supported by this backend')
    if contract['source'].get('kind') in ('verified_mlir_tensor_import_v1','verified_cpp_tensor_import_v1'):
        from .importing.admission import validate_contract_source
        validate_contract_source(contract['source'], spec)
    return spec, target


def import_reviewed_k4(project, target=Target()):
    """Fingerprint-gated semantic lifting of the reviewed K4 two-layer template.

    Does not import arbitrary C++, original deployment or original timing.
    Any unreviewed change is rejected, rather than guessed from symbol names.
    """
    project = Path(project)
    expected = {
        'chain.cpp': 'e67c0afc4dec4cc7238f1db55e7bd42d4a609ee7a59dea7b11f2c8821919b3f5',
    }
    files = {}
    for name in ('chain.cpp', 'math.hpp', 'config.hpp', 'graph.hpp', 'kernels.hpp'):
        raw = (project/name).read_bytes()
        files[name] = sha256(raw).hexdigest()
        if name in expected and files[name] != expected[name]:
            raise ValueError('unreviewed source fingerprint: '+name)
    from .k4_profile import FILES
    if (project/'run_config.hpp').exists():
        raise ValueError('run_config.hpp override requires a separate reviewed profile')
    if files != FILES:
        raise ValueError('source closure differs from reviewed K4 profile')
    source = {'kind':'reviewed_k4_semantic_import', 'path':str(project), 'files':files,
              'profile':identity(FILES), 'original_mapping_imported':False,
              'scope':'32x32 two-layer sat8, no bias/ReLU; graph A semantics only'}
    return make_contract(Spec(hidden_relu=False), target, source)
