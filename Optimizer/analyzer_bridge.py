"""Reuse Analyzer primitives without changing its existing AIE1 entrypoints."""
from pathlib import Path
import sys
from hashlib import sha256

MODEL_ROOT=Path(__file__).resolve().parents[1]/'Model'
if str(MODEL_ROOT) not in sys.path:sys.path.insert(0,str(MODEL_ROOT))
from Timing.clocked_causal_graph import (ClockedCausalGraph,CausalRule,CausalParent,
    DeclaredEvent,diagnose_clocked_graph)
from Timing.buffer_lifecycle import ClockDomain
from Timing.component_cost_registry import (resolve_program_costs,load_cost_family,
    SIGNATURE_SCHEMA,REGISTRY_SCHEMA,FAMILY_SCHEMA,DIMENSIONS)


def _hashes():
    names=('clocked_causal_graph.py','buffer_lifecycle.py','aligned_max_constraints.py',
           'component_cost_registry.py')
    values={'Analyzer/'+name:sha256((MODEL_ROOT/'Timing'/name).read_bytes()).hexdigest() for name in names}
    for name in ('analyzer_bridge.py','phase_model.py','phase_costs.py'):
        values['Optimizer/'+name]=sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return values


_LOADED_HASHES=_hashes()


def dependency_hashes():
    now=_hashes()
    if now!=_LOADED_HASHES:raise RuntimeError('phase/Analyzer source changed during this process; restart against one snapshot')
    return now
