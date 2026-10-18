"""Guarded, callable operator implementations for the whole-program backend."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Implementation:
    id: str
    version: int
    operators: tuple
    guard: object
    emit: object
    scratch_bytes: int = 0
    uses_aie_api: bool = False
    description: str = ''
    explore: bool = True


class ImplementationRegistry:
    def __init__(self): self.entries = {}

    def register(self, implementation):
        if implementation.id in self.entries: raise ValueError('duplicate implementation ID')
        self.entries[implementation.id] = implementation

    def resolve(self, name, node, values, target):
        impl = self.entries.get(name)
        if impl is None or node['op'] not in impl.operators: raise ValueError('unknown/incompatible operator implementation: ' + name)
        reason = impl.guard(node, values, target)
        if reason is not None: raise ValueError(name + ': ' + reason)
        return impl

    def options(self, node, values, target):
        return [i.id for i in self.entries.values() if node['op'] in i.operators and i.guard(node, values, target) is None]

    def catalog(self):
        return [{'id': i.id, 'version': i.version, 'operators': list(i.operators),
                 'scratch_bytes': i.scratch_bytes, 'uses_aie_api': i.uses_aie_api,
                 'description': i.description, 'explore': i.explore} for i in self.entries.values()]


def _scalar(context, order='ijk'):
    c = context; n = c['node']; typ = n['type']; dest = c['destination']; bits = typ['bits']
    if n['op'] != 'matmul':
        return [f' for(unsigned p=0;p<{c["elements"]}U;++p) {dest}[p]=int{bits}_t({c["scalar"](n,"p")});']
    a, b, init = n['operands']; m, width = typ['shape']; k = c['values'][a]['type']['shape'][1]
    seed = c['read'](init, f'i*{width}+j'); aa = c['read'](a, f'i*{k}+r'); bb = c['read'](b, f'r*{width}+j')
    if order == 'ijk':
        return [f' for(unsigned i=0;i<{m};++i) for(unsigned j=0;j<{width};++j) {{',
            f'  uint64_t acc=uint64_t(int64_t({seed}));',
            f'  for(unsigned r=0;r<{k};++r) acc+=uint64_t(int64_t({aa})*int64_t({bb}));',
            f'  {dest}[i*{width}+j]=int{bits}_t(wrap_u(acc,{bits})); }}']
    return [f' for(unsigned i=0;i<{m};++i) for(unsigned j=0;j<{width};++j) {dest}[i*{width}+j]={seed};',
        f' for(unsigned i=0;i<{m};++i) for(unsigned r=0;r<{k};++r) for(unsigned j=0;j<{width};++j)',
        f'  {dest}[i*{width}+j]=int{bits}_t(wrap_u(uint64_t(int64_t({dest}[i*{width}+j]))+uint64_t(int64_t({aa})*int64_t({bb})),{bits}));']


def _mmul_guard(grid):
    def guard(node, values, target):
        if target.architecture != 'AIE-ML' or target.toolchain != 'Vitis-2024.2': return 'requires AIE-ML / Vitis 2024.2'
        a, b, init = (values[v] for v in node['operands']); m, n = node['type']['shape']; k = a['type']['shape'][1]
        if a['type']['bits'] != 8 or b['type']['bits'] != 8 or node['type']['bits'] != 32:
            return 'requires signed i8 x i8 -> i32'
        if m % 4 or k % 8 or n % (8 * grid): return 'requires complete 4x8x8 microtiles and output grid'
        bound = k * max(map(abs, a['range'])) * max(map(abs, b['range']))
        if bound > 2**31 - 1 or node['may_wrap']: return 'full dot product and initialized result must fit signed i32'
        return None
    return guard


def _mmul(grid):
    def emit(c):
        node = c['node']; a, b, init = node['operands']; m, n = node['type']['shape']; k = c['values'][a]['type']['shape'][1]
        out = c['destination']; lines = [' {', f' for(unsigned rb=0;rb<{m};rb+=4) for(unsigned cb=0;cb<{n};cb+={8*grid}) {{',
                                       ' alignas(32) int8_t aa[32];']
        for g in range(grid): lines.append(f' alignas(32) int8_t bb{g}[64]; alignas(32) int32_t acc{g}[32];')
        lines += ['#ifndef AIE_OPT_HOST']
        for g in range(grid): lines.append(f' aie::mmul<4,8,8,int8,int8,acc32> mm{g};')
        lines += ['#endif', f' for(unsigned kb=0;kb<{k};kb+=8) {{']
        aa = c['read'](a, f'(rb+r)*{k}+kb+s')
        lines.append(f' for(unsigned r=0;r<4;++r) for(unsigned s=0;s<8;++s) aa[r*8+s]=int8_t({aa});')
        for g in range(grid):
            bb = c['read'](b, f'(kb+r)*{n}+cb+{g*8}+s')
            lines += [f' for(unsigned r=0;r<8;++r) for(unsigned s=0;s<8;++s) bb{g}[r*8+s]=int8_t({bb});',
                '#ifdef AIE_OPT_HOST', f' for(unsigned r=0;r<4;++r) for(unsigned s=0;s<8;++s) {{',
                f' int32_t value=kb?acc{g}[r*8+s]:0;',
                f' for(unsigned t=0;t<8;++t) value+=int32_t(aa[r*8+t])*int32_t(bb{g}[t*8+s]);',
                f' acc{g}[r*8+s]=value; }}', '#else',
                f' if(kb==0) mm{g}.mul(aie::load_v<32>(aa),aie::load_v<64>(bb{g}));',
                f' else mm{g}.mac(aie::load_v<32>(aa),aie::load_v<64>(bb{g}));', '#endif']
        lines += ['}', '#ifndef AIE_OPT_HOST']
        for g in range(grid): lines.append(f' aie::store_v(acc{g},mm{g}.to_accum().template to_vector<int32>(0));')
        lines += ['#endif']
        for g in range(grid):
            ix = f'(rb+r)*{n}+cb+{g*8}+s'; seed = c['read'](init, ix)
            lines.append(f' for(unsigned r=0;r<4;++r) for(unsigned s=0;s<8;++s) {out}[{ix}]=int32_t(wrap(int64_t(acc{g}[r*8+s])+int64_t({seed}),32));')
        return lines + ['}', '}']
    return emit


def default_implementations():
    from .program_ir import OPERATORS
    registry = ImplementationRegistry(); guard = lambda n, v, t: None
    registry.register(Implementation('scalar.elementwise', 1, tuple(o for o in OPERATORS if o not in ('matmul', 'empty')),
        guard, _scalar, description='Explicit modular scalar operations and logical views', explore=False))
    for order in ('ijk', 'ikj'):
        registry.register(Implementation('scalar.matmul.' + order, 1, ('matmul',), guard,
                                         lambda context, order=order: _scalar(context, order),
                                         description='Portable exact modular matrix product', explore=False))
    for grid in (1, 2):
        registry.register(Implementation(f'aie_ml.mmul.i8.grid1x{grid}', 1, ('matmul',), _mmul_guard(grid), _mmul(grid),
            32 + 192 * grid, True, 'Row-major packing, AIE MMUL, initialized row-major i32 output; no quantizer movement'))
    return registry
