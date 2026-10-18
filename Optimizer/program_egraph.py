"""Typed multi-output tensor e-graph, with replayable extraction certificates.

Raw terms form an acyclic witness DAG. Congruence may create cyclic e-classes;
only a finite, acyclic selection can be materialized. Origins and conservative
range intervals are deliberately absent from equality keys. Every extraction
recomputes those intervals through GraphBuilder instead of treating them as
semantic types or reusing a possibly incompatible range fact.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .contracts import identity, Target
from .global_extract import Node, extract_dags
from .importing.ir import GraphBuilder
from .program_ir import live_nodes, make_program, program_from_builder, seal, validate_program
from .program_equalities import default_tensor_equalities

SCHEMA = 'aie-optimizer-tensor-egraph-extraction-v1'
HARD_TERMS = 20000
HARD_STEPS = 128


def equality_identity():
    root = Path(__file__).parent
    return identity({name: sha256((root / name).read_bytes()).hexdigest() for name in
        ('program_egraph.py', 'program_equalities.py', 'program_rules.py', 'program_ir.py', 'importing/ir.py')})


class TensorEGraph:
    def __init__(self, program, registry=None):
        validate_program(program)
        self.base = deepcopy(program)
        self.registry = registry or default_tensor_equalities()
        self.terms, self.parent, self.steps = {}, {}, []
        self.step_results = []
        self.congruence_merges = 0
        self.roots = self._insert_program(program, HARD_TERMS)
        self.source_terms = set(self.terms)
        self.rebuild()

    def find(self, term):
        if term not in self.parent: raise ValueError('unknown e-graph term')
        root = term
        while self.parent[root] != root: root = self.parent[root]
        while self.parent[term] != term:
            previous = term; term = self.parent[term]; self.parent[previous] = root
        return root

    def _union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right: return False
        a, b = self.terms[left], self.terms[right]
        if a['type'] != b['type'] or (a['op'] == 'empty') != (b['op'] == 'empty'):
            raise ValueError('equality requires the same tensor type and initialization')
        self.parent[max(left, right)] = min(left, right)
        return True

    def _insert_program(self, program, max_terms):
        validate_program(program)
        if program['inputs'] != self.base['inputs']:
            raise ValueError('e-graph input identity/signature differs')
        mapped, encoded = {}, {}
        for n in program['inputs'] + live_nodes(program):
            is_input = 'op' not in n
            term = {'context': self.base['id'], 'op': 'input' if is_input else n['op'],
                'type': deepcopy(n['type']),
                'attributes': {'position': n['position']} if is_input else deepcopy(n['attributes']),
                'children': [] if is_input else [mapped[v] for v in n['operands']]}
            tid = identity(term); mapped[n['id']] = tid; encoded[tid] = term
        if len(self.terms.keys() | encoded.keys()) > max_terms:
            raise ValueError('e-graph term budget exceeded')
        # Budget failure is transactional: no partially inserted rule result.
        for tid, term in encoded.items():
            if tid not in self.terms: self.terms[tid] = term; self.parent[tid] = tid
        return [mapped[o['value']] for o in program['outputs']]

    def _key(self, tid):
        n = self.terms[tid]
        return (n['op'], identity(n['type']), identity(n['attributes']),
                tuple(self.find(v) for v in n['children']))

    def rebuild(self):
        changed = True
        while changed:
            changed = False; memo = {}
            for tid in sorted(self.terms):
                key = self._key(tid)
                if key in memo and self._union(tid, memo[key]):
                    changed = True; self.congruence_merges += 1
                else: memo[key] = tid

    def classes(self):
        result = {}; seen = set()
        for tid in sorted(self.terms):
            key = self._key(tid)
            if key in seen: continue
            seen.add(key); result.setdefault(self.find(tid), []).append(tid)
        return dict(sorted(result.items()))

    def _materialize(self, roots, selected=None):
        graph = GraphBuilder(); mapped = {}; done = set(); active = set()
        for arg in self.base['inputs']:
            graph.input(arg['name'], deepcopy(arg['type']), arg['position'])
        def token(t): return self.find(t) if selected is not None else t
        def term_for(t):
            if selected is None: return t
            if t not in selected or selected[t] not in self.terms or self.find(selected[t]) != t:
                raise ValueError('missing or mismatched selected e-node')
            return selected[t]
        pending = [(token(t), False) for t in reversed(roots)]
        while pending:
            c, exiting = pending.pop()
            if c in done: continue
            tid = term_for(c); n = self.terms[tid]
            if exiting:
                args = [mapped[token(v)] for v in n['children']]
                mapped[c] = (f'arg{n["attributes"]["position"]}' if n['op'] == 'input' else
                    graph.add(n['op'], args, deepcopy(n['type']), deepcopy(n['attributes']), {'egraph_term': tid}))
                active.remove(c); done.add(c); continue
            if c in active: raise ValueError('cyclic e-graph extraction')
            active.add(c); pending.append((c, True))
            pending.extend((token(v), False) for v in reversed(n['children']) if token(v) not in done)
        if selected is not None and set(selected) != done:
            raise ValueError('selection contains unused or noncanonical e-classes')
        result = program_from_builder(graph, [mapped[token(t)] for t in roots], self.base['entry'])
        validate_program(result)
        return result

    def materialize(self, selected):
        if type(selected) is not dict: raise ValueError('explicit selected e-node mapping required')
        return self._materialize(self.roots, selected)

    def applications(self, term):
        program = self._materialize([term]); root = program['outputs'][0]['value']
        return [s for s in self.registry.applications(program) if s.get('node_id') == root]

    def apply(self, term, step, *, max_terms=HARD_TERMS):
        if len(self.steps) >= HARD_STEPS: raise ValueError('e-graph proof step budget exceeded')
        program = self._materialize([term]); root = program['outputs'][0]['value']
        if step.get('node_id') != root: raise ValueError('local equality must target the witness root')
        changed = self.registry.apply(program, step)
        if len(changed['outputs']) != 1 or changed['outputs'][0]['type'] != program['outputs'][0]['type']:
            raise ValueError('local equality must preserve its complete result type')
        rhs = self._insert_program(changed, max_terms)[0]
        self._union(term, rhs); self.rebuild()
        self.steps.append({'term_id': term, 'rewrite': deepcopy(step)})
        self.step_results.append(rhs)
        return rhs

    def nodes(self, cost=None):
        # Zero is an optimistic time bound. It is not an operation-count model.
        return [Node(tid, c, tuple(self.find(v) for v in self.terms[tid]['children']),
                     0 if cost is None else cost(self.terms[tid]))
                for c, tids in self.classes().items() for tid in tids]

    def canonical_selection(self, selected):
        representatives = {self._key(tid):tid for terms in self.classes().values() for tid in terms}
        return {c:representatives[self._key(tid)] for c,tid in selected.items()}

    def witness_selection(self, preferred=None, *, max_states=10000):
        """Construct one finite witness, preferring source terms elsewhere.

        A static traversal preference is used only to seed structural diversity.
        All costs are zero and no measured or proxy latency pruning is applied.
        """
        forced = self.find(preferred) if preferred is not None else None
        originals = {}
        for tid in sorted(self.source_terms): originals.setdefault(self._key(tid), tid)
        nodes = self.nodes()
        if preferred is not None:
            n = self.terms[preferred]
            nodes = [v for v in nodes if v.eclass != forced] + [
                Node(preferred, forced, tuple(self.find(v) for v in n['children']), 0)]
        rows = []
        extract_dags(nodes, [self.find(r) for r in self.roots], max_states=max_states,
            max_complete=1, keep=1, visit_complete=lambda row: rows.append(row),
            option_priority=lambda n:0 if self._key(n.id) in originals else 1)
        if not rows: return None
        selected = rows[0]
        if preferred is not None and selected.get(forced) != preferred: return None
        return selected

    def saturate(self, *, max_rounds=2, max_applications=32, max_terms=1200):
        if (type(max_rounds) is not int or not 0 <= max_rounds <= 8 or
                type(max_applications) is not int or not 0 <= max_applications <= HARD_STEPS or
                type(max_terms) is not int or not len(self.terms) <= max_terms <= HARD_TERMS):
            raise ValueError('finite e-graph saturation budgets required')
        applied = set(); refused = []; rounds = 0; app_hit = False
        for _ in range(max_rounds):
            rounds += 1; did_work = False
            # New terms become eligible next round; no whole-program BFS copies.
            for tid in sorted(tuple(self.terms)):
                for step in self.applications(tid):
                    key = identity({'term_id': tid, 'rewrite': step})
                    if key in applied: continue
                    if len(self.steps) >= max_applications:
                        app_hit = True; break
                    applied.add(key)
                    try: self.apply(tid, step, max_terms=max_terms)
                    except ValueError as exc:
                        if str(exc) != 'e-graph term budget exceeded': raise
                        refused.append({'term_id': tid, 'rewrite': step, 'reason': str(exc)}); continue
                    did_work = True
                if app_hit: break
            if app_hit or not did_work: break
        return {'rounds': rounds, 'terms': len(self.terms), 'classes': len(self.classes()),
            'applications': len(self.steps), 'application_budget_reached': app_hit,
            'round_budget_reached': rounds == max_rounds and max_rounds > 0,
            'term_budget_refusals': refused, 'congruence_merges': self.congruence_merges,
            'rules': self.registry.catalog(), 'unbounded_saturation_claim': False}

    def certificate(self, selected):
        program = self.materialize(selected)
        return seal({'schema': SCHEMA, 'base_program_id': self.base['id'],
            'implementation': equality_identity(), 'rules': self.registry.catalog(),
            'steps': deepcopy(self.steps), 'roots': list(self.roots),
            'selected': dict(sorted(selected.items())), 'program_id': program['id']})


def check_extraction(certificate, original_program, registry=None):
    """Rebuild unions/congruence from source and registered rule replay only."""
    if (type(certificate) is not dict or set(certificate) !=
            {'schema', 'base_program_id', 'implementation', 'rules', 'steps', 'roots', 'selected', 'program_id', 'id'} or
            certificate['schema'] != SCHEMA or certificate != seal({k:v for k,v in certificate.items() if k != 'id'})):
        raise ValueError('invalid tensor e-graph certificate identity/schema')
    if certificate['base_program_id'] != original_program['id'] or certificate['implementation'] != equality_identity():
        raise ValueError('e-graph source/implementation identity differs')
    if type(certificate['steps']) is not list or len(certificate['steps']) > HARD_STEPS:
        raise ValueError('e-graph certificate step budget exceeded')
    graph = TensorEGraph(original_program, registry)
    if certificate['rules'] != graph.registry.catalog() or certificate['roots'] != graph.roots:
        raise ValueError('e-graph rule catalog/output roots differ')
    for step in certificate['steps']:
        if type(step) is not dict or set(step) != {'term_id', 'rewrite'}:
            raise ValueError('exact equality proof step required')
        graph.apply(step['term_id'], step['rewrite'])
    result = graph.materialize(certificate['selected'])
    if result['id'] != certificate['program_id']: raise ValueError('extracted program differs from proof')
    return result


def bind_extraction(contract, graph, selected):
    from .program_ir import validate_contract
    validate_contract(contract)
    if contract['source']['kind'] == 'verified_egraph_extraction':
        raise ValueError('restart equality saturation from the original contract, not nested provenance')
    if contract['program'] != graph.base: raise ValueError('e-graph belongs to another source program')
    certificate = graph.certificate(selected)
    program = check_extraction(certificate, contract['program'], graph.registry)
    return make_program(program, Target(**contract['target']),
        {'kind': 'verified_egraph_extraction', 'original': deepcopy(contract), 'certificate': certificate})
