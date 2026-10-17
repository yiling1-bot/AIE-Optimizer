"""Typed congruence closure and independently replayable local equalities.

The raw term DAG is immutable and finite. E-classes may contain cycles; clients
must extract an acyclic implementation. Rule callbacks are trusted extensions,
as in registry.py; certificates check their guards, versions and exact outputs.
No serialized union-find or claimed equality is accepted as evidence.
"""
from collections import deque
from dataclasses import asdict, replace
from .contracts import identity
from .registry import Match, decode_expr, validate_expression
from .semantics import expression_id, infer


class EGraph:
    def __init__(self, spec, context, registry):
        spec.validate()
        if not context:
            raise ValueError('contract/input-epoch context required')
        self.spec, self.context, self.registry = spec, context, registry
        self.terms = {}; self.parent = {}; self.members = {}; self.uses = {}
        self.memo = {}; self.keys = {}; self.dirty = deque(); self.proofs = []
        self.types = {}; self.merges = 0

    def find(self, term):
        root = term
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[term] != term:
            nxt = self.parent[term]; self.parent[term] = root; term = nxt
        return root

    def key(self, term):
        e = self.terms[term]
        return (e.op, e.name, e.axis, e.begin, e.end, e.stage, e.relu,
                tuple(self.find(expression_id(c)) for c in e.args))

    def add(self, expr):
        validate_expression(expr, self.spec)
        def visit(e):
            tid = expression_id(e)
            if tid in self.terms:
                return tid
            children = [visit(c) for c in e.args]
            self.terms[tid] = e; self.parent[tid] = tid
            self.members[tid] = {tid}; self.uses[tid] = set()
            self.types[tid] = infer(e, self.spec)
            for c in children:
                self.uses[self.find(c)].add(tid)
            self.dirty.append(tid)
            return tid
        tid = visit(expr); self.rebuild(); return tid

    def _union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        ta, tb = self.types[a], self.types[b]
        if (ta.rows, ta.cols, ta.dtype) != (tb.rows, tb.cols, tb.dtype):
            raise ValueError('e-class shape/dtype mismatch')
        # Equal values can have different conservative interval proofs. Keep the
        # tighter independently established bound; never infer an unsafe range.
        a, b = sorted((a, b))
        self.parent[b] = a
        self.types[a] = replace(self.types[a], max_abs=min(ta.max_abs, tb.max_abs))
        self.members[a].update(self.members.pop(b))
        self.uses[a].update(self.uses.pop(b))
        self.dirty.extend(sorted(self.uses[a])); self.merges += 1
        return True

    def rebuild(self):
        while self.dirty:
            tid = self.dirty.popleft(); old = self.keys.get(tid)
            if old is not None and self.memo.get(old) == tid:
                del self.memo[old]
            key = self.key(tid); self.keys[tid] = key
            other = self.memo.get(key)
            if other is not None:
                self._union(tid, other)
            else:
                self.memo[key] = tid

    def equivalent(self, a, b):
        return self.find(a) == self.find(b)

    def apply(self, lhs, rule_id, parameters):
        """Only a guard-checked registered local rewrite adds an equality."""
        node = self.terms[lhs]
        rhs, step = self.registry.apply(node, self.spec, rule_id,
                                       Match((), tuple(sorted(parameters.items()))))
        rid = self.add(rhs)
        proof = {'lhs': lhs, 'rhs': rid, 'step': step}
        if proof not in self.proofs:
            self.proofs.append(proof)
        changed = self._union(lhs, rid); self.rebuild()
        return rid, changed

    def classes(self):
        """One deterministic raw witness per canonical e-node."""
        result = {}
        for root in sorted(self.members):
            nodes = {}
            for tid in sorted(self.members[root]):
                nodes.setdefault(self.key(tid), tid)
            result[root] = tuple(sorted(nodes.values()))
        return result

    def witnesses(self):
        """Least-size finite representative; cyclic e-nodes cannot bootstrap it."""
        best = {}; classes = self.classes()
        changed = True
        while changed:
            changed = False
            for root, terms in classes.items():
                for tid in terms:
                    e = self.terms[tid]
                    children = [self.find(expression_id(c)) for c in e.args]
                    if not all(c in best for c in children):
                        continue
                    size = 1 + sum(best[c][0] for c in children)
                    expr = replace(e, args=tuple(best[c][2] for c in children))
                    candidate = (size, expression_id(expr), expr)
                    if root not in best or candidate[:2] < best[root][:2]:
                        best[root] = candidate; changed = True
        return {k: v[2] for k, v in best.items()}

    def certificate(self, roots, selected):
        if len(roots) != len(selected) or not roots:
            raise ValueError('one selected expression per root required')
        selected_ids = [self.add(e) for e in selected]
        if not all(self.equivalent(a, b) for a, b in zip(roots, selected_ids)):
            raise ValueError('selection is not in the root equivalence class')
        # Flat raw DAG avoids exponentially duplicating common subexpressions.
        terms = []
        for tid, e in self.terms.items():
            terms.append({'id': tid, 'node': {**asdict(e),
                'args': [expression_id(c) for c in e.args]}})
        cert = {'schema': 'aie-equality-certificate-v1', 'context': self.context,
                'spec': asdict(self.spec), 'terms': terms, 'equalities': self.proofs.copy(),
                'roots': list(roots), 'selected': selected_ids}
        cert['id'] = identity(cert)
        return cert


def check_certificate(cert, spec, context, registry, max_terms=20000, max_proofs=10000):
    expected = {'schema','context','spec','terms','equalities','roots','selected','id'}
    if (not isinstance(cert, dict) or set(cert) != expected or
        cert['schema'] != 'aie-equality-certificate-v1' or cert['context'] != context or
        cert['spec'] != asdict(spec) or cert['id'] != identity({k:v for k,v in cert.items() if k!='id'})):
        raise ValueError('equality certificate identity/context mismatch')
    if len(cert['terms']) > max_terms or len(cert['equalities']) > max_proofs:
        raise ValueError('equality certificate budget exceeded')
    graph = EGraph(spec, context, registry)
    for item in cert['terms']:
        if not isinstance(item,dict) or set(item) != {'id','node'}:
            raise ValueError('invalid certificate term')
        node = item['node']
        if not isinstance(node,dict) or not isinstance(node.get('args'), list):
            raise ValueError('invalid certificate term node')
        try:
            expr = decode_expr({**node, 'args': [asdict(graph.terms[c]) for c in node['args']]})
        except (KeyError, TypeError, RecursionError) as e:
            raise ValueError('certificate raw DAG must be finite and topologically ordered') from e
        if expression_id(expr) != item['id']:
            raise ValueError('certificate term hash mismatch')
        graph.add(expr)
    for proof in cert['equalities']:
        if not isinstance(proof,dict) or set(proof) != {'lhs','rhs','step'}:
            raise ValueError('invalid equality proof')
        step = proof['step']
        try:
            rule = registry._rules.get(step['rule'])
            if rule is None or rule.version != step['version'] or step['path'] != []:
                raise ValueError('unknown equality rule/version or nonlocal proof')
            actual, actual_step = registry.apply(graph.terms[proof['lhs']],spec,step['rule'],
                Match((),tuple(sorted(step['parameters'].items()))))
            if actual_step != step or expression_id(actual) != proof['rhs'] or proof['rhs'] not in graph.terms:
                raise ValueError('equality proof differs on replay')
            graph._union(proof['lhs'],proof['rhs']); graph.rebuild()
        except (KeyError, TypeError) as e:
            raise ValueError('invalid equality proof reference') from e
    if (not cert['roots'] or len(cert['roots']) != len(cert['selected']) or
        any(t not in graph.terms for t in cert['roots']+cert['selected']) or
        not all(graph.equivalent(a,b) for a,b in zip(cert['roots'],cert['selected']))):
        raise ValueError('selected roots lack an equality proof')
    graph.proofs = cert['equalities'].copy()
    return graph, [graph.terms[t] for t in cert['selected']]
