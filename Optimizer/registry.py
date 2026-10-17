"""Executable, typed local rewrites with replayable derivations.

Registration is a trusted Python extension API, not a semantic proof checker.
Rules must supply an identity argument and tests. UNKNOWN conditions fail closed.
The bounded rewrite graph is not an e-graph with congruence closure.
"""
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable
from .semantics import Expr, build, expression_id, infer


@dataclass(frozen=True)
class Match:
    path: tuple = ()
    parameters: tuple = ()

    def params(self):
        return dict(self.parameters)


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str


@dataclass(frozen=True)
class Rule:
    id: str
    version: int
    mechanism: str
    sources: tuple
    identity_argument: str
    match: Callable
    check: Callable
    rewrite: Callable

    def metadata(self):
        return {k: getattr(self, k) for k in
                ('id', 'version', 'mechanism', 'sources', 'identity_argument')}


def walk(expr, path=()):
    yield path, expr
    for i, child in enumerate(expr.args):
        yield from walk(child, path+(i,))


def at(expr, path):
    for index in path:
        if type(index) is not int or not 0 <= index < len(expr.args):
            raise ValueError('invalid rewrite path')
        expr = expr.args[index]
    return expr


def replace_at(expr, path, new):
    if not path:
        return new
    at(expr, path)
    children = list(expr.args)
    children[path[0]] = replace_at(children[path[0]], path[1:], new)
    return replace(expr, args=tuple(children))


def validate_expression(expr, spec):
    """Validate the IR grammar as well as shape/type/range analysis."""
    arity = {'input': (0, 0), 'matmul': (2, 2), 'quantize': (1, 1),
             'slice': (1, 1), 'sum': (1, 10000), 'concat': (1, 10000),
             'rematerialize': (1, 1)}
    count = 0
    for _, node in walk(expr):
        count += 1
        if count > 10000:
            raise ValueError('expression node budget exceeded')
        if node.op not in arity or not arity[node.op][0] <= len(node.args) <= arity[node.op][1]:
            raise ValueError('unsupported operator or arity')
        if not isinstance(node.args, tuple) or type(node.relu) is not bool:
            raise ValueError('invalid expression field type')
        if node.op == 'input' and node.name not in ('X', 'W0', 'W1'):
            raise ValueError('unknown input')
        if node.op in ('slice', 'concat') and type(node.axis) is not int:
            raise ValueError('integer axis required')
        if node.op == 'slice' and any(type(v) is not int for v in (node.begin, node.end)):
            raise ValueError('integer slice required')
    return infer(expr, spec)


def decode_expr(value):
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(Expr)}:
        raise ValueError('noncanonical expression serialization')
    if not isinstance(value['args'], (tuple, list)):
        raise ValueError('invalid expression arguments')
    return Expr(**{**value, 'args': tuple(decode_expr(a) for a in value['args'])})


class Registry:
    def __init__(self):
        self._rules = {}
        self._execution = None

    @property
    def execution(self):
        if self._execution is None:
            from .execution_rules import default_execution_registry
            self._execution = default_execution_registry()
        return self._execution

    def register(self, rule):
        if rule.id in self._rules:
            raise ValueError('duplicate rule: '+rule.id)
        if not rule.id or type(rule.version) is not int or rule.version < 1 or not rule.identity_argument:
            raise ValueError('version and identity argument required')
        if not all(callable(getattr(rule, k)) for k in ('match', 'check', 'rewrite')):
            raise ValueError('executable rule callbacks required')
        self._rules[rule.id] = rule
        return self

    def catalog(self):
        return [r.metadata() for r in self._rules.values()]

    def proposals(self, expr, spec):
        for path, node in walk(expr):
            for rule in self._rules.values():
                for parameters in rule.match(node, spec):
                    yield rule.id, Match(path, tuple(sorted(parameters.items())))

    def apply(self, expr, spec, rule_id, match):
        before = validate_expression(expr, spec)
        rule = self._rules[rule_id]
        node = at(expr, match.path)
        decision = rule.check(node, spec, match.params())
        if decision.status != 'ALLOW':
            raise ValueError(decision.status+': '+decision.reason)
        local = rule.rewrite(node, spec, match.params())
        if validate_expression(local, spec) != infer(node, spec):
            raise ValueError('rewrite changed local type/range contract')
        result = replace_at(expr, match.path, local)
        if validate_expression(result, spec) != before:
            raise ValueError('rewrite changed output contract')
        step = {'rule': rule.id, 'version': rule.version, 'path': list(match.path),
                'parameters': match.params(), 'before': expression_id(expr),
                'after': expression_id(result), 'condition': asdict(decision)}
        return result, step

    def replay(self, spec, steps):
        expr = build(spec)['materialized_layers'].expression
        for step in steps:
            rule = self._rules.get(step['rule'])
            if rule is None or rule.version != step['version']:
                raise ValueError('unknown rule or version: '+step['rule'])
            expr, actual = self.apply(expr, spec, step['rule'],
                Match(tuple(step['path']), tuple(sorted(step['parameters'].items()))))
            if actual != step:
                raise ValueError('rewrite certificate differs on replay')
        return expr

    def explore(self, spec, max_candidates=32, max_depth=2, max_attempts=4096):
        if any(type(v) is not int or v < 1 for v in (max_candidates, max_attempts)) or max_depth < 0:
            raise ValueError('invalid exploration budget')
        seed = build(spec)['materialized_layers'].expression
        entries = [{'expression': seed, 'id': expression_id(seed), 'steps': [], 'depth': 0}]
        seen = {entries[0]['id']}; edges = []; refused = []; attempts = 0; truncated = False
        for current in entries:
            if current['depth'] >= max_depth:
                continue
            for rule_id, match in self.proposals(current['expression'], spec):
                if attempts >= max_attempts:
                    return entries, {'edges': edges, 'refused': refused, 'attempts': attempts,
                                     'budget_reached': True}
                attempts += 1
                try:
                    candidate, step = self.apply(current['expression'], spec, rule_id, match)
                except ValueError as e:
                    refused.append({'rule': rule_id, 'path': list(match.path), 'reason': str(e)})
                    continue
                if step['after'] == current['id']:
                    continue
                if step['after'] not in seen:
                    if len(entries) >= max_candidates:
                        truncated = True
                        continue
                    seen.add(step['after'])
                    entries.append({'expression': candidate, 'id': step['after'],
                                    'steps': current['steps']+[step], 'depth': current['depth']+1})
                edges.append(step)
        return entries, {'edges': edges, 'refused': refused, 'attempts': attempts,
                         'budget_reached': truncated}
