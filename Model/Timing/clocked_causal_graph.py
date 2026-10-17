"""Conditional clocked causal graphs and complete joint cost inference.

Rules describe named operational boundaries, not physical completion by name.
The graph is an explicitly unrolled DAG: ownership/BD dependencies must already
be supplied by a separately checked program/resource binding. Native admission
is deliberately absent. Training observations constrain costs; prediction has
no argument for observations or simulator endpoint timestamps.
"""
from dataclasses import dataclass, asdict
from math import prod
import copy
import hashlib
import json

from analysis_errors import InvalidAnalysisInput, UnsupportedAnalysis
from .buffer_lifecycle import ClockDomain, validate_clock
from .aligned_max_constraints import CostDomain, MaxParent, AlignedMaxConstraint, ConstraintSolution, solve_aligned_max


def _bad(message):
    raise InvalidAnalysisInput('invalid_clocked_causal_graph', message)


def _name(value):
    if type(value) is not str or not value or value != value.strip():
        _bad('Literal nonempty identity required')


def _integer(value):
    if type(value) is not int or value < 0:
        _bad('Nonnegative integer picoseconds required')


def _evidence(value):
    if type(value) is not tuple or not value:
        _bad('Immutable explicit scope evidence required')
    for item in value:
        _name(item)


@dataclass(frozen=True)
class DeclaredEvent:
    id: str
    requested_ps: int
    provenance: str = 'declared_request'


@dataclass(frozen=True)
class CausalParent:
    event_id: str
    fixed_delay_ps: int = 0
    cost: str | None = None


@dataclass(frozen=True)
class CausalRule:
    id: str
    clock: ClockDomain
    parents: tuple[CausalParent, ...]
    observation_semantics: str


@dataclass(frozen=True)
class ClockedCausalGraph:
    declared_events: tuple[DeclaredEvent, ...]
    rules: tuple[CausalRule, ...]
    cost_names: tuple[str, ...]
    evidence: tuple[str, ...]
    scope: str = 'conditional_unrolled_operational_graph'


@dataclass(frozen=True)
class TrainingCase:
    id: str
    graph: ClockedCausalGraph
    observations_ps: tuple[tuple[str, int], ...]
    role: str = 'training_only'


def _canonical(value):
    return _canonical_json(asdict(value))


def _canonical_json(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError):
        _bad('Input must have a literal, type-preserving JSON identity')
    return hashlib.sha256(raw).hexdigest()


def _graph(graph):
    if type(graph) is not ClockedCausalGraph or graph.scope != 'conditional_unrolled_operational_graph':
        _bad('Exact conditional unrolled graph required')
    _evidence(graph.evidence)
    if any(type(x) is not tuple for x in (graph.declared_events, graph.rules, graph.cost_names)):
        _bad('Graph collections must be immutable tuples')
    for name in graph.cost_names:
        _name(name)
    if len(set(graph.cost_names)) != len(graph.cost_names):
        _bad('Unique cost identities required')
    available = set()
    for event in graph.declared_events:
        if type(event) is not DeclaredEvent or event.provenance != 'declared_request':
            _bad('Roots must be declared requests, not observed endpoints')
        _name(event.id); _integer(event.requested_ps)
        if event.id in available:
            _bad('Event identities must be unique')
        available.add(event.id)
    if not available or not graph.rules:
        _bad('A nonempty request-to-observation graph required')
    used = set()
    for rule in graph.rules:
        if type(rule) is not CausalRule or type(rule.clock) is not ClockDomain:
            _bad('Exact typed clocked rule required')
        _name(rule.id); _name(rule.observation_semantics); validate_clock(rule.clock)
        if rule.id in available or type(rule.parents) is not tuple or not rule.parents:
            _bad('Unique observations and immutable nonempty causal parents required')
        parent_ids = set()
        for parent in rule.parents:
            if type(parent) is not CausalParent:
                _bad('Exact typed parent required')
            _name(parent.event_id); _integer(parent.fixed_delay_ps)
            if parent.event_id not in available or parent.event_id in parent_ids:
                _bad('Parents must be distinct earlier events; cycles/forward references refused')
            parent_ids.add(parent.event_id)
            if parent.cost is not None:
                _name(parent.cost)
                if parent.cost not in graph.cost_names:
                    _bad('Every cost must be explicitly bound')
                used.add(parent.cost)
        available.add(rule.id)
    if used != set(graph.cost_names):
        _bad('Unused costs must be omitted from this graph contract')


def _assignment(graph, assignment):
    if type(assignment) is not dict or set(assignment) != set(graph.cost_names):
        _bad('One exact complete cost assignment required')
    for value in assignment.values():
        _integer(value)


def _evaluate(graph, assignment):
    times = {event.id: event.requested_ps for event in graph.declared_events}
    rows = []
    for rule in graph.rules:
        bounds = [(p.event_id, times[p.event_id] + p.fixed_delay_ps + (0 if p.cost is None else assignment[p.cost])) for p in rule.parents]
        ready = max(time for _, time in bounds)
        time = rule.clock.edge(ready)
        times[rule.id] = time
        rows.append({'id': rule.id, 'time_ps': time, 'ready_ps': ready,
            'critical_parents': tuple(name for name, bound in bounds if bound == ready),
            'parent_bounds_ps': tuple(bounds), 'clock_rounding_ps': time-ready,
            'observation_semantics': rule.observation_semantics, 'physical_boundary_proven': False})
    return times, tuple(rows)


def diagnose_clocked_graph(graph, assignment):
    """Evaluate declared costs; no observation/reference argument exists."""
    if type(graph) is not ClockedCausalGraph or type(assignment) is not dict:
        _bad('Exact graph and assignment required')
    before = _canonical(graph); before_assignment = _canonical_json(assignment)
    g, costs = copy.deepcopy(graph), copy.deepcopy(assignment)
    _graph(g); _assignment(g, costs)
    if _canonical(g) != before or graph != g or _canonical_json(costs) != before_assignment or _canonical_json(assignment) != before_assignment:
        _bad('Caller input changed while snapshotting/validating')
    times, events = _evaluate(g, costs)
    if _canonical(graph) != before or graph != g or _canonical_json(assignment) != before_assignment:
        _bad('Caller input changed during evaluation')
    return {'role': 'conditional_cost_graph_diagnostic', 'graph_sha256': before,
        'times_ps': times, 'events': events, 'native_admitted': False}


def infer_clocked_graph_costs(domains, training_cases, *, max_boxes=100000):
    """Convert every fully observed training DAG into joint exact constraints.

    Observed predecessor times are used only here. Missing internal boundaries
    must be replaced with a justified compound before inference, not guessed.
    The solution preserves all compatible assignments from all cases at once.
    """
    if type(domains) is not tuple or type(training_cases) is not tuple or not training_cases:
        _bad('Immutable domains and nonempty training cases required')
    for domain in domains:
        if type(domain) is not CostDomain:
            _bad('Exact typed domains required')
    if any(type(case) is not TrainingCase for case in training_cases):
        _bad('Exact typed training cases required')
    before = _canonical_json(([asdict(d) for d in domains], [asdict(c) for c in training_cases]))
    domain_names = tuple(d.name for d in domains)
    domains_copy, cases = copy.deepcopy(domains), copy.deepcopy(training_cases); constraints = []; ids = set()
    for case in cases:
        if type(case) is not TrainingCase or case.role != 'training_only':
            _bad('Reference data must have explicit training-only role')
        _name(case.id); _graph(case.graph)
        if case.id in ids or set(domain_names) != set(case.graph.cost_names):
            _bad('Unique training IDs and common exact cost domain required')
        ids.add(case.id)
        if type(case.observations_ps) is not tuple or any(type(row) is not tuple or len(row) != 2 for row in case.observations_ps):
            _bad('Immutable explicit observation pairs required')
        observations = {}
        for name, time in case.observations_ps:
            _name(name); _integer(time)
            if name in observations:
                _bad('Duplicate training observation')
            observations[name] = time
        if set(observations) != {rule.id for rule in case.graph.rules}:
            raise UnsupportedAnalysis('unobserved_causal_boundary', 'Every modeled operational rule requires one exact training observation')
        parents = {event.id:event.requested_ps for event in case.graph.declared_events} | observations
        for rule in case.graph.rules:
            constraints.append(AlignedMaxConstraint(case.id+'/'+rule.id, observations[rule.id],
                rule.clock.period_ps, rule.clock.phase_ps,
                tuple(MaxParent(parents[p.event_id]+p.fixed_delay_ps, p.cost) for p in rule.parents)))
    solution = solve_aligned_max(domains_copy, tuple(constraints), max_boxes=max_boxes)
    after = _canonical_json(([asdict(d) for d in domains], [asdict(c) for c in training_cases]))
    if after != before or training_cases != cases or domains != domains_copy:
        _bad('Caller training cases changed during inference')
    return solution


def diagnose_clocked_graph_envelope(graph, solution):
    """Exact marginal min/max for the complete retained union of cost boxes.

    Every edge is nonnegative max-plus with monotone clock rounding. Each box's
    low/high corners therefore attain each event's min/max. Correlations between
    boxes are retained; the result does not assert every interior time is feasible.
    """
    if type(graph) is not ClockedCausalGraph or type(solution) is not ConstraintSolution:
        _bad('Exact graph and retained constraint solution required')
    before = _canonical(graph); before_solution = _canonical(solution)
    g, s = copy.deepcopy(graph), copy.deepcopy(solution); _graph(g)
    if _canonical(g) != before or graph != g or _canonical(s) != before_solution or _canonical(solution) != before_solution:
        _bad('Caller input changed while validating')
    if type(s.variable_names) is not tuple or set(s.variable_names) != set(g.cost_names) or len(set(s.variable_names)) != len(s.variable_names):
        _bad('Solution must bind exactly this graph cost domain')
    if type(s.boxes) is not tuple or not s.boxes or s.role != 'mathematical_constraints_not_native_admission':
        raise UnsupportedAnalysis('empty_or_unadmitted_cost_envelope', 'A nonempty mathematical solution is required')
    count = 0; extrema = []
    for box in s.boxes:
        from .aligned_max_constraints import ParameterBox
        if type(box) is not ParameterBox or type(box.bounds_ps) is not tuple or len(box.bounds_ps) != len(s.variable_names):
            _bad('Complete immutable parameter box required')
        for bound in box.bounds_ps:
            if type(bound) is not tuple or len(bound) != 2:
                _bad('Literal interval pair required')
            for value in bound:
                _integer(value)
            if bound[0] > bound[1]:
                _bad('Nonempty interval required')
        count += prod(hi-lo+1 for lo,hi in box.bounds_ps)
        low = dict(zip(s.variable_names, (lo for lo,hi in box.bounds_ps)))
        high = dict(zip(s.variable_names, (hi for lo,hi in box.bounds_ps)))
        extrema.append((_evaluate(g, low)[0], _evaluate(g, high)[0]))
    for i,a in enumerate(s.boxes):
        for b in s.boxes[i+1:]:
            if not any(ahi < blo or bhi < alo for (alo,ahi),(blo,bhi) in zip(a.bounds_ps,b.bounds_ps)):
                _bad('Retained boxes must be disjoint')
    if type(s.assignment_count) is not int or count != s.assignment_count:
        _bad('Solution assignment count must equal complete disjoint-box volume')
    bounds = {rule.id: (min(lo[rule.id] for lo,hi in extrema), max(hi[rule.id] for lo,hi in extrema)) for rule in g.rules}
    if _canonical(graph) != before or graph != g or _canonical(solution) != before_solution or solution != s:
        _bad('Caller input changed during envelope evaluation')
    return {'role':'conditional_complete_cost_envelope', 'graph_sha256':before,
        'bounds_ps':bounds, 'retained_boxes':len(s.boxes), 'retained_assignments':count,
        'marginal_extrema_exact':True, 'continuous_interior_coverage_claimed':False, 'native_admitted':False}


def predict_native_clocked_graph(*args, **kwargs):
    raise UnsupportedAnalysis('unadmitted_native_clocked_graph', 'Program/resource/boundary/cost/holdout certificates are not admitted')
