"""Exact bounded integer constraints for clock-aligned max-plus costs.

This mathematical inference layer admits no native profile. A solution is a
disjoint union of parameter boxes, never a hand-picked best fit or a truncated
Cartesian search. Every observation contributes upper bounds plus a clause
requiring at least one causal parent to attain the rounded maximum.
"""
from dataclasses import dataclass
from math import prod
from analysis_errors import InvalidAnalysisInput, UnsupportedAnalysis


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidAnalysisInput('invalid_max_constraint_integer', label+' must be a literal integer')


def _name(value):
    if type(value) is not str or not value.strip():
        raise InvalidAnalysisInput('invalid_max_constraint_name', 'Explicit variable/constraint identity required')


@dataclass(frozen=True)
class CostDomain:
    name: str
    minimum_ps: int
    maximum_ps: int


@dataclass(frozen=True)
class MaxParent:
    base_ps: int
    variable: str | None = None


@dataclass(frozen=True)
class AlignedMaxConstraint:
    label: str
    observed_ps: int
    period_ps: int
    phase_ps: int
    parents: tuple[MaxParent, ...]


@dataclass(frozen=True)
class ParameterBox:
    # Names are shared through ConstraintSolution.variable_names.
    bounds_ps: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ConstraintSolution:
    variable_names: tuple[str, ...]
    boxes: tuple[ParameterBox, ...]
    assignment_count: int
    constraints: int
    unobserved_variables: tuple[str, ...]
    role: str = 'mathematical_constraints_not_native_admission'

    def contains(self, assignment):
        if type(assignment) is not dict or set(assignment) != set(self.variable_names):
            raise InvalidAnalysisInput('invalid_parameter_assignment', 'Complete exact variable names required')
        values=[]
        for name in self.variable_names:
            value=assignment[name];_int(value,name);values.append(value)
        return any(all(lo<=x<=hi for x,(lo,hi) in zip(values,box.bounds_ps)) for box in self.boxes)


def _merge_adjacent(boxes):
    """Merge only exact rectangular unions; never fill a missing assignment."""
    boxes=sorted(set(boxes))
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(boxes):
            for j in range(i+1,len(boxes)):
                b=boxes[j];different=[k for k,(x,y) in enumerate(zip(a,b)) if x!=y]
                if len(different)!=1:continue
                k=different[0];alo,ahi=a[k];blo,bhi=b[k]
                if ahi+1==blo or bhi+1==alo:
                    merged=list(a);merged[k]=(min(alo,blo),max(ahi,bhi))
                    boxes=[box for index,box in enumerate(boxes) if index not in (i,j)]+[tuple(merged)]
                    boxes.sort();changed=True;break
            if changed:break
    return boxes


def solve_aligned_max(domains, constraints, *, max_boxes=100000):
    """Solve Y=edge(max(parent_time+cost)) exactly on a finite ps domain.

    Y-p < max(parent+cost) <= Y. The right inequality reduces each
    variable's upper bound. The left becomes an OR of lower thresholds.
    Repeated variable branches retain their strongest (latest) parent.
    A complexity budget refuses the entire inference instead of dropping
    hypotheses. Finite bounds must be registered by the calling study.
    """
    _int(max_boxes,'max_boxes',1)
    if type(domains) is not tuple or type(constraints) is not tuple:
        raise InvalidAnalysisInput('mutable_max_constraints', 'Immutable domains and constraints required')
    names=[];bounds=[]
    for domain in domains:
        if not isinstance(domain,CostDomain):
            raise InvalidAnalysisInput('untyped_cost_domain', 'Typed finite cost domain required')
        _name(domain.name);_int(domain.minimum_ps,domain.name);_int(domain.maximum_ps,domain.name)
        if domain.minimum_ps>domain.maximum_ps or domain.name in names:
            raise InvalidAnalysisInput('invalid_cost_domain', 'Unique ordered nonempty domain required')
        names.append(domain.name);bounds.append((domain.minimum_ps,domain.maximum_ps))
    positions={name:index for index,name in enumerate(names)}
    labels=set();clauses=[];used=set();impossible=False
    for constraint in constraints:
        if not isinstance(constraint,AlignedMaxConstraint):
            raise InvalidAnalysisInput('untyped_aligned_constraint', 'Typed aligned observation required')
        _name(constraint.label)
        if constraint.label in labels:
            raise InvalidAnalysisInput('duplicate_aligned_constraint', 'Each measurement identity is unique')
        labels.add(constraint.label)
        _int(constraint.observed_ps,'observation');_int(constraint.period_ps,'period',1);_int(constraint.phase_ps,'phase')
        if constraint.phase_ps>=constraint.period_ps or constraint.observed_ps<constraint.phase_ps or (constraint.observed_ps-constraint.phase_ps)%constraint.period_ps:
            raise InvalidAnalysisInput('unaligned_max_observation', 'Observation must lie on its declared clock grid')
        if type(constraint.parents) is not tuple or not constraint.parents:
            raise InvalidAnalysisInput('missing_max_parents', 'Complete immutable causal parents required')
        parent_by_variable={};constant=None
        for parent in constraint.parents:
            if not isinstance(parent,MaxParent):
                raise InvalidAnalysisInput('untyped_max_parent', 'Typed causal parent required')
            _int(parent.base_ps,'parent time')
            if parent.variable is None:
                constant=parent.base_ps if constant is None else max(constant,parent.base_ps)
            else:
                _name(parent.variable)
                if parent.variable not in positions:
                    raise InvalidAnalysisInput('unbound_max_variable', 'Parent uses an undeclared cost')
                used.add(parent.variable)
                parent_by_variable[parent.variable]=max(parent.base_ps,parent_by_variable.get(parent.variable,0))
        lower=constraint.observed_ps-constraint.period_ps+1
        if constant is not None and constant>constraint.observed_ps:impossible=True
        clause=[]
        for name,base in parent_by_variable.items():
            index=positions[name];lo,hi=bounds[index]
            bounds[index]=(lo,min(hi,constraint.observed_ps-base))
            clause.append((index,lower-base))
        if constant is None or constant<lower:
            if not clause:impossible=True
            clauses.append(tuple(sorted(clause)))
    if any(lo>hi for lo,hi in bounds):impossible=True
    boxes=[] if impossible else [tuple(bounds)]
    for clause in clauses:
        next_boxes=[]
        for original in boxes:
            remainder=list(original)
            # Partition by the first satisfied parent. Its siblings constrain
            # earlier parents false, so the retained boxes remain disjoint.
            for index,threshold in clause:
                lo,hi=remainder[index]
                if hi<threshold:continue
                if lo>=threshold:
                    next_boxes.append(tuple(remainder));break
                accepted=list(remainder);accepted[index]=(threshold,hi)
                next_boxes.append(tuple(accepted))
                remainder[index]=(lo,threshold-1)
            if len(next_boxes)>max_boxes:
                raise UnsupportedAnalysis('max_constraint_box_budget_exceeded', 'Inference budget exceeded; no hypothesis may be truncated')
        boxes=_merge_adjacent(next_boxes)
        if not boxes:break
    result_boxes=tuple(ParameterBox(box) for box in sorted(boxes))
    count=sum(prod(hi-lo+1 for lo,hi in box.bounds_ps) for box in result_boxes)
    return ConstraintSolution(tuple(names),result_boxes,count,len(constraints),tuple(name for name in names if name not in used))


def predict_native_from_constraints(*args, **kwargs):
    raise UnsupportedAnalysis('unadmitted_constraint_profile', 'Mathematical consistency is not native profile admission')
