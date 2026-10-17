"""Resolve component costs by explicit semantic applicability, never scale.

``compile_identity`` names the compiled *component* (body, resource interaction,
clock/unit/boundary semantics), not necessarily its containing program. A legacy
aggregate component can deliberately use a whole-geometry identity. The caller
must bind the containing program's compiler evidence separately. Resolution is
cost availability under a declared contract, not native accuracy admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


class CostRegistryRefusal(ValueError):
    """Malformed or untrustworthy registry/signature; no cost is returned."""


SIGNATURE_SCHEMA = 'aie-component-signature-v1'
REGISTRY_SCHEMA = 'aie-component-cost-registry-v1'
FAMILY_SCHEMA = 'aie-correlated-component-cost-family-v1'
DIMENSIONS = ('architecture', 'clocks', 'units', 'boundaries', 'operations',
              'interactions', 'compile_identity')
SIGNATURE_FIELDS = {'schema', 'component_id', 'mechanism', *DIMENSIONS}
ENTRY_FIELDS = {'cost_id', 'mechanism', 'applicability', 'cost_family',
                'evidence_refs', 'discrepancy'}
FORBIDDEN_KEYS = {'actor_count', 'core_count', 'nactors', 'n_actors', 'case_id',
                  'caseid', 'observed_owner', 'target_trace', 'future_timestamp',
                  'holdout_fit', 'ho_fit', 'per_case_residual'}


def _need(condition, message):
    if not condition:
        raise CostRegistryRefusal(message)


def _name(value, label):
    _need(type(value) is str and bool(value), label + ': nonempty string required')


def _json(value, *, depth=0):
    _need(depth <= 24, 'Finite JSON attribute depth exceeded')
    if type(value) is dict:
        for key, item in value.items():
            _name(key, 'attribute key')
            _need(key.lower() not in FORBIDDEN_KEYS, 'Scale/case dispatch or target-observation attribute forbidden: ' + key)
            _json(item, depth=depth + 1)
    elif type(value) is list:
        for item in value:
            _json(item, depth=depth + 1)
    else:
        _need(value is None or type(value) in (str, int, float, bool), 'Literal finite JSON attributes required')
        _need(type(value) is not float or math.isfinite(value), 'Nonfinite cost/attribute forbidden')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def component_identity(compiled_component_attributes):
    """Canonical structural identity; callers supply verified component facts.

    The input excludes case/scale dispatch fields. It should include the body,
    port operations, banks/ownership interactions and relevant boundaries. This
    function hashes facts; it does not prove that an extractor supplied them.
    """
    _need(type(compiled_component_attributes) is dict and compiled_component_attributes,
          'Nonempty verified component attribute object required')
    _json(compiled_component_attributes)
    return hashlib.sha256(_canonical(compiled_component_attributes).encode()).hexdigest()


def _pin(ref):
    _need(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}, 'Exact pinned cost/evidence reference required')
    _need(type(ref['path']) is str and Path(ref['path']).is_absolute(), 'Absolute reference path required')
    _need(type(ref['bytes']) is int and ref['bytes'] >= 1 and type(ref['sha256']) is str
          and re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None, 'Literal length and SHA256 required')


def _verify(ref):
    _pin(ref)
    raw = Path(ref['path']).read_bytes()
    _need(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'],
          'Pinned component cost/evidence changed: ' + ref['path'])
    return raw


def _evidence(rows, *, train_required=False):
    _need(type(rows) is list, 'Explicit evidence-reference list required')
    for row in rows:
        _need(type(row) is dict and set(row) == {'role', 'ref'}
              and row['role'] in ('TRAIN', 'compatibility_proof', 'architecture_reference', 'software_fixture'),
              'Only declared TRAIN/compatibility/architecture/software evidence roles allowed; HO fit refused')
        _pin(row['ref'])
    _need(not train_required or any(row['role'] == 'TRAIN' for row in rows),
          'TRAIN-calibrated costs require pinned TRAIN evidence')


def _signature(signature):
    _need(type(signature) is dict and set(signature) == SIGNATURE_FIELDS
          and signature['schema'] == SIGNATURE_SCHEMA, 'Exact versioned component signature required')
    for name in ('component_id', 'mechanism', 'architecture', 'compile_identity'):
        _name(signature[name], name)
    for name in ('clocks', 'units', 'boundaries', 'operations', 'interactions'):
        _need(type(signature[name]) is dict, name + ': explicit JSON attribute object required')
        _json(signature[name])


def _registry(registry):
    _need(type(registry) is dict and set(registry) == {'schema', 'registry_id', 'supported_mechanisms', 'entries'}
          and registry['schema'] == REGISTRY_SCHEMA, 'Exact versioned cost registry required')
    _name(registry['registry_id'], 'registry_id')
    mechanisms = registry['supported_mechanisms']
    _need(type(mechanisms) is list and all(type(item) is str and item for item in mechanisms)
          and len(set(mechanisms)) == len(mechanisms), 'Unique supported mechanism list required')
    _need(type(registry['entries']) is list, 'Explicit registry entry list required')
    seen, refs = set(), []
    for entry in registry['entries']:
        _need(type(entry) is dict and set(entry) == ENTRY_FIELDS, 'Exact cost entry fields required')
        _name(entry['cost_id'], 'cost_id')
        _need(entry['cost_id'] not in seen and entry['mechanism'] in mechanisms, 'Unique cost ID and supported mechanism required')
        seen.add(entry['cost_id'])
        domains = entry['applicability']
        _need(type(domains) is dict and set(domains) == set(DIMENSIONS), 'Every applicability dimension must be explicit')
        for name, rule in domains.items():
            _need(type(rule) is dict and rule.get('kind') in ('exact', 'one_of'), 'Only exact/one_of applicability; no wildcard')
            field = 'value' if rule['kind'] == 'exact' else 'values'
            _need(set(rule) == {'kind', field}, 'Exact applicability rule fields required')
            _json(rule[field])
            if field == 'values':
                _need(type(rule[field]) is list and rule[field]
                      and len({_canonical(value) for value in rule[field]}) == len(rule[field]), 'Unique finite explicit applicability choices required')
        family = entry['cost_family']
        _need(type(family) is dict and set(family) == {'schema', 'origin', 'model_refs', 'inline_models', 'correlation_group'}
              and family['schema'] == FAMILY_SCHEMA and family['origin'] in ('TRAIN_calibrated', 'software_conditional'),
              'Explicit correlated TRAIN/software cost family required')
        _name(family['correlation_group'], 'correlation_group')
        _need(type(family['model_refs']) is list and type(family['inline_models']) is list
              and (family['model_refs'] or family['inline_models']), 'Nonempty complete cost family required')
        for ref in family['model_refs']:
            _pin(ref)
            refs.append(ref)
        _json(family['inline_models'])
        for model in family['inline_models']:
            _need(type(model) is dict and set(model) == {'model_id', 'parameters'}
                  and type(model['parameters']) is dict, 'Inline model is a complete model_id/parameters assignment')
            _name(model['model_id'], 'model_id')
        _need(len({model['model_id'] for model in family['inline_models']}) == len(family['inline_models']), 'Unique inline model IDs required')
        _evidence(entry['evidence_refs'], train_required=family['origin'] == 'TRAIN_calibrated')
        refs += [row['ref'] for row in entry['evidence_refs']]
        if any(rule['kind'] == 'one_of' and len(rule['values']) > 1 for rule in domains.values()):
            _need(any(row['role'] in ('compatibility_proof', 'software_fixture') for row in entry['evidence_refs']),
                  'Expanded compatibility domain requires pinned compatibility/fixture evidence')
        discrepancy = entry['discrepancy']
        _need(type(discrepancy) is dict and set(discrepancy) == {'status', 'bound_ps', 'evidence_refs'}
              and discrepancy['status'] in ('uncalibrated', 'calibrated'), 'Separate explicit model discrepancy required')
        _evidence(discrepancy['evidence_refs'])
        refs += [row['ref'] for row in discrepancy['evidence_refs']]
        if discrepancy['status'] == 'uncalibrated':
            _need(discrepancy['bound_ps'] is None and not discrepancy['evidence_refs'],
                  'Uncalibrated discrepancy must have unknown bound, never a promised error box')
        else:
            _need(type(discrepancy['bound_ps']) is int and discrepancy['bound_ps'] >= 0
                  and any(row['role'] == 'TRAIN' for row in discrepancy['evidence_refs']), 'Discrepancy bound needs separate pinned TRAIN evidence')
    return refs


def resolve_cost(component_signature, registry):
    """Return all compatible correlated families with explicit reuse proof.

    Known mechanism but mismatched component domain returns needs_measurement.
    An unimplemented mechanism returns unsupported. All matched alternatives
    are retained; no representative, independent parameter flattening, physical
    FIFO interpretation, ranking or native-error guarantee is introduced.
    """
    _signature(component_signature)
    refs = _registry(registry)
    for ref in refs:
        _verify(ref)
    result = _resolve_validated(component_signature, registry)
    for ref in refs:
        _verify(ref)
    return result


def _resolve_validated(component_signature, registry):
    """Pure lookup after the caller validates and pins the whole registry."""
    matches, rejected = [], []
    mechanism = component_signature['mechanism']
    for entry in registry['entries']:
        if entry['mechanism'] != mechanism:
            continue
        proof = []
        for dimension in DIMENSIONS:
            rule = entry['applicability'][dimension]
            expected = [rule['value']] if rule['kind'] == 'exact' else rule['values']
            passed = any(_canonical(component_signature[dimension]) == _canonical(value) for value in expected)
            proof.append({'dimension': dimension, 'rule': deepcopy(rule), 'signature_value': deepcopy(component_signature[dimension]), 'matched': passed})
        row = {'cost_id': entry['cost_id'], 'applicability_proof': proof}
        if all(item['matched'] for item in proof):
            row.update(cost_family=deepcopy(entry['cost_family']), evidence_refs=deepcopy(entry['evidence_refs']),
                       model_discrepancy=deepcopy(entry['discrepancy']))
            matches.append(row)
        else:
            row['mismatched_dimensions'] = [item['dimension'] for item in proof if not item['matched']]
            rejected.append(row)
    status = ('reused' if matches else 'needs_measurement' if mechanism in registry['supported_mechanisms'] else 'unsupported')
    return {'schema': 'aie-component-cost-resolution-v1', 'status': status,
            'component_id': component_signature['component_id'], 'component_signature': deepcopy(component_signature),
            'registry_id': registry['registry_id'], 'compatible_costs': matches, 'rejected_domains': rejected,
            'reason': {'reused': 'Explicit compiled component domain matched; all compatible families retained',
                       'needs_measurement': 'Mechanism is supported but no cost family covers this component domain',
                       'unsupported': 'Component mechanism is not implemented by this registry'}[status],
            'parameter_family_envelope': {'available_for_caller_evaluation': bool(matches),
                'meaning': 'Conditional complete correlated parameter families; no model-discrepancy guarantee',
                'family_ids': [row['cost_id'] for row in matches], 'native_error_band': False},
            'model_discrepancy': {'status': 'uncalibrated' if not matches or any(row['model_discrepancy']['status'] == 'uncalibrated' for row in matches) else 'calibrated_component_only',
                                  'global_error_bound_ps': None, 'global_ranking_guarantee': False},
            'all_compatible_families_retained': True, 'identity_scope': 'compiled_component_not_total_program_scale',
            'source_and_cost_pins_verified_before_after': True, 'timing_prediction': None,
            'numeric_native_accuracy_admission': False, 'physical_FIFO_capacity_admission': False,
            'new_fit_count': 0, 'target_observations_used': False}


def resolve(component_signature, registry):
    """Public alias for the generic resolver."""
    return resolve_cost(component_signature, registry)


def resolve_program_costs(component_signatures, registry):
    """Resolve a finite component collection without a core-count dispatcher."""
    _need(type(component_signatures) is list and component_signatures, 'Nonempty component signature list required')
    ids = [signature.get('component_id') if type(signature) is dict else None for signature in component_signatures]
    _need(all(type(item) is str and item for item in ids) and len(set(ids)) == len(ids), 'Unique component IDs required')
    # Shared evidence and complete families belong to the program query. Read
    # them once before/after the batch, rather than once per resource role.
    # Every signature still receives exactly the same applicability checks.
    for signature in component_signatures:
        _signature(signature)
    refs = _registry(registry)
    refs = list({(r['path'], r['bytes'], r['sha256']): r for r in refs}.values())
    for ref in refs:
        _verify(ref)
    rows = [_resolve_validated(signature, registry) for signature in component_signatures]
    for ref in refs:
        _verify(ref)
    missing = [row['component_id'] for row in rows if row['status'] == 'needs_measurement']
    unsupported = [row['component_id'] for row in rows if row['status'] == 'unsupported']
    groups = {}
    for row in rows:
        for cost in row['compatible_costs']:
            groups.setdefault(cost['cost_family']['correlation_group'], []).append({'component_id': row['component_id'], 'cost_id': cost['cost_id']})
    return {'schema': 'aie-program-component-cost-resolution-v1',
            'status': 'unsupported' if unsupported else 'needs_measurement' if missing else 'reused',
            'components': rows, 'needs_measurement_components': missing, 'unsupported_components': unsupported,
            'correlation_groups': groups, 'correlation_groups_require_joint_caller_assignment': True,
            'timing_prediction': None, 'numeric_native_accuracy_admission': False,
            'global_model_discrepancy_bound_ps': None, 'global_ranking_guarantee': False,
            'new_fit_count': 0, 'target_observations_used': False}


def load_cost_family(resolution, cost_id):
    """Read exact compatible referenced families; never alter their geometry.

    The result preserves source-file identity and schema. It does not select a
    model, construct an independent cross-product or rewrite legacy IO costs.
    """
    _need(type(resolution) is dict and resolution.get('schema') == 'aie-component-cost-resolution-v1'
          and resolution.get('status') == 'reused', 'A successful component resolution is required')
    costs = [row for row in resolution['compatible_costs'] if row['cost_id'] == cost_id]
    _need(len(costs) == 1, 'Exactly one resolved cost ID required')
    row = costs[0]
    refs = row['cost_family']['model_refs'] + [item['ref'] for item in row['evidence_refs']]
    for ref in refs:
        _verify(ref)
    documents = [{'ref': deepcopy(ref), 'document': json.loads(_verify(ref).decode('utf-8'))}
                 for ref in row['cost_family']['model_refs']]
    for ref in refs:
        _verify(ref)
    return {'schema': 'aie-loaded-component-cost-family-v1', 'cost_id': cost_id,
            'correlation_group': row['cost_family']['correlation_group'], 'origin': row['cost_family']['origin'],
            'referenced_documents': documents, 'inline_models': deepcopy(row['cost_family']['inline_models']),
            'model_discrepancy': deepcopy(row['model_discrepancy']), 'legacy_geometry_rewritten': False,
            'native_accuracy_admission': False}


def reconstruct_legacy_request(resolution, cost_id, source_csv):
    """Rebuild an exact legacy prospective request with new source pins only.

    This bridge deliberately keeps every old geometry/calibration/wrapper/core
    field. It is not an N-actor adapter and cannot admit a changed compilation.
    The request is one explicitly stored legal family member, not the full
    family's midpoint trajectory or a newly calibrated model.
    """
    _need(resolution.get('component_signature', {}).get('mechanism') == 'legacy_fixed_compile_cascade_bundle',
          'Only the explicit legacy aggregate profile can reconstruct its request')
    loaded = load_cost_family(resolution, cost_id)
    templates = [row['document'] for row in loaded['referenced_documents']
                 if type(row['document']) is dict and row['document'].get('schema') == 'aie-backpressured-cascade-request-v1']
    _need(len(templates) == 1, 'Exactly one pinned original legacy request template required')
    template = templates[0]
    _need(type(source_csv) is dict and set(source_csv) == set(template['source_csv']),
          'Prospective source set must equal the original compiled profile')
    for ref in source_csv.values():
        _verify(ref)
    result = deepcopy(template)
    result['source_csv'] = deepcopy(source_csv)
    # Nested legacy references stay byte-for-byte; the normal legacy entry point
    # still owns every existing geometry, extent and applicability guard.
    for key in ('plans', 'io_geometry', 'io_calibration', 'wrapper_contract', 'effective_fifo_parameters'):
        _verify(result[key])
    for ref in source_csv.values():
        _verify(ref)
    return result
