"""Finite dense-packet scheduling witnesses for the compiled Analyzer bridge.

Only inter-destination interleaving is changed. Every header, payload, TLAST and
per-source/per-destination packet sequence is preserved. This is one candidate
family, not an SSA/e-graph optimizer or a cross-compile equivalence proof.
"""
from collections import defaultdict
from dataclasses import dataclass
import csv
import io
from pathlib import Path

from .compiled_analyzer import (strict_json, read_ref, file_ref, save_new,
                                make_candidate, seal, identity)


@dataclass(frozen=True)
class PacketRule:
    id: str
    version: int
    order: object


class PacketRuleRegistry:
    def __init__(self): self.entries = {}
    def register(self, rule):
        if rule.id in self.entries: raise ValueError('duplicate packet rule ID')
        self.entries[rule.id] = rule


def dense_packets(raw):
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig')),strict=True))
    if not rows or rows[0] != ['CMD','D','TLAST','TKEEP']: raise ValueError('word32 CSV header required')
    packets, current = [], []
    for row in rows[1:]:
        if (len(row) != 4 or row[0] != 'DATA' or row[2] not in ('0','1') or row[3] != '-1' or
                not row[1].isdigit() or str(int(row[1])) != row[1] or not 0 <= int(row[1]) < 2**32):
            raise ValueError('packet schedule generation requires complete dense DATA, without STALL')
        if not current and row[2] != '0': raise ValueError('packet header cannot carry TLAST')
        current.append(row)
        if row[2] == '1': packets.append(current); current = []
    if current or not packets: raise ValueError('complete nonempty packets required')
    return packets


def packet_signature(packets):
    groups = defaultdict(list)
    for packet in packets: groups[str(int(packet[0][1]) & 31)].append(identity(packet))
    return dict(groups)


def _grouped(packets, reverse=False):
    groups = defaultdict(list)
    for packet in packets: groups[int(packet[0][1]) & 31].append(packet)
    return [p for key in sorted(groups,reverse=reverse) for p in groups[key]]


def _round_robin(packets):
    groups = defaultdict(list)
    for packet in packets: groups[int(packet[0][1]) & 31].append(packet)
    return [groups[key][i] for i in range(max(map(len,groups.values()))) for key in sorted(groups) if i < len(groups[key])]


def default_packet_rules():
    registry = PacketRuleRegistry()
    registry.register(PacketRule('packet.reference',1,lambda p:p))
    registry.register(PacketRule('packet.round_robin',1,_round_robin))
    registry.register(PacketRule('packet.group_destinations_ascending',1,_grouped))
    registry.register(PacketRule('packet.group_destinations_descending',1,lambda p:_grouped(p,True)))
    return registry


def schedule_candidates(request_ref, output, *, registry=None):
    request_ref = file_ref(request_ref) if not isinstance(request_ref,dict) else request_ref
    request = strict_json(read_ref(request_ref))
    if request.get('schema') != 'aie-generic-analyzer-request-v1' or request.get('mode') != 'predict':
        raise ValueError('normal compiled prediction request required')
    registry = registry or default_packet_rules()
    if not 1 <= len(registry.entries) <= 16: raise ValueError('finite 1..16 rule registry required')
    sources = {name:dense_packets(read_ref(ref)) for name,ref in request['source_csv'].items()}
    signatures = {name:packet_signature(packets) for name,packets in sources.items()}
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    candidates, seen, duplicates = [], {}, []
    for rule in registry.entries.values():
        reordered = {name:rule.order(list(packets)) for name,packets in sources.items()}
        if {name:packet_signature(p) for name,p in reordered.items()} != signatures:
            raise ValueError('packet rule changes payload or destination FIFO order: ' + rule.id)
        content = {}
        for name,packets in reordered.items():
            stream = io.StringIO(newline=''); writer = csv.writer(stream,lineterminator='\n')
            writer.writerow(['CMD','D','TLAST','TKEEP']);writer.writerows(row for packet in packets for row in packet)
            content[name] = stream.getvalue()
        key = identity(content)
        if key in seen:
            duplicates.append({'rule_id':rule.id,'retained_candidate_id':seen[key]});continue
        case = output/f'candidate-{len(candidates):02d}';case.mkdir()
        refs = {}
        for i,(name,text) in enumerate(sorted(content.items())):
            path = case/f'input-{i}.csv';path.write_text(text,encoding='utf-8',newline='\n');refs[name]=file_ref(path)
        changed = {**request,'source_csv':refs}
        ref = save_new(case/'predict-request.json',changed)
        candidate = make_candidate(ref,label=rule.id,policy='dense_packet_interleaving')
        save_new(case/'candidate.json',candidate);candidates.append(candidate);seen[key]=candidate['id']
    report = seal({'schema':'aie-optimizer-packet-schedule-space-v1','candidates':candidates,
        'source_request_ref':request_ref,'source_sequence_fingerprint':identity(signatures),
        'rules':[{'id':r.id,'version':r.version} for r in registry.entries.values()],
        'duplicates':duplicates,'new_compilations':0,'new_cost_fits':0,
        'comparison_contract':'same compiled program and input values; dense inter-destination scheduling is an allowed decision',
        'scope':'one finite scheduling family; not general structural optimization or measured speedup'})
    read_ref(request_ref)
    for ref in request['source_csv'].values(): read_ref(ref)
    save_new(output/'space.json',report)
    return report
