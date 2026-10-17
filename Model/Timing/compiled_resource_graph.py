"""Generic, static AIE compiler resource graph; no cost or timing admission.

Input artifacts are pinned files or explicitly labelled software fixtures. All
router nets participate in the physical resource occupancy proof, including nets
which could not be bound to a compiler connection. Packet branches sharing one
source are represented once; independent source sharing is never called private.
Actor count, names, placement and number of slots are data, not case dispatch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .cascade_compiled_evidence import parse_dma_control

SCHEMA = "aie-compiled-resource-graph-v1"
INPUT_SCHEMA = "aie-compiled-resource-inputs-v1"
REQUIRED = frozenset({"compiler_report", "router_soln", "driver_control", "driver_config",
                      "simulator_config"})
OPTIONAL = frozenset({"manifest", "compile_result", "packet_switching_report"})


class CompiledResourceGraphError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


def need(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise CompiledResourceGraphError(code, detail)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    need(type(value) is int and value >= minimum, "invalid_integer", name)
    return value


def name(value: Any, detail: str) -> str:
    need(type(value) is str and bool(value) and value == value.strip() and not any(c in value for c in "\r\n\0"),
         "invalid_identity", detail)
    return value


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise CompiledResourceGraphError("nonfinite_json", str(error)) from error


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_ref(path: str | Path) -> dict:
    p = Path(path).absolute()
    raw = p.read_bytes()
    return {"path": str(p), "bytes": len(raw), "sha256": sha(raw)}


def _read_inputs(value: dict) -> tuple[dict, list]:
    need(type(value) is dict and set(value) == {"schema", "provenance_kind", "artifacts"},
         "input_fields", "schema, provenance_kind, artifacts required")
    need(value["schema"] == INPUT_SCHEMA, "input_schema", INPUT_SCHEMA)
    actual = value["provenance_kind"] == "actual_compiler_artifacts"
    need(actual or value["provenance_kind"] == "software_fixture_not_native_compilation",
         "input_provenance", "Actual compiler artifacts or labelled software fixture required")
    artifacts = value["artifacts"]
    need(type(artifacts) is dict and REQUIRED <= artifacts.keys() <= REQUIRED | OPTIONAL,
         "artifact_roles", "Five raw compiler roles required; packet report, manifest and compile_result are optional roles subject to mechanism validation")
    data, pins = {}, []
    for role, ref in sorted(artifacts.items()):
        need(type(ref) is dict, "artifact_ref", role)
        if set(ref) == {"data"}:
            need(not actual, "unbound_actual_artifact", role)
            payload = ref["data"]
            raw = payload.encode("utf-8") if role == "driver_control" and type(payload) is str else canonical(payload)
            pin = {"role": role, "kind": "inline_software_fixture", "bytes": len(raw), "sha256": sha(raw)}
            data[role] = copy.deepcopy(payload)
        else:
            need(set(ref) == {"path", "bytes", "sha256"} and type(ref["path"]) is str,
                 "artifact_ref", role)
            integer(ref["bytes"], role + ".bytes")
            need(type(ref["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is not None,
                 "artifact_hash", role)
            got = file_ref(ref["path"])
            need(got["bytes"] == ref["bytes"] and got["sha256"] == ref["sha256"], "artifact_identity_mismatch", role)
            text = Path(ref["path"]).read_text(encoding="utf-8-sig")
            try:
                data[role] = text if role == "driver_control" else json.loads(text)
            except json.JSONDecodeError as error:
                raise CompiledResourceGraphError("artifact_json", role) from error
            pin = {"role": role, "kind": "pinned_file", **got}
        pins.append(pin)
    need(type(data["driver_control"]) is str, "driver_text", "Generated driver source required")
    for role in data.keys() - {"driver_control"}:
        need(type(data[role]) is dict, "artifact_object", role)
    if "compile_result" in data:
        need(data["compile_result"].get("returncode") == 0, "compile_failed", "Successful compile_result required")
    return data, pins


def _communication_evidence(data: dict) -> dict:
    """Omitting a packet report requires positive circuit/memory evidence.

    This classifies static report roles only. It never implies zero packet or
    adapter service cost, a supported executor, or a physical link width.
    """
    if "packet_switching_report" in data:
        packet, ports = data["packet_switching_report"], data["compiler_report"]["portInstances"]
        records, selectors = 0, {}
        for category in ("Inputs", "Outputs", "Internal"):
            need(type(packet.get(category)) is list, "packet_report_collection", category)
            for record in packet[category]:
                need(type(record) is dict and len(set(record) & {"Input", "Output"}) == 1 and
                     type(record.get("PacketID")) is list and bool(record["PacketID"]),
                     "packet_report_record", category)
                side = "Input" if "Input" in record else "Output"
                need(type(record[side]) is dict and record[side].get("Port") in ports,
                     "packet_report_port", category)
                port = record[side]["Port"]
                for pid in record["PacketID"]: integer(pid, port+".PacketID")
                need(len(set(record["PacketID"])) == len(record["PacketID"]), "packet_id_duplicate", port)
                key = (side, port)
                need(key not in selectors or selectors[key] == record["PacketID"], "packet_id_conflict", port)
                selectors[key] = record["PacketID"]; records += 1
        return {"packet_report": "supplied", "packet_report_records": records, "timing_admission": False}
    compiler, text = data["compiler_report"], data["driver_control"]
    blocks = compiler.get("blockTypes", {})
    mappings = compiler.get("mapping", {}).get("portInstanceMapping", {})
    packet_types = [k for k, b in blocks.items() if "pkt" in str(b.get("type", "")).lower()
                    or any("pkt" in str(p.get("type", "")).lower() for p in b.get("ports", {}).values())]
    packet_mappings = [k for k, m in mappings.items() if "packetID" in m.get("dmaInfo", {})]
    packet_calls = sorted(set(re.findall(r"\b(XAie_(?:StrmPkt\w*|DmaSetPkt\w*))\s*\(", text)))
    need(not packet_types and not packet_mappings and not packet_calls, "packet_report_required",
         "Packet mechanism evidence requires packet_switching_report: " + str({
             "block_types": packet_types, "port_mappings": packet_mappings, "driver_calls": packet_calls}))
    known_types = {"function", "plio", "sharedbuffer"}
    known_ports = {"mem", "stream", "dma", "cascade"}
    need(bool(blocks) and all(b.get("type") in known_types and
         all(p.get("type") in known_ports for p in b.get("ports", {}).values()) for b in blocks.values()),
         "packet_report_absence_unproven", "Unknown compiler communication type cannot establish a circuit/memory-only graph")
    circuit_calls = len(re.findall(r"\bXAie_StrmConnCctEnable\s*\(", text))
    need(circuit_calls > 0 and bool(data["router_soln"]), "packet_report_absence_unproven",
         "Literal driver circuit configuration and complete router inventory required without packet report")
    return {"packet_report": "omitted_verified_circuit_and_memory", "driver_circuit_calls": circuit_calls,
            "basis": "compiler block/port types, DMA selectors, generated driver calls and complete router inventory",
            "packet_service_cost_inferred": False, "timing_admission": False}


def _physical(label: str) -> dict:
    name(label, "router physical port")
    match = re.fullmatch(r"([SM])_(.+)_ch(\d+)_C(\d+)(?:_R(\d+))?", label)
    if not match:
        return {"id": "router-unknown:" + label, "label": label, "kind": "unknown_router_port",
                "physical_key": [label], "supported_structure": False}
    side, kind, channel, column, row = match.groups()
    integer(int(channel), label)
    supported = kind in {"NORTH", "SOUTH", "EAST", "WEST", "SHIM_NORTH", "SHIM_SOUTH", "S2MM_DMA", "MM2S_DMA", "CORE"}
    # Horizontal shim labels occur in actual compiler routing when a PLIO
    # reaches another shim column. Recognize identity only: no AIE-port FIFO,
    # clock, credit or service cost is implied by this structural classification.
    supported |= kind in {"SHIM_EAST", "SHIM_WEST"} and row is None
    supported |= kind in {"MEMTILE_NORTH", "MEMTILE_SOUTH", "MEMTILE_EAST", "MEMTILE_WEST",
                          "MEMTILE_S2MM_DMA", "MEMTILE_MM2S_DMA"} and row is not None
    physical_key = [side, kind, int(channel), int(column), None if row is None else int(row)]
    normalized = f"{side}_{kind}_ch{int(channel)}_C{int(column)}" + ("" if row is None else f"_R{int(row)}")
    return {"id": "router-port:" + normalized, "label": normalized, "kind": "stream_switch_port",
            "physical_key": physical_key, "side": side, "port_kind": kind,
            "channel": int(channel), "column": int(column), "row": None if row is None else int(row),
            "tile_kind": "shim" if row is None else "memtile" if kind.startswith("MEMTILE_") else "aie", "supported_structure": supported}


def _trees(router: dict) -> dict:
    result = {}
    for net_id, raw in router.items():
        name(net_id, "router net")
        need(type(raw) is dict and "tree" in raw and type(raw["tree"]) is list and bool(raw["tree"]),
             "malformed_router_tree", net_id)
        outgoing, incoming, edges = defaultdict(list), defaultdict(list), []
        for edge in raw["tree"]:
            need(type(edge) is list and len(edge) == 2 and all(type(x) is str for x in edge),
                 "malformed_router_edge", net_id)
            left, right = (_physical(x)["label"] for x in edge)
            need(left != right and [left, right] not in edges, "duplicate_or_self_router_edge", net_id)
            edges.append([left, right]); outgoing[left].append(right); incoming[right].append(left)
        nodes = {x for edge in edges for x in edge}
        roots = sorted(nodes - incoming.keys())
        need(len(roots) == 1 and all(len(v) == 1 for v in incoming.values()), "router_not_directed_tree", net_id)
        visited, queue = set(), deque(roots)
        while queue:
            current = queue.popleft()
            need(current not in visited, "router_cycle", net_id)
            visited.add(current); queue.extend(outgoing[current])
        need(visited == nodes, "router_disconnected_or_cycle", net_id)
        result[net_id] = {"edges": edges, "nodes": nodes, "root": roots[0], "outgoing": outgoing}
    need(bool(result), "router_empty", "Complete raw router tree inventory required")
    return result


def _path(tree: dict, start: str, end: str) -> list | None:
    if start not in tree["nodes"] or end not in tree["nodes"]:
        return None
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        queue.extend((next_node, path + [next_node]) for next_node in tree["outgoing"][node])
    return None


def _driver_routing(text: str, row_origin: int, shim_row: int | None, device_config: dict | None = None) -> dict:
    """Recover literal switch/selector/header fields, not arbiter timing."""
    result = {"circuits": [], "packet_slave_rules": [], "packet_master_ports": [], "unsupported_calls": []}
    loc = r"&DevInst,\s*XAie_TileLoc\((\d+),\s*(\d+)\),\s*"
    def physical(column, physical_row, side, direction, channel):
        column, physical_row, channel = int(column), int(physical_row), int(channel)
        if device_config is not None:
            if "num_columns" in device_config:
                need(column < device_config["num_columns"], "driver_switch_column", str(column))
            if "num_rows" in device_config:
                need(physical_row < device_config["num_rows"], "driver_switch_row", str(physical_row))
        if physical_row == shim_row:
            label = f"{side}_SHIM_{direction}_ch{channel}_C{column}"
        elif device_config is not None and device_config["mem_tile_row_start"] <= physical_row < device_config["mem_tile_row_start"] + device_config["mem_tile_num_rows"]:
            kind = ("MM2S_DMA" if side == "S" else "S2MM_DMA") if direction == "DMA" else direction
            label = f"{side}_MEMTILE_{kind}_ch{channel}_C{column}_R{physical_row-device_config['mem_tile_row_start']}"
        else:
            need(physical_row >= row_origin, "driver_switch_row", str(physical_row))
            kind = ("MM2S_DMA" if side == "S" else "S2MM_DMA") if direction == "DMA" else direction
            label = f"{side}_{kind}_ch{channel}_C{column}_R{physical_row-row_origin}"
        return _physical(label)["id"]
    specifications = [
        ("XAie_StrmConnCctEnable", loc + r"([A-Z_]+),\s*(\d+),\s*([A-Z_]+),\s*(\d+)\)", "circuits"),
        ("XAie_StrmPktSwSlaveSlotEnable", loc + r"([A-Z_]+),\s*(\d+),\s*(\d+),\s*\{\s*\.PktId\s*=\s*(\d+),\s*\.PktType\s*=\s*(\d+)\s*\},\s*(0x[0-9a-fA-F]+|\d+),\s*(\d+),\s*(\d+)\)", "packet_slave_rules"),
        ("XAie_StrmPktSwMstrPortEnable", loc + r"([A-Z_]+),\s*(\d+),\s*(XAIE_SS_PKT_[A-Z_]+),\s*(\d+),\s*(0x[0-9a-fA-F]+|\d+)\)", "packet_master_ports"),
    ]
    for function, arguments, collection in specifications:
        matches = list(re.finditer(r"\b" + function + r"\(" + arguments, text))
        if len(matches) != len(re.findall(r"\b" + function + r"\s*\(", text)):
            result["unsupported_calls"].append({"function": function, "reason": "Nonliteral/unparsed switch configuration"})
        for match in matches:
            args = match.groups(); column, physical_row = map(int, args[:2])
            record = {"driver_tile": [column, physical_row], "source_line": text.count("\n", 0, match.start()) + 1}
            if collection == "circuits":
                record.update(src_physical_port_id=physical(*args[:2], "S", args[2], args[3]),
                              dst_physical_port_id=physical(*args[:2], "M", args[4], args[5]))
            elif collection == "packet_slave_rules":
                record.update(physical_port_id=physical(*args[:2], "S", args[2], args[3]),
                              slot=int(args[4]), packet_id=int(args[5]), packet_type=int(args[6]), selector_mask=int(args[7], 0),
                              master_select=int(args[8]), arbiter_number=int(args[9]))
            else:
                record.update(physical_port_id=physical(*args[:2], "M", args[2], args[3]),
                              header_mode=args[4], drop_header=True if args[4] == "XAIE_SS_PKT_DROP_HEADER" else
                              False if args[4] == "XAIE_SS_PKT_DONOT_DROP_HEADER" else None,
                              arbiter_number=int(args[5]), master_select_mask=int(args[6], 0), arbiter_service_cost=None)
            result[collection].append(record)
    return result


def _build_compiled_resource_graph(compiler_inputs: dict) -> dict:
    """Build one generic graph from complete pinned raw compiler inputs.

    ``artifacts`` has compiler_report, router_soln, driver_control, driver_config,
    simulator_config, plus packet_switching_report when packet evidence requires
    it, optionally manifest/compile_result.
    Each is ``{path,bytes,sha256}``; ``{data:...}`` is allowed only for explicitly
    labelled software fixtures. Malformed/contradictory evidence raises
    CompiledResourceGraphError. Unknown mechanisms remain structural output with
    unsupported diagnostics and cannot admit timing.
    """
    data, pins = _read_inputs(compiler_inputs)
    compiler = data["compiler_report"]
    need(compiler.get("schema") == "MEGraphSchema-0.4", "compiler_schema", "MEGraphSchema-0.4 required")
    for key in ("blockTypes", "blockInstances", "portInstances", "nets", "mapping"):
        need(type(compiler.get(key)) is dict, "compiler_collection", key)
    types, instances, cp, nets, mapping = (compiler[k] for k in ("blockTypes", "blockInstances", "portInstances", "nets", "mapping"))
    need(type(mapping.get("blockInstanceMapping")) is dict and type(mapping.get("portInstanceMapping")) is dict,
         "compiler_mapping", "block and port mappings required")
    need(set(mapping["portInstanceMapping"]) == set(cp), "compiler_port_mapping_coverage", "Every compiler port must be represented")
    communication_evidence = _communication_evidence(data)
    metadata = data["driver_config"].get("aie_metadata")
    need(type(metadata) is dict and type(metadata.get("driver_config")) is dict, "driver_config", "aie_metadata/driver_config")
    row_origin = integer(metadata["driver_config"].get("aie_tile_row_start"), "aie_tile_row_start")
    shim_row = metadata["driver_config"].get("shim_row")
    if shim_row is not None: integer(shim_row, "shim_row")
    hw_gen = integer(metadata["driver_config"].get("hw_gen"), "hw_gen", 1)
    compiler_device = compiler.get("aie_driver_config")
    if compiler_device is not None:
        need(type(compiler_device) is dict and type(compiler_device.get("hw_gen")) is int and compiler_device["hw_gen"] == hw_gen,
             "hardware_generation_mismatch", "Compiler and driver hw_gen differ")
    simulator = data["simulator_config"].get("SimulationConfig", {})
    if "aiearch" in simulator:
        need(simulator["aiearch"] == {1: "aie", 2: "aie-ml"}.get(hw_gen),
             "architecture_evidence_mismatch", "Simulator architecture must agree with driver hw_gen")
    if "part" in compiler and "phy_device_file" in simulator:
        need(compiler["part"] == simulator["phy_device_file"], "part_evidence_mismatch",
             "Compiler and Simulator physical part differ")
    if hw_gen == 2:
        from .aie_ml_resource_binding import build_aie_ml_resource_graph
        graph = build_aie_ml_resource_graph(compiler_inputs, data, pins)
        graph["communication_evidence"] = communication_evidence
        if communication_evidence.get("packet_report_records", 0):
            graph["diagnostics"].append({"code": "aie_ml_packet_report_semantics_unsupported", "status": "unsupported",
                "detail": "Supplied packet selectors require an AIE-ML packet binding; the static circuit binder does not infer their execution semantics"})
        return graph
    diagnostics = []
    def diagnostic(code, status, detail, **context):
        diagnostics.append({"code": code, "status": status, "detail": detail, **context})
    if hw_gen != 1:
        diagnostic("architecture_semantics_unsupported", "unsupported", "Structure retained; this frontend has only AIE1 coordinate semantics", hw_gen=hw_gen)
    architecture = "AIE1" if hw_gen == 1 else f"hw_gen_{hw_gen}"
    routing_configuration = _driver_routing(data["driver_control"], row_origin, shim_row)
    for unknown in routing_configuration["unsupported_calls"]:
        diagnostic("driver_route_configuration_unsupported", "unsupported", unknown["reason"], function=unknown["function"])
    actor_rows, ports, nodes, resources, connections, routes = [], {}, {}, {}, [], []
    def resource(identity, kind, physical_key, owner=None, **fields):
        row = {"id": identity, "kind": kind, "mechanism": kind, "physical_key": physical_key, **fields}
        if identity not in resources:
            resources[identity] = {**row, "owners": [], "cost": None}
        else:
            existing = resources[identity]
            need(all(existing.get(k) == v for k, v in row.items()), "physical_resource_definition_conflict", identity)
        if owner is not None and owner not in resources[identity]["owners"]:
            resources[identity]["owners"].append(owner)
        return identity
    def tile(value, detail):
        need(type(value) is dict and value.get("tile") in {"aie", "shim"}, "tile_mapping", detail)
        return [integer(value.get("column"), detail + ".column"), integer(value.get("row", 0), detail + ".row")]
    for block_id, instance in instances.items():
        name(block_id, "block id")
        need(type(instance) is dict and instance.get("blockType") in types, "block_type_binding", block_id)
        block = types[instance["blockType"]]
        need(type(block) is dict and (type(block.get("ports")) is dict or
             ("ports" not in block and instance.get("fabric") != "me" and
              not any(p.get("blockInstance") == block_id for p in cp.values()))), "block_type", block_id)
        nodes["block:" + block_id] = {"id": "block:" + block_id, "kind": block.get("type"), "compiler_block_id": block_id}
        if block.get("type") == "function" and instance.get("fabric") == "me":
            core = mapping["blockInstanceMapping"].get(block_id, {}).get("coreInfo")
            need(type(core) is dict, "actor_core_mapping", block_id)
            core_tile = tile(core, block_id)
            core_id = resource(f"core:{core_tile[0]}:{core_tile[1]}", "core", core_tile, block_id)
            actor_rows.append({"id": block_id, "actor_id": block_id, "program_id": instance.get("qualifiedName"),
                               "function": block.get("function"), "source_ref": block.get("srcFile"),
                               "actual_compute_tile": core_tile, "core_resource_id": core_id,
                               "compiler_schedule_field": core.get("schedule"), "input_ports": [], "output_ports": [],
                               "wrapper_semantics": {"status": "needs_measurement", "selected_slot_sequence": None,
                                                     "reason": "Static DMA ring is not dynamic wrapper ACQ/call/REL semantics"}})
    need(bool(actor_rows), "actors_empty", "At least one mapped compute actor required")
    actor_by_id = {a["id"]: a for a in actor_rows}
    for port_id, port in cp.items():
        name(port_id, "port id")
        need(type(port) is dict and port.get("blockInstance") in instances, "port_actor_binding", port_id)
        block_id = port["blockInstance"]
        block = types[instances[block_id]["blockType"]]
        need(port.get("port") in block["ports"], "port_type_binding", port_id)
        info = block["ports"][port["port"]]
        need(info.get("direction") in {"in", "out"} and type(info.get("type")) is str,
             "port_direction_or_type", port_id)
        pmap = mapping["portInstanceMapping"][port_id]
        need(type(pmap) is dict, "port_mapping", port_id)
        index = info.get("parameterIndex")
        if index is not None: integer(index, port_id + ".parameterIndex")
        ports[port_id] = {"id": port_id, "port_id": port_id, "actor_id": block_id if block_id in actor_by_id else None,
                          "block_id": block_id, "name": port.get("portName", info.get("portName")),
                          "qualified_name": port.get("qualifiedName"), "direction": "input" if info["direction"] == "in" else "output",
                          "parameter_index": index, "input_index": index if info["direction"] == "in" else None,
                          "output_index": index if info["direction"] == "out" else None,
                          "type": info["type"], "element_type": info.get("elementType"), "bytes": port.get("blockSize"),
                          "mechanism": "window_dma" if "dmaInfo" in pmap else info["type"],
                          "channel_resource_id": None, "buffer_slots": [], "BD_slots": [], "packet_ids": None}
        if port.get("blockSize") is not None: integer(port["blockSize"], port_id + ".bytes", 1)
        if block_id in actor_by_id:
            actor_by_id[block_id]["input_ports" if info["direction"] == "in" else "output_ports"].append(port_id)
    for actor in actor_rows:
        for direction, collection, index_field in (("input", "input_ports", "input_index"), ("output", "output_ports", "output_index")):
            actor[collection].sort(key=lambda p: (ports[p]["parameter_index"] is None, ports[p]["parameter_index"] or 0, p))
            for index, port_id in enumerate(actor[collection]):
                ports[port_id][index_field] = index
    for net_id, net in nets.items():
        name(net_id, "compiler net id")
        need(type(net) is dict and net.get("srcPort") in ports and net.get("dstPort") in ports,
             "compiler_net_port", net_id)
        src, dst = ports[net["srcPort"]], ports[net["dstPort"]]
        need(net.get("srcInstance") == src["block_id"] and net.get("dstInstance") == dst["block_id"] and
             src["direction"] == "output" and dst["direction"] == "input", "compiler_net_direction", net_id)
        mechanism = "cascade" if src["type"] == dst["type"] == "cascade" else (
            "window_dma" if src["mechanism"] == "window_dma" or dst["mechanism"] == "window_dma" else "packet_or_stream")
        connections.append({"id": net_id, "src_port_id": src["id"], "dst_port_id": dst["id"],
                            "source_actor": src["actor_id"], "target_actor": dst["actor_id"], "mechanism": mechanism,
                            "element_type": src["element_type"], "token_width_bits": None, "token_lanes": None,
                            "route_ids": [], "resource_ids": [], "mechanism_status": "static_bound"})
        if mechanism == "cascade":
            need(src["element_type"] == dst["element_type"], "cascade_element_mismatch", net_id)
            st, dt = (actor_by_id.get(x["actor_id"], {}).get("actual_compute_tile") for x in (src, dst))
            need(st is not None and dt is not None, "cascade_actor_binding", net_id)
            rid = resource("cascade:" + ":".join(map(str, st + dt)), "cascade", st + dt, net_id,
                           element_type=src["element_type"], token_width_bits=None, token_lanes=None)
            connections[-1]["resource_ids"] = [rid]
            diagnostic("cascade_transaction_width_unknown", "needs_measurement", "Element type does not establish dynamic response width/count", connection_id=net_id)
    connection_by_id = {c["id"]: c for c in connections}
    # Parse complete generated driver, checking that no DMA writes/queue calls were ignored.
    try:
        descriptors, queues = parse_dma_control(data["driver_control"])
    except ValueError as error:
        raise CompiledResourceGraphError("driver_descriptor_unsupported", str(error)) from error
    need(len(descriptors) == len(re.findall(r"\bXAie_DmaWriteBd\s*\(", data["driver_control"])) and
         len(queues) == len(re.findall(r"\bXAie_DmaChannelPushBdToQueue\s*\(", data["driver_control"])),
         "driver_DMA_coverage", "Every driver BD write and queue push must be parsed")
    descriptor_use, queue_use = set(), set()
    channel_descriptors, channel_queues = defaultdict(dict), {}
    for port_id, port in ports.items():
        pmap = mapping["portInstanceMapping"][port_id]
        if "dmaInfo" not in pmap: continue
        dma = pmap["dmaInfo"]
        need(type(dma) is dict and dma.get("direction") in {"s2mm", "mm2s"}, "DMA_mapping", port_id)
        dtile = tile(dma, port_id)
        channel = integer(dma.get("channel"), port_id + ".channel")
        direction = dma["direction"]
        need(direction == ("s2mm" if port["direction"] == "input" else "mm2s"), "DMA_port_direction", port_id)
        channel_id = resource(f"dma:{dtile[0]}:{dtile[1]}:{direction}:{channel}", "dma_channel",
                              dtile + [direction, channel], port_id, tile=dtile, direction=direction, channel=channel)
        port["channel_resource_id"] = channel_id
        port["packet_ids"] = copy.deepcopy(dma.get("packetID"))
        if port["packet_ids"] is not None:
            need(type(port["packet_ids"]) is list and bool(port["packet_ids"]), "DMA_packet_ids", port_id)
            for pid in port["packet_ids"]: integer(pid, port_id + ".packetID")
            need(len(set(port["packet_ids"])) == len(port["packet_ids"]), "DMA_packet_ids_duplicate", port_id)
        buffers = pmap.get("bufferInfo")
        need(type(buffers) is list and bool(buffers) and type(dma.get("bufferDescriptor")) is list,
             "DMA_buffers", port_id)
        bound = []
        for ordinal, buffer in enumerate(buffers):
            need(type(buffer) is dict, "buffer_mapping", port_id)
            btile = tile(buffer, port_id + ".buffer")
            offset = integer(buffer.get("offset"), port_id + ".offset")
            size = integer(buffer.get("size"), port_id + ".size", 1)
            name(buffer.get("generatedName"), port_id + ".bufferName")
            locks = buffer.get("lock")
            need(type(locks) is list and len(locks) == 1, "buffer_lock_binding", port_id)
            lock_id = integer(locks[0], port_id + ".lock")
            matches = [(i, d) for i, d in enumerate(descriptors) if d["buffer"] == buffer["generatedName"] and d["port"] == port["qualified_name"]]
            need(len(matches) == 1, "driver_buffer_binding", port_id)
            index, descriptor = matches[0]; descriptor_use.add(index)
            need(descriptor["driver_tile"] == [btile[0], btile[1] + row_origin] and
                 descriptor["offset_bytes"] == offset and descriptor["size_bytes"] == size and
                 descriptor["acquire_lock"]["id"] == descriptor["release_lock"]["id"] == lock_id and btile == dtile,
                 "driver_buffer_mismatch", port_id)
            buffer_id = resource(f"buffer:{btile[0]}:{btile[1]}:{offset}:{size}", "buffer",
                                 btile + [offset, size], port_id, tile=btile, byte_start=offset, byte_length=size)
            lock_resource = resource(f"lock:{btile[0]}:{btile[1]}:{lock_id}", "lock", btile + [lock_id], port_id,
                                     tile=btile, lock_id=lock_id, initial_value=None)
            bd_id = resource(f"bd:{btile[0]}:{btile[1]}:{descriptor['bd']}", "BD", btile + [descriptor["bd"]], port_id,
                             tile=btile, descriptor_id=descriptor["bd"], next_descriptor_id=descriptor["next_bd"],
                             buffer_resource_id=buffer_id, lock_resource_id=lock_resource,
                             acquire_lock=descriptor["acquire_lock"], release_lock=descriptor["release_lock"])
            port["buffer_slots"].append({"ordinal": ordinal, "buffer_name": buffer["generatedName"],
                                        "buffer_resource_id": buffer_id, "lock_resource_id": lock_resource, "BD_resource_id": bd_id})
            port["BD_slots"].append(descriptor["bd"]); bound.append(descriptor)
        need(set(port["BD_slots"]) == set(dma["bufferDescriptor"]) and len(port["BD_slots"]) == len(set(port["BD_slots"])),
             "DMA_BD_mapping", port_id)
        matching_queues = [(i, q) for i, q in enumerate(queues) if q["driver_tile"] == [dtile[0], dtile[1] + row_origin]
                           and q["direction"] == direction and q["channel"] == channel]
        need(len(matching_queues) == 1, "DMA_queue_binding", port_id)
        qi, q = matching_queues[0]; queue_use.add(qi)
        for descriptor in bound:
            definition = {k: v for k, v in descriptor.items() if k not in {"buffer", "port", "control_source_line"}}
            previous = channel_descriptors[channel_id].setdefault(descriptor["bd"], definition)
            need(previous == definition, "DMA_descriptor_definition_conflict", port_id)
        channel_queues[channel_id] = q
        port["initial_channel_BD"] = q["first_bd"]
    for channel_id, by_bd in channel_descriptors.items():
        q = channel_queues[channel_id]
        seen, cursor = [], q["first_bd"]
        while cursor not in seen:
            need(cursor in by_bd, "DMA_ring_broken", channel_id)
            seen.append(cursor); cursor = by_bd[cursor]["next_bd"]
        need(cursor == q["first_bd"] and len(seen) == len(by_bd), "DMA_ring_broken", channel_id)
        resources[channel_id]["BD_ring"] = seen
    need(descriptor_use == set(range(len(descriptors))) and queue_use == set(range(len(queues))),
         "driver_DMA_unbound", "All generated driver descriptors and queues must be bound")
    # Preserve optional manifest evidence, but never use it to select a graph/case.
    manifest = data.get("manifest")
    if manifest is not None:
        need(type(manifest) is dict, "manifest_object", "Static manifest")
        if "actors" in manifest:
            need({a.get("actor_id"): a.get("actual_compute_tile") for a in manifest["actors"]} ==
                 {a["id"]: a["actual_compute_tile"] for a in actor_rows}, "manifest_actor_mismatch", "Raw compiler actors/tiles")
        for key, ref in manifest.get("artifacts", {}).items():
            need(type(ref) is dict and ref.get("relative_path") == key, "manifest_artifact_ref", key)
            got = file_ref(ref["path"])
            need(got["bytes"] == ref.get("bytes") and got["sha256"] == ref.get("sha256"), "manifest_artifact_identity", key)
            pins.append({"role": "manifest_artifact:" + key, "kind": "pinned_file", **got})
        raw_role_paths = {"compiler_report": "Work/reports/compiler_report.json", "router_soln": "Work/temp/router_soln.json",
                          "driver_control": "Work/ps/c_rts/aie_control.cpp", "driver_config": "Work/ps/c_rts/aie_control_config.json",
                          "simulator_config": "Work/config/scsim_config.json", "packet_switching_report": "Work/reports/packet_switching_report.json",
                          "compile_result": "compile-result.json"}
        for role, relative in raw_role_paths.items():
            if relative in manifest.get("artifacts", {}) and role in data:
                expected = manifest["artifacts"][relative]
                actual_pin = next(p for p in pins if p["role"] == role)
                need(expected["sha256"] == actual_pin["sha256"] and expected["bytes"] == actual_pin["bytes"],
                     "manifest_supplied_artifact_mismatch", role)
        for window in manifest.get("window_ports", []):
            need(window.get("port_id") in ports, "manifest_port_mismatch", "Window port")
            port = ports[window["port_id"]]
            need(port["BD_slots"] == [s["descriptor"]["bd"] for s in window["slots"]], "manifest_BD_mismatch", port["id"])
            for slot, reported in zip(port["buffer_slots"], window["slots"]):
                buffer = resources[slot["buffer_resource_id"]]
                need(buffer["tile"] == reported["actual_memory_tile"] and buffer["byte_start"] == reported["offset_bytes"] and
                     buffer["byte_length"] == reported["size_bytes"], "manifest_buffer_mismatch", port["id"])
                bank = integer(reported["bank"], port["id"] + ".bank")
                bank_id = resource(f"bank:{buffer['tile'][0]}:{buffer['tile'][1]}:{bank}", "memory_bank",
                                   buffer["tile"] + [bank], port["id"], tile=buffer["tile"], bank=bank)
                buffer["bank_resource_id"] = bank_id
    # Packet IDs prove fixed selectors only; they do not supply FIFO/arbiter costs.
    packet_ids = {}
    packet = data.get("packet_switching_report")
    for category in ("Inputs", "Outputs", "Internal"):
        if packet is None: break
        need(type(packet.get(category)) is list, "packet_report_collection", category)
        for record in packet[category]:
            need(type(record) is dict and len(set(record) & {"Input", "Output"}) == 1 and type(record.get("PacketID")) is list,
                 "packet_report_record", category)
            side = "Input" if "Input" in record else "Output"
            port = record[side].get("Port")
            need(port in ports and bool(record["PacketID"]), "packet_report_port", str(port))
            for pid in record["PacketID"]: integer(pid, str(port) + ".PacketID")
            need(len(set(record["PacketID"])) == len(record["PacketID"]), "packet_id_duplicate", str(port))
            key = (side, port)
            need(key not in packet_ids or packet_ids[key] == record["PacketID"], "packet_id_conflict", str(port))
            packet_ids[key] = record["PacketID"]
    config = data["simulator_config"].get("SimulationConfig")
    need(type(config) is dict and type(config.get("pl_ip_block")) is list, "simulator_config", "SimulationConfig")
    clocks = []
    def clock(identity, hz, evidence):
        need(type(hz) in {int, float} and math.isfinite(hz) and hz > 0, "clock_frequency", identity)
        clocks.append({"id": identity, "hz": hz, "period_ps": 1e12 / hz, "phase_ps": None, "evidence": evidence})
    clock("aie", config.get("aie_freq"), "SimulationConfig/aie_freq")
    driver_hz = metadata.get("DeviceData", {}).get("AIEFrequency")
    need(type(driver_hz) in {int, float} and driver_hz * 1e6 == config["aie_freq"], "clock_evidence_mismatch", "AIEFrequency")
    plios = {}
    for plio in config["pl_ip_block"]:
        if not plio.get("axi_stream"): continue
        pname = name(plio.get("name"), "PLIO name")
        need(pname not in plios and type(plio["axi_stream"]) is list, "PLIO_duplicate_or_interfaces", pname)
        plios[pname] = plio; clock("plio:" + pname, plio.get("pl_freq"), "SimulationConfig/pl_ip_block/" + pname)
    def ancestor(block_id, visiting=()):
        if block_id in visiting: return None
        typ = types[instances[block_id]["blockType"]].get("type")
        if typ == "plio": return block_id
        incoming = [n for n in nets.values() if n["dstInstance"] == block_id]
        if typ != "pktsplit" or len(incoming) != 1: return None
        return ancestor(incoming[0]["srcInstance"], visiting + (block_id,))
    trees = _trees(data["router_soln"])
    covered_edges, net_groups = defaultdict(set), defaultdict(set)
    group_packet_destinations = defaultdict(lambda: defaultdict(set))
    source_packet_domains = {}
    for port_id, port in ports.items():
        if port["channel_resource_id"] is None: continue
        inbound = port["direction"] == "input"
        candidates = [(nid, n) for nid, n in nets.items() if n["dstPort" if inbound else "srcPort"] == port_id]
        if len(candidates) != 1:
            diagnostic("route_endpoint_compiler_binding_unsupported", "unsupported", "Not one compiler connection for DMA port", port_id=port_id)
            continue
        nid, net = candidates[0]
        ext = ancestor(net["srcInstance"]) if inbound else net["dstInstance"]
        if ext is None or types[instances[ext]["blockType"]].get("type") != "plio":
            diagnostic("route_external_ancestor_unsupported", "unsupported", "Route structure needs a known external source/sink", port_id=port_id)
            continue
        pname = instances[ext].get("logicalName")
        if pname not in plios or len(plios[pname]["axi_stream"]) != 1:
            diagnostic("PLIO_interface_binding_unsupported", "unsupported", "Not one bound PLIO interface", port_id=port_id)
            continue
        interface = plios[pname]["axi_stream"][0]
        need(interface.get("direction") == ("pl_to_me" if inbound else "me_to_pl"), "PLIO_direction_mismatch", pname)
        col = integer(interface.get("column"), pname + ".column")
        stream = integer(interface.get("stream_id"), pname + ".stream_id")
        width = integer(interface.get("bus_width"), pname + ".bus_width", 1)
        dma = resources[port["channel_resource_id"]]
        dc, dr = dma["tile"]; ch = dma["channel"]
        start = f"S_SHIM_SOUTH_ch{stream}_C{col}" if inbound else f"S_MM2S_DMA_ch{ch}_C{dc}_R{dr}"
        end = f"M_S2MM_DMA_ch{ch}_C{dc}_R{dr}" if inbound else f"M_SHIM_SOUTH_ch{stream}_C{col}"
        bound = [(key, path) for key, tree in trees.items() if (path := _path(tree, start, end)) is not None]
        need(len(bound) == 1, "router_endpoint_binding_ambiguous", start + " -> " + end)
        key, path = bound[0]
        group = "plio-source:" + ext if inbound else "dma-source:" + port["channel_resource_id"]
        fixed = True
        pids = port["packet_ids"]
        if inbound and pids is not None:
            need(packet_ids.get(("Input", port_id)) == pids and packet_ids.get(("Output", net["srcPort"])) == pids,
                 "fixed_packet_selector_mismatch", port_id)
            source_port_ids = [pid for pid, p in ports.items() if p["block_id"] == ext and p["direction"] == "output"]
            need(len(source_port_ids) == 1 and set(pids) <= set(packet_ids.get(("Input", source_port_ids[0]), [])),
                 "packet_source_domain_mismatch", pname)
            source_packet_domains[group] = set(packet_ids[("Input", source_port_ids[0])])
            for pid in pids: group_packet_destinations[group][pid].add(end)
        elif inbound and ports[net["srcPort"]]["type"] == "pktstream":
            fixed = False
            diagnostic("packet_selector_missing", "unsupported", "Packet route without compiler ID selection", port_id=port_id)
        covered_edges[key].update(zip(path, path[1:])); net_groups[key].add(group)
        route_id = "route:" + port_id
        routes.append({"id": route_id, "compiler_net_id": nid, "router_net_id": key, "window_port_id": port_id,
                       "src_port_id": net["srcPort"], "dst_port_id": net["dstPort"], "start": start, "end": end,
                       "ordered_physical_ports": ["router-port:" + p for p in path], "edge_count": len(path) - 1,
                       "source_group": group, "PLIO": pname, "clock_id": "plio:" + pname,
                       "stream_word_bits": width, "packet_ids": pids, "fixed_packet_branch": fixed,
                       "resource_ids": [], "latency_ps": None, "FIFO_capacity": None, "credit": None, "CDC_latency_ps": None})
        connection_by_id[nid]["route_ids"].append(route_id)
    unsafe_groups = set()
    for group, selectors in group_packet_destinations.items():
        if set(selectors) != source_packet_domains[group]:
            unsafe_groups.add(group)
            diagnostic("packet_source_domain_not_fully_routed", "unsupported", "Compiler source advertises packet IDs without a bound branch", source_group=group,
                       unrouted_packet_ids=sorted(source_packet_domains[group] - selectors.keys()))
        if any(len(destinations) > 1 for destinations in selectors.values()):
            unsafe_groups.add(group)
            diagnostic("packet_multicast_semantics_unsupported", "unsupported", "Same packet ID reaches multiple destinations", source_group=group)
    for route in routes:
        if not route["fixed_packet_branch"]: unsafe_groups.add(route["source_group"])
    # One physical port/link per identity across the COMPLETE router inventory.
    for key, tree in trees.items():
        unbound = set(map(tuple, tree["edges"])) - covered_edges[key]
        if unbound or not net_groups[key]:
            net_groups[key].add("unbound-router-net:" + key)
            diagnostic("router_net_not_fully_bound", "unsupported", "All raw edges retained, but occupancy source is not fully proven", router_net_id=key,
                       unbound_edges=[list(e) for e in sorted(unbound)])
        for label in sorted(tree["nodes"]):
            physical = _physical(label)
            if physical.get("tile_kind") == "memtile" and hw_gen != 2:
                diagnostic("memory_tile_architecture_mismatch", "unsupported",
                           "Memory-tile switch requires a compatible hardware generation", label=label, hw_gen=hw_gen)
            identity = physical["id"]
            nodes.setdefault(identity, physical)
            resource(identity, "stream_switch_port", physical["physical_key"], key,
                     physical_port=physical, FIFO_capacity=None, credit=None, CDC_latency_ps=None)
            if not physical["supported_structure"]:
                diagnostic("physical_port_kind_unsupported", "unsupported", "Physical label retained without mechanism assumptions", label=label)
        for left, right in tree["edges"]:
            lp, rp = _physical(left), _physical(right)
            rid = "route-edge:" + lp["label"] + "->" + rp["label"]
            edge_kind = ("unknown_route_edge" if "column" not in lp or "column" not in rp else
                         "switch_connection" if lp["column"] == rp["column"] and lp["row"] == rp["row"] else "physical_link")
            resource(rid, edge_kind, [lp["id"], rp["id"]], key,
                     src_node=lp["id"], dst_node=rp["id"], latency_ps=None, capacity=None)
    for record in routing_configuration["packet_slave_rules"] + routing_configuration["packet_master_ports"]:
        physical_id = record["physical_port_id"]
        if physical_id in nodes:
            nodes[physical_id].setdefault("driver_packet_configuration", []).append(record)
    arbiter_members = defaultdict(set)
    for record in routing_configuration["packet_slave_rules"] + routing_configuration["packet_master_ports"]:
        key = tuple(record["driver_tile"] + [record["arbiter_number"]])
        arbiter_members[key].add(record["physical_port_id"])
    arbiter_unknown_ports = {}
    for (column, driver_row, arbiter), members in arbiter_members.items():
        rid = f"packet-arbiter-driver:{column}:{driver_row}:{arbiter}"
        resource(rid, "packet_arbiter", [column, driver_row, arbiter],
                 driver_tile=[column, driver_row], arbiter_number=arbiter, member_physical_port_ids=sorted(members),
                 arbiter_service_cost=None)
        for physical_id in members:
            if physical_id in resources:
                resources[rid]["owners"].extend(n for n in resources[physical_id]["owners"] if n not in resources[rid]["owners"])
            else:
                arbiter_unknown_ports.setdefault(rid, []).append(physical_id)
    for r in resources.values():
        r["owners"].sort()
        if r["kind"] in {"stream_switch_port", "switch_connection", "physical_link", "unknown_route_edge", "packet_arbiter"}:
            groups = sorted({g for n in r["owners"] for g in net_groups[n]})
            if r["id"] in arbiter_unknown_ports:
                groups += ["unbound-driver-port:" + p for p in sorted(arbiter_unknown_ports[r["id"]])]
            private = len(groups) == 1 and not groups[0].startswith("unbound-") and groups[0] not in unsafe_groups
            r["ownership"] = {"status": "proven_source_private" if private else "unsupported_shared_or_unbound",
                              "source_groups": groups, "all_router_nets_examined": True, "router_net_count": len(trees),
                              "proof": "global physical occupancy of complete raw router trees; fixed source packet branches" if private else None}
            if not private:
                diagnostic("shared_or_unbound_router_resource_unsupported", "unsupported", "No private service assumed for ambiguous/shared resource", resource_id=r["id"], source_groups=groups)
            elif r["kind"] == "packet_arbiter":
                diagnostic("private_packet_arbiter_cost_unknown", "needs_measurement", "Known source-private arbiter configuration is not an arbiter service model", resource_id=r["id"])
        else:
            r["ownership"] = {"status": "single_declared_owner" if len(r["owners"]) == 1 else "shared_declared_resource",
                              "physical_deduplication": True}
            if len(r["owners"]) > 1:
                diagnostic("shared_resource_service_unknown", "unsupported" if r["kind"] != "memory_bank" else "needs_measurement",
                           "Physical sharing retained without arbitration/service costs", resource_id=r["id"])
    for route in routes:
        route["resource_ids"] = list(route["ordered_physical_ports"]) + ["route-edge:" + a[12:] + "->" + b[12:]
                                                                      for a, b in zip(route["ordered_physical_ports"], route["ordered_physical_ports"][1:])]
        route["resource_ids"] += [r["id"] for r in resources.values() if r["kind"] == "packet_arbiter" and
                                  set(r["member_physical_port_ids"]) & set(route["ordered_physical_ports"])]
        route["private_status"] = "proven_source_private" if all(resources[x]["ownership"]["status"] == "proven_source_private" for x in route["resource_ids"]) else "unsupported_shared_or_unbound"
        route["mechanism_status"] = "static_source_private_bound" if route["private_status"] == "proven_source_private" and all(nodes[p]["supported_structure"] for p in route["ordered_physical_ports"]) else "unsupported"
        diagnostic("route_dynamic_cost_unknown", "needs_measurement", "Static private routes do not establish capacity/credit/CDC/service costs", route_id=route["id"])
    for c in connections:
        if c["mechanism"] == "window_dma":
            p = ports[c["dst_port_id"]] if ports[c["dst_port_id"]]["channel_resource_id"] else ports[c["src_port_id"]]
            c["resource_ids"].append(p["channel_resource_id"])
            if not c["route_ids"]: c["mechanism_status"] = "unsupported"
    diagnostic("wrapper_execution_semantics_unknown", "needs_measurement", "Explicit per-call slot/lock order must come from a separately pinned compiled-program plan")
    per_tile_buffers = defaultdict(list)
    for r in resources.values():
        if r["kind"] == "buffer": per_tile_buffers[tuple(r["tile"])].append(r)
    for buffers in per_tile_buffers.values():
        ordered = sorted(buffers, key=lambda r: r["byte_start"])
        for index, left in enumerate(ordered):
            for right in ordered[index+1:]:
                if right["byte_start"] >= left["byte_start"] + left["byte_length"]: break
                diagnostic("overlapping_memory_buffers", "unsupported", "Physical byte ranges overlap; no independent ownership assumed",
                           resource_ids=[left["id"], right["id"]])
    # Recheck pinned artifacts so an externally changed source cannot silently pass.
    for pin in pins:
        if pin["kind"] == "pinned_file":
            got = file_ref(pin["path"])
            need(got["sha256"] == pin["sha256"] and got["bytes"] == pin["bytes"], "artifact_changed_during_build", pin["role"])
    identity_payload = sorted(({k: p[k] for k in ("role", "bytes", "sha256")} for p in pins), key=lambda p: p["role"])
    compile_identity = sha(canonical(identity_payload))
    graph = {"schema": SCHEMA, "provenance_kind": compiler_inputs["provenance_kind"],
            "status": "static_structure_with_unsupported_mechanisms" if any(d["status"] == "unsupported" for d in diagnostics) else "static_structure_bound_costs_unmeasured",
            "architecture": architecture, "compile_identity": compile_identity,
            "compile_identity_scope": "content identities of supplied raw inputs and optional manifest artifact inventory; not an asserted whole-Work archive hash",
            "compile_pins": pins, "actors": sorted(actor_rows, key=lambda a: a["id"]), "ports": sorted(ports.values(), key=lambda p: p["id"]),
            "nodes": sorted(nodes.values(), key=lambda n: n["id"]), "resources": sorted(resources.values(), key=lambda r: r["id"]),
            "routes": sorted(routes, key=lambda r: r["id"]), "connections": connections, "clocks": clocks,
            "diagnostics": diagnostics, "routing_configuration": routing_configuration,
            "coordinate_domains": {"compiler_core_memory": "zero-based AIE row", "driver_row_origin": row_origin, "shim_row": shim_row},
            "router_inventory": [{"id": key, "edge_count": len(tree["edges"]), "root": tree["root"],
                                  "source_groups": sorted(net_groups[key]), "all_edges_bound": not any(g.startswith("unbound-router-net:") for g in net_groups[key])}
                                 for key, tree in sorted(trees.items())],
            "counts": {"actors": len(actor_rows), "compute_tiles": len({tuple(a["actual_compute_tile"]) for a in actor_rows}),
                       "ports": len(ports), "router_nets": len(trees), "routes": len(routes), "driver_BDs": len(descriptors)},
            "native_admitted": False, "cost_admitted": False, "timing_prediction": None,
            "unknowns": ["FIFO_capacity", "credit", "CDC_phase_and_visibility", "arbiter_service", "dynamic_wrapper_slot_selection", "cascade_transaction_count_and_width"],
            "input_sources_verified_before_and_after": True}
    # Actual AIE1 packet bindings pin this complete canonical graph. Keep its
    # existing shape while still performing _communication_evidence validation
    # above. The new circuit-only contract can publish its report-role proof.
    if hw_gen != 1 or packet is None:
        graph["communication_evidence"] = communication_evidence
    return graph


def build_compiled_resource_graph(compiler_inputs: dict) -> dict:
    """Public normal API; see INPUT_SCHEMA and _build_compiled_resource_graph.

    Malformed evidence consistently raises CompiledResourceGraphError; unknown
    mechanisms yield structural output with explicit unsupported diagnostics.
    """
    try:
        return _build_compiled_resource_graph(compiler_inputs)
    except CompiledResourceGraphError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError, UnicodeError, FileNotFoundError) as error:
        raise CompiledResourceGraphError("malformed_compiler_inputs", str(error)) from error
