"""Bind intact AIE1 compiler artifacts without admitting a timing model.

This adapter reads Vitis MEGraphSchema-0.4, generated window drivers and
ELF32 linked listings. It never executes source/ELF or imports a predictor.
Static instruction order is not a dynamic execution/transaction schedule.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "aie-cascade-compiled-evidence-v1"


class EvidenceError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise EvidenceError(code, detail)


def file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


class ArtifactReader:
    def __init__(self, raw: Path):
        self.raw = raw.resolve()
        self.refs: dict[str, dict[str, Any]] = {}

    def path(self, key: str) -> Path:
        relative = PurePosixPath(key)
        require(not relative.is_absolute() and ":" not in key and ".." not in relative.parts,
                "artifact_path_escape", key)
        path = (self.raw / Path(*relative.parts)).resolve()
        require(path.is_relative_to(self.raw), "artifact_path_escape", key)
        require(path.is_file(), "artifact_missing", key)
        return path

    def ref(self, key: str) -> str:
        if key not in self.refs:
            self.refs[key] = {**file_identity(self.path(key)), "relative_path": key}
        return key

    def text(self, key: str) -> str:
        self.ref(key)
        return self.path(key).read_text(encoding="utf-8", errors="strict")

    def json(self, key: str) -> Any:
        return json.loads(self.text(key))

    def verify(self, key: str, expected: dict[str, Any]) -> None:
        self.ref(key)
        actual = self.refs[key]
        require(expected.get("sha256") == actual["sha256"] and expected.get("bytes") == actual["bytes"],
                "artifact_identity_mismatch", key)


def parse_elf_text(data: bytes) -> list[dict[str, Any]]:
    require(len(data) >= 52 and data[:7] == b"\x7fELF\x01\x01\x01", "elf_format_unsupported", "ELF32 little endian required")
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    require(header[2] == 264 and header[11] == 40 and header[12] > 0,
            "elf_architecture_unsupported", "linked AIE1 ELF machine 264 and standard section headers required")
    offset, count, names_index = header[6], header[12], header[13]
    require(offset + 40 * count <= len(data) and names_index < count, "elf_truncated", "section table")
    sections = [struct.unpack_from("<IIIIIIIIII", data, offset + 40 * i) for i in range(count)]
    table = sections[names_index]
    require(table[4] + table[5] <= len(data), "elf_truncated", "section names")
    names = data[table[4]:table[4] + table[5]]
    result = []
    for section in sections:
        require(section[0] < len(names), "elf_truncated", "invalid section name offset")
        name = names[section[0]:].split(b"\0", 1)[0].decode("ascii")
        if name != ".text":
            continue
        require(section[1] == 1 and section[4] + section[5] <= len(data), "elf_truncated", ".text")
        result.append({"address": section[3], "size_bytes": section[5],
                       "file_offset": section[4], "data": data[section[4]:section[4] + section[5]]})
    require(bool(result), "elf_text_missing", "no linked .text")
    return result


def parse_linked(map_text: str, listing: str, elf: bytes, kernel_name: str) -> dict[str, Any]:
    """Byte-check whole PM functions, including RET delay/tail instructions."""
    functions = []
    memory = None
    for line_number, line in enumerate(map_text.splitlines(), 1):
        heading = re.match(r"Memory map for memory '([^']+)':", line)
        if heading:
            memory = heading[1]
        match = re.match(r"\s*(0x[0-9a-f]+)\.\.(0x[0-9a-f]+)\s+\(\s*(\d+) items\)\s*:\s*(.+)::([^\s]+) \(Function,", line)
        if memory == "PM" and match:
            start, end, size = int(match[1], 16), int(match[2], 16), int(match[3])
            require(end - start + 1 == size, "linked_range_mismatch", match[5])
            functions.append({"symbol": match[5], "object": match[4], "pc_start": start,
                              "pc_end_inclusive": end, "size_bytes": size, "map_line": line_number})
    require(bool(functions), "linked_functions_missing", kernel_name)
    require(all(functions[i]["pc_end_inclusive"] < functions[i + 1]["pc_start"] for i in range(len(functions) - 1)),
            "linked_ranges_overlap", kernel_name)
    matches = [x for x in functions if kernel_name in x["symbol"]]
    require(len(matches) == 1 and sum(x["symbol"] == "_main" for x in functions) == 1,
            "linked_kernel_wrapper_ambiguous", kernel_name)
    kernel_symbol = matches[0]["symbol"]
    sections = parse_elf_text(elf)
    bundles = []
    directives = []
    loop_nesting = None
    for line_number, line in enumerate(listing.splitlines(), 1):
        match = re.match(r"\s*(\d+)\s+((?:0x[0-9a-f]{2}\s+)+)(.*)$", line)
        if not match:
            stripped = line.strip()
            if stripped.startswith("."):
                directives.append({"line": line_number, "text": stripped})
                if stripped == ".function_start":
                    loop_nesting = None
                nesting = re.match(r"\.loop_nesting (\d+)$", stripped)
                if nesting:
                    loop_nesting = int(nesting[1])
            continue
        pc = int(match[1])
        encoding = bytes(int(x, 16) for x in match[2].split())
        owners = [x for x in functions if x["pc_start"] <= pc <= x["pc_end_inclusive"]]
        require(len(owners) == 1, "linked_pc_unowned", str(pc))
        section = [x for x in sections if x["address"] <= pc and pc + len(encoding) <= x["address"] + x["size_bytes"]]
        require(len(section) == 1, "linked_pc_outside_elf", str(pc))
        actual = section[0]["data"][pc - section[0]["address"]:pc - section[0]["address"] + len(encoding)]
        # darts prints each complete instruction bundle most-significant byte
        # first; ELF stores that bundle least-significant byte first.
        require(actual == encoding[::-1], "linked_bytes_mismatch", str(pc))
        assembly = match[3].strip()
        operations = [x.strip() for x in assembly.split(";") if x.strip()]
        kinds = []
        if any(re.match(r"VMOV\s+[^,]+,\s*SCD$", op) for op in operations): kinds.append("cascade_read_SCD")
        if any(re.match(r"VMOV\s+MCD,", op) for op in operations): kinds.append("cascade_write_MCD")
        if any(op.startswith("VMAC") for op in operations): kinds.append("VMAC")
        if any(op.startswith("VST") for op in operations): kinds.append("vector_store")
        if any(op.startswith(("VLDA", "VLDB")) for op in operations): kinds.append("vector_load")
        if any(op.startswith("RET ") for op in operations): kinds.append("return")
        if any(op.startswith(("JAL ", "JALR ")) for op in operations): kinds.append("call")
        bundles.append({"pc": pc, "size_bytes": len(encoding), "listing_line": line_number,
                        "function": owners[0]["symbol"], "operations": operations, "event_kinds": kinds,
                        "directives_before": directives, "static_loop_nesting_annotation": loop_nesting,
                        "dynamic_execution_count": None, "dynamic_transaction_ordinal": None})
        directives = []
    require(len({x["pc"] for x in bundles}) == len(bundles), "linked_duplicate_pc", kernel_name)
    for function in functions:
        selected = [x for x in bundles if x["function"] == function["symbol"]]
        require(bool(selected), "linked_function_listing_missing", function["symbol"])
        cursor = function["pc_start"]
        for bundle in selected:
            require(bundle["pc"] == cursor, "linked_function_truncated", f"{function['symbol']} at {cursor}")
            cursor += bundle["size_bytes"]
        require(cursor == function["pc_end_inclusive"] + 1, "linked_function_truncated", function["symbol"])
        function["bundle_count"] = len(selected)
        function["event_pcs"] = {kind: [x["pc"] for x in selected if kind in x["event_kinds"]]
                                  for kind in sorted({k for x in selected for k in x["event_kinds"]})}
        ret = [x["pc"] for x in selected if "return" in x["event_kinds"]]
        function["cascade_writes_after_first_static_RET"] = [x["pc"] for x in selected if ret and x["pc"] > ret[0] and "cascade_write_MCD" in x["event_kinds"]]
        function["range_is_whole_linked_function"] = True
    return {"kernel_symbol": kernel_symbol, "wrapper_symbol": "_main", "all_pm_functions": functions,
            "listing_encoding_order": "darts most-significant-byte-first per whole bundle; reverse bundle to compare ELF",
            "all_linked_bundles": bundles, "whole_function_bytes_match_elf": True,
            "static_order_is_dynamic_schedule": False, "compute_cycles": None,
            "warning": "elapsed may include window/cascade/bank waits; never add cascade wait to waiting-inclusive compute"}


def parse_wrapper(text: str, function: str) -> dict[str, Any]:
    def line_at(offset: int) -> int: return text.count("\n", 0, offset) + 1
    macros = dict(re.findall(r"^#define\s+(LOCK_\w+)\s+(\d+)\s*$", text, re.M))
    arrays = {name: int(count) * 16 for name, count in re.findall(r"\bv8int16\s+(\w+)\[(\d+)\];", text)}
    init_index = re.findall(r"\bint32\s+index\s*=\s*(\d+)\s*;", text)
    updates = re.findall(r"(?m)^\s*index\s*=\s*([^;]+);", text)
    require(init_index == ["1"] and len(updates) == 1 and re.sub(r"\s+", "", updates[0]) == "1-index",
            "wrapper_index_state_unsupported", function)
    require(len(re.findall(r"\bindex\s*=", text)) == 2, "wrapper_index_state_unsupported", "extra index assignment")
    calls = list(re.finditer(r"(?m)^\s*" + re.escape(function) + r"\(([^;]+)\);", text))
    require(len(calls) == 1, "wrapper_call_ambiguous", function)
    windows = []
    for match in re.finditer(r"\bwindow_init\(([^;]+)\);", text):
        args = [x.strip() for x in match[1].split(",")]
        require(len(args) == 8 and args[1] == "1" and all(x.isdigit() for x in args[6:]),
                "wrapper_window_init_unsupported", match[1])
        window, _, ping, lock0, pong, lock1, n0, n1 = args
        require(ping in arrays and pong in arrays and arrays[ping] == int(n0) * 16 and arrays[pong] == int(n1) * 16,
                "wrapper_window_size_mismatch", window)
        variable = re.findall(r"\b(?:input|output)_window_int16\s*\*(\w+)\s*=\s*\(get_(input|output)_window_int16\(" + re.escape(window) + r"\)\);", text)
        require(len(variable) == 1, "wrapper_window_binding_ambiguous", window)
        handle, direction = variable[0]
        lockvar = "lockid_" + handle.split("_window_")[1]
        initial = re.findall(r"\bint\s+" + re.escape(lockvar) + r"\s*=\s*(LOCK_\w+)\s*;", text)
        require(initial == [lock0] and lock0 in macros and lock1 in macros, "wrapper_initial_lock_mismatch", window)
        require(len(re.findall(r"\b" + re.escape(lockvar) + r"\s*=", text)) == 2,
                "wrapper_selector_binding_unsupported", "extra lock assignment: " + window)
        switch = re.search(re.escape(lockvar) + r"\s*=\s*index\s*\?\s*(LOCK_\w+)\s*:\s*(LOCK_\w+)\s*;", text)
        adjust = re.search(r"adjustPingPongBuffer\(" + re.escape(window) + r",\s*" + re.escape(ping) + r"_ptr,\s*" + re.escape(pong) + r"_ptr,\s*0,\s*index\);", text)
        pointer = re.search(re.escape(handle) + r"->ptr\s*=\s*" + re.escape(window) + r"->head\s*;", text)
        require(switch is not None and list(switch.groups()) == [lock1, lock0] and adjust is not None and pointer is not None,
                "wrapper_selector_binding_unsupported", window)
        acquire = re.search(r"acquire\(" + re.escape(lockvar) + r",\s*(ACQ_READ|ACQ_WRITE)\);", text)
        release = re.search(r"release\(" + re.escape(lockvar) + r",\s*(REL_READ|REL_WRITE)\);", text)
        require(acquire is not None and release is not None and acquire.start() < calls[0].start() < release.start() < switch.start() < adjust.start() < pointer.start(),
                "wrapper_stage_order_mismatch", window)
        require(acquire[1] == ("ACQ_READ" if direction == "input" else "ACQ_WRITE") and
                release[1] == ("REL_WRITE" if direction == "input" else "REL_READ"), "wrapper_lock_mode_mismatch", window)
        slots = []
        for ordinal, (buffer, macro) in enumerate([(ping, lock0), (pong, lock1)]):
            lock_parts = re.fullmatch(r"LOCK_(\d+)_(\d+)_(\d+)_(\d+)", macro)
            require(lock_parts is not None, "wrapper_lock_macro_unsupported", macro)
            slots.append({"ordinal": ordinal, "buffer": buffer, "size_bytes": arrays[buffer], "lock_macro": macro,
                          "core_local_lock_operand": int(macros[macro]), "physical_lock_tile": [int(lock_parts[1]), int(lock_parts[2])],
                          "physical_lock_id": int(lock_parts[3])})
        windows.append({"port_id": handle.split("_window_")[1], "window_object": window, "direction": direction,
                        "slots": slots, "initial_lock_macro": lock0, "window_init_arguments": args,
                        "source_lines": {"init": line_at(match.start()), "acquire": line_at(acquire.start()),
                                         "release": line_at(release.start()), "selector": line_at(switch.start()),
                                         "adjust": line_at(adjust.start()), "pointer": line_at(pointer.start())},
                        "next_lock_expression": switch[0], "adjust_expression": adjust[0], "pointer_expression": pointer[0],
                        "buffer_head_semantics": "conditional on vendor window_init/adjustPingPongBuffer contract"})
    require(len(windows) in (2, 3), "wrapper_window_scope_unsupported", function)
    phases = [{"line": line_at(x.start()), "text": x[0]} for x in re.finditer(r"(?m)^\s*(?:acquire\([^;]+|release\([^;]+|chess_memory_fence\(\)|done\(\))\s*;", text)]
    return {"kernel_call": {"line": line_at(calls[0].start()), "text": calls[0][0].strip()},
            "windows": windows, "initial_index": 1, "index_update": updates[0].strip(),
            "state_transition_basis": "actual generated statements; no modulo selector assumed",
            "sync_bounds_source": "sync_buffer[1]", "dynamic_call_count": None,
            "full_source_preserved_by_artifact": True, "phase_statements": phases}


def parse_dma_control(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    descriptors = []
    for match in re.finditer(r"//Setting buffer (\w+) of ([^\r\n]+)\s*\{([^}]+)\}", text):
        body = match[3]
        def one(pattern: str) -> tuple[str, ...]:
            found = re.findall(pattern, body)
            require(len(found) == 1, "dma_descriptor_unsupported", match[1] + ": " + pattern)
            return found[0] if isinstance(found[0], tuple) else (found[0],)
        tile = tuple(map(int, one(r"XAie_DmaDescInit\(&DevInst, &DmaInst, XAie_TileLoc\((\d+),\s*(\d+)\)\)")))
        addr, size = one(r"XAie_DmaSetAddrLen\(&DmaInst, (0x[0-9a-f]+), (\d+)\)")
        acqlock, acqval, rellock, relval = map(int, one(r"XAie_DmaSetLock\(&DmaInst, XAie_LockInit\((\d+), (\d+)\), XAie_LockInit\((\d+), (\d+)\)\)"))
        next_bd, = one(r"XAie_DmaSetNextBd\(&DmaInst, (\d+), XAIE_ENABLE\)")
        wc, wr, bd = map(int, one(r"XAie_DmaWriteBd\(&DevInst, &DmaInst, XAie_TileLoc\((\d+),\s*(\d+)\), (\d+)\)"))
        require(tile == (wc, wr) and "XAie_DmaEnableBd(&DmaInst)" in body, "dma_descriptor_mismatch", match[1])
        descriptors.append({"buffer": match[1], "port": match[2].strip(), "driver_tile": list(tile),
                            "offset_bytes": int(addr, 16), "size_bytes": int(size), "bd": bd, "next_bd": int(next_bd),
                            "acquire_lock": {"id": acqlock, "value": acqval}, "release_lock": {"id": rellock, "value": relval},
                            "control_source_line": text.count("\n", 0, match.start()) + 1})
    queues = [{"driver_tile": [int(c), int(r)], "channel": int(channel), "direction": direction.lower(), "first_bd": int(bd),
               "control_source_line": text.count("\n", 0, match.start()) + 1}
              for match in re.finditer(r"XAie_DmaChannelPushBdToQueue\(&DevInst, XAie_TileLoc\((\d+),\s*(\d+)\), (\d+), DMA_(S2MM|MM2S), (\d+)\)", text)
              for c, r, channel, direction, bd in [match.groups()]]
    require(bool(descriptors) and bool(queues), "dma_control_missing", "descriptor/queue source")
    return descriptors, queues


def check_dma_ring(slots: list[dict[str, Any]], first_bd: int) -> None:
    by_bd = {x["bd"]: x for x in slots}
    require(len(by_bd) == len(slots), "dma_duplicate_bd", str(first_bd))
    visited = []
    cursor = first_bd
    while cursor not in visited:
        require(cursor in by_bd, "dma_ring_broken", str(cursor))
        visited.append(cursor)
        cursor = by_bd[cursor]["next_bd"]
    require(cursor == first_bd and len(visited) == len(slots), "dma_ring_broken", "not a single complete ring")


def bind_router(routes: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    """Resolve endpoints, never equate router/compiler textual net IDs."""
    candidates = []
    for key, value in routes.items():
        edges = value.get("tree", [])
        nodes = {node for edge in edges for node in edge}
        if start in nodes and end in nodes:
            candidates.append((key, edges))
    require(len(candidates) == 1, "router_endpoint_binding_ambiguous", f"{start} -> {end}")
    key, edges = candidates[0]
    require(all(isinstance(x, list) and len(x) == 2 for x in edges), "router_tree_unsupported", key)
    outgoing: dict[str, list[str]] = {}
    for left, right in edges:
        outgoing.setdefault(left, []).append(right)
    path = [start]
    while path[-1] != end:
        next_nodes = outgoing.get(path[-1], [])
        require(len(next_nodes) == 1 and next_nodes[0] not in path, "router_path_unsupported", key)
        path.append(next_nodes[0])
    require(len(path) - 1 == len(edges), "router_extra_edges_unsupported", key)
    return {"router_key": key, "binding_method": "unique DMA and PLIO endpoint markers",
            "start": start, "end": end, "ordered_nodes": path, "edge_count": len(edges), "edge_latency_ps": None}


def extract_compiled_evidence(folder: str | Path) -> dict[str, Any]:
    """Extract an auditable static manifest from the intact normalized archive."""
    folder = Path(folder).resolve()
    reader = ArtifactReader(folder / "raw")
    evidence_file = folder / "compiler-evidence.json"
    require(evidence_file.is_file(), "compiler_evidence_missing", str(evidence_file))
    previous = json.loads(evidence_file.read_text(encoding="utf-8"))
    require(previous.get("schema") == "aie2lean.compiler-evidence.v1", "compiler_evidence_schema_unsupported", str(evidence_file))
    for key, expected in previous["artifacts"].items():
        require(expected.get("relative_path") == key, "artifact_reference_mismatch", key)
        reader.verify(key, expected)
    compiler = reader.json("Work/reports/compiler_report.json")
    require(compiler.get("schema") == "MEGraphSchema-0.4", "compiler_mapping_schema_unsupported", str(compiler.get("schema")))
    candidate = reader.json("candidate.json")
    compile_result = reader.json("compile-result.json")
    require(candidate["candidate_id"] == previous["candidate_id"] == compile_result["candidate_id"] and compile_result.get("returncode") == 0,
            "compiled_candidate_identity_mismatch", "candidate/compile-result/evidence")
    require(candidate["source_identity"] == previous["source_identity"], "compiled_source_identity_mismatch", "candidate/evidence")
    command = compile_result.get("command", [])
    require("--mode" in command and command[command.index("--mode") + 1] == "aie" and
            any("/Vitis/2024.2/" in x for x in command), "toolchain_scope_unsupported", "Vitis 2024.2 AIE mode required")
    ir = reader.json("model/ir.json")
    require(ir.get("schema") == "aie2lean4.ir.v1", "semantic_ir_schema_unsupported", str(ir.get("schema")))
    for source in ir["sources"]:
        basename = source["file"].replace("\\", "/").rsplit("/", 1)[-1]
        reader.verify(basename, source)
    shape = ir["computation"]["shape"]
    require(shape == {"m": 48, "k": 96, "n": 48, "shards": 3, "k_per_shard": 32},
            "semantic_work_scope_unsupported", str(shape))
    config = reader.json("Work/config/scsim_config.json")["SimulationConfig"]
    control_config = reader.json("Work/ps/c_rts/aie_control_config.json")["aie_metadata"]
    require(control_config["driver_config"]["hw_gen"] == 1, "architecture_scope_unsupported", "AIE1 required")
    row_origin = control_config["driver_config"]["aie_tile_row_start"]
    core_hz = config["aie_freq"]
    require(isinstance(core_hz, (int, float)) and math.isfinite(core_hz) and core_hz > 0,
            "clock_invalid", "aie_freq")
    require(core_hz == control_config["DeviceData"]["AIEFrequency"] * 1e6,
            "clock_evidence_mismatch", "config and driver AIE frequency")
    clocks = {"aie": {"hz": core_hz, "period_ps": 1e12 / core_hz,
                      "evidence": ["Work/config/scsim_config.json#/SimulationConfig/aie_freq", "Work/ps/c_rts/aie_control_config.json#/aie_metadata/DeviceData/AIEFrequency"]}, "plio": []}
    plios = {x["name"]: x for x in config["pl_ip_block"] if x.get("axi_stream")}
    for name, plio in plios.items():
        require(math.isfinite(plio["pl_freq"]) and plio["pl_freq"] > 0, "clock_invalid", name)
        clocks["plio"].append({"name": name, "hz": plio["pl_freq"], "period_ps": 1e12 / plio["pl_freq"], "interfaces": plio["axi_stream"]})
    control = reader.text("Work/ps/c_rts/aie_control.cpp")
    descriptors, queues = parse_dma_control(control)
    bank_report = reader.text("Work/reports/graph_mapping_analysis_report.txt")
    lock_report = reader.json("Work/reports/dma_lock_report.json")
    packet = reader.json("Work/reports/packet_switching_report.json")
    router = reader.json("Work/temp/router_soln.json")
    actors, windows = [], []
    ports = compiler["portInstances"]
    types = compiler["blockTypes"]
    instances = compiler["blockInstances"]
    mapping = compiler["mapping"]
    for block_id, instance in instances.items():
        block_type = types[instance["blockType"]]
        if block_type.get("type") != "function" or instance.get("fabric") != "me":
            continue
        core = mapping["blockInstanceMapping"][block_id]["coreInfo"]
        require(core["tile"] == "aie", "compute_tile_unsupported", block_id)
        tile = [core["column"], core["row"]]
        tile_name = "_".join(map(str, tile))
        base = "Work/aie/" + tile_name
        elf_key, map_key, lst_key, wrapper_key = base + "/Release/" + tile_name, base + "/Release/" + tile_name + ".map", base + "/Release/" + tile_name + ".lst", base + "/src/" + tile_name + ".cc"
        reader.ref(elf_key)
        linked = parse_linked(reader.text(map_key), reader.text(lst_key), reader.path(elf_key).read_bytes(), block_type["function"])
        wrapper = parse_wrapper(reader.text(wrapper_key), block_type["function"])
        function_source = block_type["srcFile"]
        reader.ref(function_source)
        require("saturation_mode::none" in reader.text(function_source), "normalization_condition_missing", function_source)
        logical = instance["qualifiedName"].rsplit(".", 1)[-1]
        orders = [x for x in ir["deployment"]["source_order"] if x["instance"] == logical]
        require(len(orders) == 1, "logical_source_order_ambiguous", logical)
        cascade_order = [{"source_event_index": i, "kind": event["kind"], "offset": event["offset"], "lanes": event["lanes"],
                          "lane_bits": event["lane_bits"], "location": event["location"]}
                         for i, event in enumerate(orders[0]["events"]) if event["kind"] in ("cascade_read", "cascade_write")]
        for kind in ("cascade_read", "cascade_write"):
            sequence = [x for x in cascade_order if x["kind"] == kind]
            require(not sequence or (len(sequence) == 288 and all(x["lanes"] == 8 and x["lane_bits"] == 48 and x["offset"] == i * 8 for i, x in enumerate(sequence))),
                    "logical_cascade_order_unsupported", logical + ":" + kind)
        actor = {"actor_id": block_id, "program_id": instance["qualifiedName"], "function": block_type["function"],
                 "actual_compute_tile": tile, "compiler_schedule_field": core["schedule"], "source_ref": function_source,
                 "elf_ref": elf_key, "map_ref": map_key, "listing_ref": lst_key, "wrapper_ref": wrapper_key,
                 "linked_program": linked, "wrapper": wrapper,
                 "logical_cascade_order": cascade_order, "logical_order_ref": "model/ir.json",
                 "logical_to_dynamic_PC_binding": None, "compute_cost": None}
        actors.append(actor)
        for window in wrapper["windows"]:
            port_id = window["port_id"]
            require(port_id in ports and ports[port_id]["blockInstance"] == block_id, "wrapper_compiler_port_mismatch", port_id)
            port = ports[port_id]
            port_map = mapping["portInstanceMapping"][port_id]
            allocated = port_map["bufferInfo"]
            dma = port_map["dmaInfo"]
            require(len(allocated) == len(window["slots"]) == 2, "window_slot_scope_unsupported", port_id)
            physical_slots = []
            for slot in window["slots"]:
                matches = [x for x in allocated if x["generatedName"] == slot["buffer"]]
                require(len(matches) == 1, "window_buffer_binding_ambiguous", slot["buffer"])
                buffer = matches[0]
                buffer_tile = [buffer["column"], buffer["row"]]
                require(slot["size_bytes"] == buffer["size"] == port["blockSize"] and
                        slot["physical_lock_tile"] == buffer_tile and buffer["lock"] == [slot["physical_lock_id"]],
                        "window_resource_mismatch", slot["buffer"])
                bank_rows = re.findall(r"\b" + re.escape(slot["buffer"]) + r"\s+MG\((\d+),(\d+)\):(\d+)\s+(0x[0-9a-f]+)\s+(\d+)", bank_report)
                banks = {int(x[2]) for x in bank_rows if [int(x[0]), int(x[1])] == buffer_tile and int(x[3], 16) == buffer["offset"] and int(x[4]) == buffer["size"]}
                require(len(banks) == 1, "bank_evidence_ambiguous", slot["buffer"])
                descs = [x for x in descriptors if x["buffer"] == slot["buffer"] and x["port"] == port["qualifiedName"]]
                require(len(descs) == 1, "dma_buffer_binding_ambiguous", slot["buffer"])
                desc = descs[0]
                require(desc["driver_tile"] == [buffer_tile[0], buffer_tile[1] + row_origin] and
                        desc["offset_bytes"] == buffer["offset"] and desc["size_bytes"] == buffer["size"] and
                        desc["acquire_lock"]["id"] == desc["release_lock"]["id"] == slot["physical_lock_id"],
                        "dma_buffer_resource_mismatch", slot["buffer"])
                reports = [x for x in lock_report[dma["direction"].upper()] if x["KernelPort"]["VariableName"] == port["qualifiedName"]]
                require(len(reports) == 1 and any(x["Column"] == buffer_tile[0] and x["Row"] == buffer_tile[1] and
                        x["LockID"] == slot["physical_lock_id"] and x["Channel"] == dma["channel"] and x["BD"] == [desc["bd"]] for x in reports[0]["BufferInfo"]),
                        "dma_lock_report_mismatch", slot["buffer"])
                physical_slots.append({**slot, "actual_memory_tile": buffer_tile, "offset_bytes": buffer["offset"],
                                       "bank": next(iter(banks)), "descriptor": desc, "initial_physical_lock_value": None})
            bound_queues = [x for x in queues if x["driver_tile"] == [dma["column"], dma["row"] + row_origin] and x["direction"] == dma["direction"] and x["channel"] == dma["channel"]]
            require(len(bound_queues) == 1 and {x["descriptor"]["bd"] for x in physical_slots} == set(dma["bufferDescriptor"]),
                    "dma_queue_binding_mismatch", port_id)
            check_dma_ring([x["descriptor"] for x in physical_slots], bound_queues[0]["first_bd"])
            windows.append({"port_id": port_id, "qualified_port": port["qualifiedName"], "actor_id": block_id,
                            "direction": window["direction"], "bytes": port["blockSize"], "slots": physical_slots,
                            "dma": dma, "queue": bound_queues[0], "wrapper_ref": wrapper_key,
                            "compiler_pointer": "/mapping/portInstanceMapping/" + port_id,
                            "selection": {k: window[k] for k in ("initial_lock_macro", "next_lock_expression", "adjust_expression", "pointer_expression", "buffer_head_semantics")}})
    actual_tiles = sorted({tuple(x["actual_compute_tile"]) for x in actors})
    active = reader.json("Work/aie/active_cores.json")
    active_tiles = sorted(tuple(map(int, key.split("_"))) for record in active["ActiveCores"] for key in record)
    require(len(actors) == 3 and len(actual_tiles) == 3 and actual_tiles == active_tiles == sorted(map(tuple, previous["actual_compute_tile_set"])),
            "actual_compute_mapping_mismatch", "three distinct active cores required in this adapter scope")
    require(len(windows) == 7 and len(descriptors) == 14 and len(queues) == 7, "window_dma_coverage_mismatch", "expected actual seven dual-slot windows")
    nets = compiler["nets"]
    actor_ids = {x["actor_id"] for x in actors}
    def port_type(port_id: str) -> dict[str, Any]:
        port = ports[port_id]
        return types[instances[port["blockInstance"]]["blockType"]]["ports"][port["port"]]
    def plio_ancestor(block_id: str) -> str:
        if types[instances[block_id]["blockType"]]["type"] == "plio": return block_id
        incoming = [x for x in nets.values() if x["dstInstance"] == block_id]
        require(len(incoming) == 1 and types[instances[block_id]["blockType"]]["type"] == "pktsplit", "plio_ancestor_unsupported", block_id)
        require(types[instances[incoming[0]["srcInstance"]]["blockType"]]["type"] == "plio", "plio_ancestor_unsupported", block_id)
        return incoming[0]["srcInstance"]
    routes = []
    for window in windows:
        inbound = window["direction"] == "input"
        edges = [(key, net) for key, net in nets.items() if net["dstPort" if inbound else "srcPort"] == window["port_id"]]
        require(len(edges) == 1, "window_net_binding_ambiguous", window["port_id"])
        net_id, net = edges[0]
        external = plio_ancestor(net["srcInstance"]) if inbound else net["dstInstance"]
        name = instances[external]["logicalName"]
        require(name in plios and len(plios[name]["axi_stream"]) == 1, "plio_interface_binding_unsupported", name)
        interface = plios[name]["axi_stream"][0]
        dma = window["dma"]
        c, r, channel = dma["column"], dma["row"], dma["channel"]
        start = f"S_SHIM_SOUTH_ch{interface['stream_id']}_C{interface['column']}" if inbound else f"S_MM2S_DMA_ch{channel}_C{c}_R{r}"
        end = f"M_S2MM_DMA_ch{channel}_C{c}_R{r}" if inbound else f"M_SHIM_SOUTH_ch{interface['stream_id']}_C{interface['column']}"
        binding = bind_router(router, start, end)
        packet_ids = None
        if inbound:
            source_ids = [x["PacketID"] for x in packet["Internal"] if x.get("Output", {}).get("Port") == net["srcPort"]]
            destination_ids = [x["PacketID"] for x in packet["Internal"] if x.get("Input", {}).get("Port") == window["port_id"]]
            require(len(source_ids) == len(destination_ids) == 1 and source_ids[0] == destination_ids[0] == dma["packetID"] and len(source_ids[0]) == 1,
                    "packet_id_binding_mismatch", window["port_id"])
            packet_ids = source_ids[0]
        routes.append({"compiler_net_id": net_id, "src_port_id": net["srcPort"], "dst_port_id": net["dstPort"],
                       "window_port_id": window["port_id"], "plio": name, "packet_ids": packet_ids,
                       "route": binding, "route_ref": "Work/temp/router_soln.json", "clock_domain": name})
    cascade_links = []
    for net_id, net in nets.items():
        if port_type(net["srcPort"])["type"] != "cascade": continue
        require(port_type(net["dstPort"])["type"] == "cascade" and net["srcInstance"] in actor_ids and net["dstInstance"] in actor_ids,
                "cascade_net_type_mismatch", net_id)
        producer = next(x for x in actors if x["actor_id"] == net["srcInstance"])
        consumer = next(x for x in actors if x["actor_id"] == net["dstInstance"])
        writes = [x for x in producer["logical_cascade_order"] if x["kind"] == "cascade_write"]
        reads = [x for x in consumer["logical_cascade_order"] if x["kind"] == "cascade_read"]
        require([(x["offset"], x["lanes"], x["lane_bits"]) for x in writes] == [(x["offset"], x["lanes"], x["lane_bits"]) for x in reads] and len(writes) == 288,
                "cascade_logical_sequence_mismatch", net_id)
        producer_pcs = [x["pc"] for x in producer["linked_program"]["all_linked_bundles"] if x["function"] == producer["linked_program"]["kernel_symbol"] and "cascade_write_MCD" in x["event_kinds"]]
        consumer_pcs = [x["pc"] for x in consumer["linked_program"]["all_linked_bundles"] if x["function"] == consumer["linked_program"]["kernel_symbol"] and "cascade_read_SCD" in x["event_kinds"]]
        require(bool(producer_pcs) and bool(consumer_pcs), "cascade_link_instruction_missing", net_id)
        cascade_links.append({"channel_id": net_id, "producer_actor": producer["actor_id"], "consumer_actor": consumer["actor_id"],
                              "src_port_id": net["srcPort"], "dst_port_id": net["dstPort"], "element_type": "acc48",
                              "source_logical_transactions_per_application": 288, "logical_lanes_per_transaction": 8, "logical_lane_bits": 48,
                              "source_offset_sequence": [x["offset"] for x in writes], "producer_static_MCD_PCs": producer_pcs,
                              "consumer_static_SCD_PCs": consumer_pcs, "source_order_agrees": True,
                              "physical_beat_bits": None, "FIFO_capacity": None, "dynamic_ordinal_to_PC": None,
                              "backpressure_cost": None, "physical_route": None, "clock_domain": "aie"})
    require(len(cascade_links) == 2, "cascade_scope_unsupported", "two real cascade links required")
    bank_groups: dict[tuple[int, int, int], list[str]] = {}
    for window in windows:
        for slot in window["slots"]:
            bank_groups.setdefault((*slot["actual_memory_tile"], slot["bank"]), []).append(slot["buffer"])
    shared_banks = [{"memory_tile": list(key[:2]), "bank": key[2], "buffers": names, "arbitration_cost": None}
                    for key, names in sorted(bank_groups.items()) if len(names) > 1]
    unknowns = [
        {"code": "cascade_FIFO_capacity_unknown", "status": "needs_measurement", "detail": "no actual capacity/credit evidence; empty stream FIFO guidance does not prove capacity zero"},
        {"code": "cascade_physical_beat_and_route_unbound", "status": "needs_measurement", "detail": "compiler cascade endpoints and linked I/O exist; physical beat geometry/direct wire and publication frontier are not bound"},
        {"code": "dynamic_cascade_PC_ordinal_unknown", "status": "needs_measurement", "detail": "software-pipelined linked PC order is not runtime token order or occurrence count"},
        {"code": "instruction_bundle_issue_retirement_unknown", "status": "needs_measurement", "detail": "MCD/SCD/VMAC may share a bundle; no serial additive timing assigned"},
        {"code": "compute_wait_decomposition_unknown", "status": "needs_measurement", "detail": "waiting-inclusive kernel elapsed is forbidden as pure compute cost"},
        {"code": "bank_arbitration_unknown", "status": "needs_measurement", "detail": "bank allocation is known; dynamic contention/port capacity is unmeasured"},
        {"code": "packet_router_service_unknown", "status": "needs_measurement", "detail": "endpoints/path/ID are bound; header/drop/service/arbiter costs are not"},
        {"code": "window_vendor_head_contract_conditional", "status": "conditional", "detail": "actual index/lock/pointer statements retained, vendor API head update semantics remains a condition"},
        {"code": "initial_physical_lock_state_unknown", "status": "needs_measurement", "detail": "descriptor acquire/release values are bound; reset/startup lock values are not independently observed"},
        {"code": "multi_input_cascade_numeric_lowering_unsupported", "status": "unsupported", "detail": "static manifest is available; existing numeric factory does not execute this mechanism"},
        {"code": "independent_timing_holdout_missing", "status": "needs_measurement", "detail": "existing functional_development records are not TRAIN/HO timing admission"},
    ]
    return {"schema": SCHEMA, "status": "static_compiled_evidence_bound", "candidate_id": candidate["candidate_id"],
            "scope": {"architecture": "AIE1", "toolchain": "Vitis 2024.2", "compiler_schema": "MEGraphSchema-0.4", "source_shape": shape,
                      "source_semantics_status": ir["status"], "scope_is_generic_timing_support": False},
            "input_compiler_evidence": file_identity(evidence_file), "source_identity": candidate["source_identity"],
            "compile_result_ref": "compile-result.json", "compile_command": command,
            "coordinate_domains": {"compiler_core_memory": "zero-based AIE row", "driver": "XAie physical row",
                                   "aie_driver_row_origin": row_origin, "row_origin_ref": "Work/ps/c_rts/aie_control_config.json#/aie_metadata/driver_config/aie_tile_row_start"},
            "actual_compute_tile_set": [list(x) for x in actual_tiles], "actual_compute_core_count": len(actual_tiles),
            "actors": actors, "window_ports": windows, "cascade_channels": cascade_links,
            "packet_DMA_routes": routes, "shared_bank_resources": shared_banks, "clocks": clocks,
            "compiler_netlist": nets, "packet_switching_ref": "Work/reports/packet_switching_report.json",
            "artifacts": reader.refs, "unknowns": unknowns, "timing_prediction": None,
            "numeric_admitted": False, "cost_admitted": False, "new_native_run": False,
            "functional_observation_role": previous["observation_role"],
            "cost_policy": {"compute_cost": None, "copy_cost_reused": False, "waiting_inclusive_elapsed_is_compute": False,
                            "cascade_wait_added_to_elapsed": False, "static_instruction_counts_are_cycles": False},
            "evidence_boundary": "hashed compiler/source facts and conditional semantic order; no vendor compiler proof or dynamic timing admission"}


def capability_report(manifest: dict[str, Any]) -> dict[str, Any]:
    return {"schema": "aie-cascade-static-capability-v1", "candidate_id": manifest["candidate_id"],
            "status": manifest["status"], "actual_compute_core_count": manifest["actual_compute_core_count"],
            "bound": {"actors": len(manifest["actors"]), "windows": len(manifest["window_ports"]),
                      "buffers": sum(len(x["slots"]) for x in manifest["window_ports"]), "DMA_rings": len(manifest["window_ports"]),
                      "packet_DMA_routes": len(manifest["packet_DMA_routes"]), "cascade_links": len(manifest["cascade_channels"]),
                      "linked_bundles": sum(len(x["linked_program"]["all_linked_bundles"]) for x in manifest["actors"]),
                      "whole_linked_ELF_bytes_verified": True},
            "unknowns": manifest["unknowns"], "component_costs": [
                {"component": name, "status": "needs_measurement", "cost": None, "source": None}
                for name in ("kernel_progress_without_wait", "cascade_backpressure", "window_wrapper", "packet_DMA_route", "shared_bank_arbitration")],
            "cost_reuse_admitted": False, "numeric_admitted": False, "timing_prediction": None}
