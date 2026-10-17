"""Initial rule specifications: evidence and obligations, not speedup claims."""
RULES = {
    'ordered_output_tree': {
        'kind':'deployment',
        'rewrite':'recursively assemble disjoint ordered output row intervals in existing compute cores',
        'guards':['each row appears once','children complete before parent gather','buffer extents match intervals'],
        'resource_effect':'fewer inputs per root with extra intermediate output buffering and copies',
        'origin':'compiler-guided response to failed eight-group centralized gather; no latency benefit presumed',
    },
    'partition_rows': {
        'kind':'semantic', 'match':'pure two-layer INT8 region',
        'rewrite':'partition output row domain and propagate slices through both GEMMs and Q0/Q1',
        'guards':['complete disjoint row cover','pointwise quantization','no cross-row state'],
        'resource_effect':'independent row tasks; physical core count determined by placement/reuse',
        'origin':'tensor identity implemented in semantics.partition_output',
    },
    'partition_hidden_reduction': {
        'kind':'semantic',
        'rewrite':'Q1(sum_j Q0(X*W0[:,J_j])*W1[J_j,:])',
        'guards':['full first-layer K reduction before Q0','INT32 partial sum range',
                  'Q1 after complete hidden reduction'],
        'resource_effect':'changes intermediate lifetime and output accumulator storage',
        'origin':'tensor identity implemented in semantics.rewrite_k_partition',
    },
    'explicit_rematerialization': {
        'kind':'semantic_execution',
        'rewrite':'recompute H separately for disjoint output column regions',
        'guards':['pure inputs','stable weights within a frame','preserve Q0'],
        'resource_effect':'extra MACs with independent intermediate ownership',
        'origin':'semantics.partition_output with explicit rematerialize',
    },
    'temporal_layer_reuse': {
        'kind':'deployment',
        'rewrite':'bind both layer stages of a row region to one compute tile',
        'guards':['producer before consumer','live buffers fit','full region completion and output gather'],
        'resource_effect':'one occupied tile per row group instead of two dedicated layer tiles',
        'source':'https://arxiv.org/html/2401.10417v2#S4',
        'attribution':'inspired by spatial/sequential composition; this specific generated implementation is ours',
    },
    'native_mmul_4_8_8': {
        'kind':'lowering', 'rewrite':'packed int8 tiles to AIE-ML MMUL and INT32 accumulators',
        'guards':['M divisible by 4','K/H/N divisible by 8','shift zero', 'Vitis 2024.2'],
        'resource_effect':'buffer layout, explicit accumulator storage, full-frame PLIO delivery',
        'origin':'reviewed existing math.hpp API shape; independent generated implementation',
    },
}
