#include "fa3_hdim128_bf16.cuh"

using AttnKernel = pipelink::attention::sm90::StockKernel<
    bool(FA3_STOCK_CAUSAL), FA3_STOCK_CLUSTER_M, bool(FA3_STOCK_IS_LOCAL)>::AttnKernel;

// CUTLASS takes one grid-constant, by-value Params record (device_kernel.h).
template __global__ void cutlass::device_kernel<AttnKernel>(
    CUTLASS_GRID_CONSTANT AttnKernel::Params const params);
