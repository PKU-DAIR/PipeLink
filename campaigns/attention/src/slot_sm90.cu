#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "slot/request.cuh"
#include "slot/fixtures.cuh"
#include "sm90/family_params.cuh"
#include "sm90/slot_chunk.cuh"

#define CUDA_CHECK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e_)); } while (0)
#define CU_CHECK(expr) do { CUresult e_ = (expr); if (e_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(e_, &s_); throw std::runtime_error(std::string(#expr) + ": " + (s_ ? s_ : "driver error")); } } while (0)
#include "slot/timing.cuh"
#include "slot/family.cuh"

namespace {
using namespace pipelink::attention::slot;
using pipelink::attention::sm90::StockKernel;
using Element = typename StockKernel<true, 1>::Element;

struct Case {
  std::string id;
  int seqlen, batch, heads_q, heads_kv, head_dim;
  uint32_t numeric_case;
  float scale, softcap;
  int window, chunk;
  bool phi = false;
  pipelink::attention::sm90::PhiPattern phi_params{};
  bool msa = false;
};

Case resolve_case(std::string const& path, std::string const& id) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open cells JSON: " + path);
  std::string raw{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  Json document = JsonParser(raw).parse();
  if (document.at("schema").string() != "pipelink-sm90-cells-v1")
    throw std::runtime_error("unsupported cells JSON schema");
  for (auto const& row : document.at("cells").array()) {
    if (row.at("id").string() != id) continue;
    auto const& mask = row.at("mask");
    std::string kind = mask.at("kind").string();
    if (kind != "causal" && kind != "sliding_window" && kind != "chunked" &&
        kind != "phi3_blocksparse" && kind != "msa_selected")
      throw std::runtime_error(id + ": mask not supported until P4");
    Case c{id, row.at("seqlen_q").integer(), row.at("batch").integer(),
           row.at("heads_q").integer(), row.at("heads_kv").integer(),
           row.at("head_dim").integer(), uint32_t(row.at("numeric_case").integer()),
           float(row.at("scale").number()),
           row.at("softcap").kind == Json::Kind::null ? 0.f : float(row.at("softcap").number()),
           kind == "sliding_window" ? mask.at("window").integer() : 0,
           kind == "chunked" ? mask.at("chunk").integer() : 0};
    if (row.at("dtype").string() != "bf16" || c.head_dim != 128 ||
        c.seqlen != row.at("seqlen_k").integer() || c.seqlen <= 0 ||
        c.batch <= 0 || c.heads_q <= 0 || c.heads_kv <= 0 ||
        c.heads_q % c.heads_kv != 0 || !std::isfinite(c.scale) || c.scale <= 0 ||
        !std::isfinite(c.softcap) ||
        (row.at("softcap").kind != Json::Kind::null && c.softcap <= 0) ||
        (kind == "sliding_window" && c.window <= 0))
      throw std::runtime_error(id + ": requires bf16 d128 prefill with valid GQA dimensions, scale and window");
    if (kind == "chunked")
      pipelink::attention::sm90::validate_chunk_cell(c.batch, c.seqlen, c.chunk);
    if (kind == "phi3_blocksparse") {
      c.phi = true;
      c.phi_params = pipelink::attention::sm90::phi_pattern(mask);
      if (c.seqlen % 128 != 0 || c.softcap != 0)
        throw std::runtime_error("Phi requires complete 128-row tiles without softcap");
    }
    if (kind == "msa_selected") {
      c.msa = true;
      if (c.batch != 1 || c.heads_q != 64 || c.heads_kv != 4 ||
          c.seqlen % 128 != 0 || c.seqlen > 131072 || c.softcap != 0 ||
          mask.at("block").integer() != 128 || mask.at("topk").integer() != 16 ||
          !mask.at("forced_local").boolean())
        throw std::runtime_error("MSA requires B1 H64/4 block128 top16 with forced local and no softcap");
    }
    return c;
  }
  throw std::runtime_error("unknown sm90 cell: " + id);
}

struct Options {
  std::string mode, cubin, meta, link_json, linked_cubin, control_cubin;
  std::string product = "final", load = "dense", mask = "default";
  std::string cells_json, cell, timing = "parity", selection;
  bool seam_cost = false;
};
Options parse_options(int argc, char** argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--mode=", 0) == 0) out.mode = a.substr(7);
    else if (a.rfind("--stock-cubin=", 0) == 0) out.cubin = a.substr(14);
    else if (a.rfind("--stock-meta=", 0) == 0) out.meta = a.substr(13);
    else if (a.rfind("--link-json=", 0) == 0) out.link_json = a.substr(12);
    else if (a.rfind("--linked-cubin=", 0) == 0) out.linked_cubin = a.substr(15);
    else if (a.rfind("--control-cubin=", 0) == 0) out.control_cubin = a.substr(16);
    else if (a.rfind("--product=", 0) == 0) out.product = a.substr(10);
    else if (a.rfind("--load=", 0) == 0) out.load = a.substr(7);
    else if (a.rfind("--mask=", 0) == 0) out.mask = a.substr(7);
    else if (a.rfind("--cells-json=", 0) == 0) out.cells_json = a.substr(13);
    else if (a.rfind("--cell=", 0) == 0) out.cell = a.substr(7);
    else if (a.rfind("--timing=", 0) == 0) out.timing = a.substr(9);
    else if (a.rfind("--selection=", 0) == 0) out.selection = a.substr(12);
    else if (a == "--seam-cost") out.seam_cost = true;
    else throw std::runtime_error("unknown option: " + a);
  }
  if (out.cells_json.empty() || out.cell.empty())
    throw std::runtime_error("--cells-json and --cell are required");
  if (out.timing != "parity" && out.timing != "absolute")
    throw std::runtime_error("--timing must be absolute or parity");
  if (out.mode != "stock" && out.mode != "linked" && out.mode != "control")
    throw std::runtime_error("--mode must be stock, linked, or control");
  if (out.seam_cost && (out.mode != "linked" || out.timing != "absolute"))
    throw std::runtime_error("--seam-cost requires --mode=linked --timing=absolute");
  if (out.product != "final" || (out.load != "dense" && out.load != "block" && out.load != "msa") ||
      (out.mask != "default" && out.mask != "window" && out.mask != "phi"))
    throw std::runtime_error("sm90 requires final product and a supported load/mask pair");
  if (out.mode == "stock") {
    if (out.cubin.empty() || out.meta.empty())
      throw std::runtime_error("--stock-cubin and --stock-meta are required");
  } else if (out.link_json.empty() || out.linked_cubin.empty() || out.control_cubin.empty()) {
    throw std::runtime_error("--link-json, --linked-cubin, and --control-cubin are required for both family modes");
  }
  // Reject absent images before parsing stdin or touching CUDA, including the
  // companion image needed by the linked/control correctness gates.
  auto require_file = [](std::string const& path) {
    if (!std::ifstream(path, std::ios::binary))
      throw std::runtime_error("cannot open required input: " + path);
  };
  if (out.mode == "stock") {
    require_file(out.cubin); require_file(out.meta);
  } else {
    require_file(out.link_json); require_file(out.linked_cubin); require_file(out.control_cubin);
  }
  if (out.load == "msa") require_file(out.selection);
  return out;
}

struct Metadata { std::string symbol; int cluster_m; };
Metadata load_metadata(std::string const& path, Case const& c) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open stock metadata: " + path);
  std::string raw{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  Json meta = JsonParser(raw).parse();
  if (meta.at("schema").string() != "pipelink-attention-stock-image-v1" ||
      meta.at("dtype").string() != "bf16" || meta.at("head_dim").integer() != 128 ||
      meta.at("causal").boolean() != (c.window == 0) ||
      meta.at("is_local").boolean() != (c.window > 0))
    throw std::runtime_error("stock metadata does not match the sm90 cell");
  Metadata result{meta.at("kernel_symbol").string(), meta.at("cluster_m").integer()};
  if (result.symbol.empty() || result.cluster_m != 1)
    throw std::runtime_error("stock requires a kernel symbol and cluster_m=1");
  return result;
}

LinkMetadata load_link_metadata(Options const& opts, Case const& c) {
  return pipelink::attention::slot::load_link_metadata(opts.link_json, {
      opts.load == "msa" ? "msa-load" : opts.load == "block" ? "block-load" : "tma-load",
      "core", c.msa ? "epilogue-partial" : "epilogue", "empty",
      opts.mask == "phi" ? "mask-phi" : opts.mask == "window" ? "mask-window" : "mask-default",
      c.softcap > 0 ? "score-softcap" : "score-identity"});
}

std::vector<int> oracle_rows(Case const& c) {
  std::vector<int> rows{0, 1, 127, 128, c.seqlen / 2, c.seqlen - 1};
  if (c.window > 0)
    rows.insert(rows.end(), {c.window - 1, c.window, c.window + 1});
  if (c.chunk > 0)
    rows.insert(rows.end(), {c.chunk - 1, c.chunk, c.chunk + 1,
                            c.seqlen - c.chunk - 1, c.seqlen - c.chunk, c.seqlen - c.chunk + 1});
  if (c.phi)
    rows.insert(rows.end(), {63, 64, 65, 1023, 1024, 1087, 1088,
                            c.seqlen / 2 + 63, c.seqlen / 2 + 64, c.seqlen - 65, c.seqlen - 64});
  if (c.msa) rows.insert(rows.end(), {129, 1919, 1920, 2047, 2048, c.seqlen / 2 + 127});
  rows.erase(std::remove_if(rows.begin(), rows.end(),
      [&](int q) { return q < 0 || q >= c.seqlen; }), rows.end());
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

struct Problem {
  Case c;
  uint64_t q_count, kv_count, rows;
  std::vector<int> sample_rows;
  uint64_t sample_count;
  Buffer<Element> q, k, v, o;
  Buffer<float> lse, ref_o, ref_lse, scratch;
  Buffer<int> semaphore, bad, samples;

  explicit Problem(Case c) : c(c), q_count(uint64_t(c.batch) * c.seqlen * c.heads_q * c.head_dim),
      kv_count(uint64_t(c.batch) * c.seqlen * c.heads_kv * c.head_dim), rows(q_count / c.head_dim),
      sample_rows(oracle_rows(c)), sample_count(sample_rows.size() * c.batch * c.heads_q),
      q(q_count), k(kv_count), v(kv_count), o(q_count), lse(rows), ref_o(sample_count * 128),
      ref_lse(sample_count), scratch(sample_count * c.seqlen),
      semaphore(1), bad(1), samples(sample_rows.size()) {
    CUDA_CHECK(cudaMemcpy(samples.data, sample_rows.data(), sample_rows.size() * sizeof(int), cudaMemcpyHostToDevice));
    fill_philox_bf16<<<int((q_count + 255) / 256), 256>>>(q.data, q_count, c.numeric_case, 1);
    fill_philox_bf16<<<int((kv_count + 255) / 256), 256>>>(k.data, kv_count, c.numeric_case, 2);
    fill_philox_bf16<<<int((kv_count + 255) / 256), 256>>>(v.data, kv_count, c.numeric_case, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  Flash_fwd_params arguments(int num_sm) const {
    Flash_fwd_params p{};
    p.q_ptr = q.data; p.k_ptr = k.data; p.v_ptr = v.data;
    p.o_ptr = o.data; p.softmax_lse_ptr = lse.data;
    p.b = c.batch; p.b_k = c.batch; p.h = c.heads_q; p.h_k = c.heads_kv;
    p.seqlen_q = p.seqlen_k = c.seqlen;
    p.d = p.dv = 128;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.q_row_stride = p.o_row_stride = c.heads_q * 128;
    p.k_row_stride = p.v_row_stride = c.heads_kv * 128;
    p.q_batch_stride = p.o_batch_stride = p.q_row_stride * c.seqlen;
    p.k_batch_stride = p.v_batch_stride = p.k_row_stride * c.seqlen;
    p.v_dim_stride = 1;
    p.scale_softmax = c.scale;
    p.softcap = c.softcap;
    p.num_splits = 1; p.is_causal = c.window == 0; p.is_bf16 = true;
    p.window_size_left = c.window > 0 ? c.window - 1 : c.seqlen - 1;
    p.window_size_right = 0;
    p.tile_count_semaphore = semaphore.data;
    p.arch = 90; p.num_sm = num_sm;
    return p;
  }

  void reference(int32_t const* selection = nullptr) {
    spot_reference<<<int(sample_count), 256>>>(
        q.data, k.data, v.data, samples.data, int(sample_rows.size()),
        c.seqlen, c.heads_q, c.heads_kv, c.window, c.chunk, c.phi, selection, c.scale, c.softcap,
        scratch.data, ref_o.data, ref_lse.data);
    CUDA_CHECK(cudaGetLastError());
  }

  void check(Element const* output, float const* output_lse, char const* image) {
    CUDA_CHECK(cudaMemset(bad.data, 0, sizeof(int)));
    compare_spots<<<int(sample_count), 128>>>(output, output_lse, ref_o.data,
        ref_lse.data, samples.data, int(sample_rows.size()), 0, c.seqlen, c.heads_q, bad.data);
    CUDA_CHECK(cudaGetLastError());
    int failed = 0;
    CUDA_CHECK(cudaMemcpy(&failed, bad.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (failed & 1) throw std::runtime_error(std::string("Gate B: ") + image + " output differs from fp32 spot oracle");
    if (failed & 2) throw std::runtime_error(std::string("Gate C: ") + image + " LSE differs from fp32 spot oracle");
  }
};

template <bool Causal, bool IsLocal>
class LoadedKernel {
  using Adapter = StockKernel<Causal, 1, IsLocal>;
  using Kernel = typename Adapter::AttnKernel;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  typename Kernel::Params params_;
  int* semaphore_;
  CUlaunchConfig launch_{};
 public:
  LoadedKernel(Options const& opts, Metadata const& meta, Problem const& p, int num_sm)
      : params_(Adapter::make_params(p.arguments(num_sm), 0)), semaphore_(p.semaphore.data) {
    CU_CHECK(cuModuleLoad(&module_, opts.cubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&function_, module_, meta.symbol.c_str()));
    CU_CHECK(cuFuncSetAttribute(function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, Kernel::SharedStorageSize));
    dim3 grid = Kernel::get_grid_shape(params_);
    launch_.gridDimX = grid.x; launch_.gridDimY = grid.y; launch_.gridDimZ = grid.z;
    launch_.blockDimX = Kernel::MaxThreadsPerBlock; launch_.blockDimY = launch_.blockDimZ = 1;
    launch_.sharedMemBytes = Kernel::SharedStorageSize;
  }
  ~LoadedKernel() { if (module_) cuModuleUnload(module_); }
  LoadedKernel(LoadedKernel const&) = delete;
  LoadedKernel& operator=(LoadedKernel const&) = delete;

  void launch() {
    // FA3-as-shipped flash_api.cpp zeroes the dynamic scheduler counter for
    // every invocation. This is inside every timed batch, not just warmup.
    CUDA_CHECK(cudaMemsetAsync(semaphore_, 0, sizeof(int), nullptr));
    // cutlass::device_kernel takes one __grid_constant__ Params BY VALUE;
    // kernelParams points at that entire host record, not a device pointer.
    void* args[] = {&params_};
    CU_CHECK(cuLaunchKernelEx(&launch_, function_, args, nullptr));
  }
};

void write_prefix(Request const& r, Case const& c) {
  std::cout << std::setprecision(17)
            << "{\"schema_version\":\"m3-parity-slot-result-v1\",\"action\":" << json_quote(r.action)
            << ",\"case_id\":" << json_quote(r.case_id) << ",\"image_id\":" << json_quote(r.image_id)
            << ",\"image_role\":" << json_quote(r.image_role)
            << ",\"scheduler_semaphore_memset_per_launch\":" << (c.msa ? "false" : "true")
            << ",\"combine_per_launch\":" << (c.msa ? "true" : "false") << ',';
}

template <class Launch>
void measure(Request const& r, Options const& opts, Case const& c,
             char const* checks, Launch&& launch) {
  if (opts.timing == "absolute") {
    auto timing = time_absolute(launch);
    write_prefix(r, c);
    std::cout << checks << "\"timing\":true,";
    write_absolute_timing(std::cout, timing);
  } else {
    auto timing = time_slot(r, launch);
    write_prefix(r, c);
    double latency = double(timing.elapsed_ms) / timing.iterations;
    std::cout << checks << "\"timing\":true,\"iterations\":" << timing.iterations
              << ",\"latency_ms\":" << latency << ",\"throughput\":" << 1000.0 / latency
              << ",\"throughput_unit\":\"iterations/s\",";
    write_timing(std::cout, r, timing);
  }
  std::cout << "\"output_dtype\":\"bf16\",\"dtype\":\"bf16\",\"arch\":\"sm90\","
            << "\"oracle\":\"fp32-boundary-rows-all-heads\","
            << "\"work_definition\":\"attention O and natural-log LSE\"}\n";
}

// A stock image is both gate_b operands: there is no linked/control pair and
// Gate A is inapplicable (null), never a fabricated successful bitwise check.
constexpr char correctness[] = "\"correctness\":{\"all_true\":true,\"checked_before_warmup\":true,"
    "\"gate_a_bitwise\":null,\"gate_b_linked\":true,\"gate_b_control\":true,\"gate_c_lse\":true},";

template <bool Causal, bool IsLocal>
void process(Request const& r, Options const& opts, Metadata const& meta, Case const& c, cudaDeviceProp const& device) {
  Problem p(c);
  LoadedKernel<Causal, IsLocal> current(opts, meta, p, device.multiProcessorCount);
  if (r.action == "sanitize") {
    current.launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    write_prefix(r, c);
    std::cout << "\"images_executed\":[\"stock\"],\"timing\":false}\n";
    return;
  }
  p.reference();
  current.launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  p.check(p.o.data, p.lse.data, "stock");
  if (r.action == "warmup") {
    for (int i = 0; i < r.warmups; ++i) current.launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    write_prefix(r, c);
    std::cout << correctness << "\"timing\":false,\"warmups_completed\":" << r.warmups << "}\n";
    return;
  }
  measure(r, opts, c, correctness, [&] { current.launch(); });
}

template <bool Causal>
void process_family(Request const& r, Options const& opts, LinkMetadata const& meta,
                    Case const& c, cudaDeviceProp const& device) {
  using namespace pipelink::attention::sm90;
  Problem p(c);
  Buffer<Element> control_o(p.q_count);
  Buffer<float> control_lse(p.rows);
  int grid = std::min((c.seqlen + 127) / 128 * c.heads_q * c.batch, device.multiProcessorCount);
  auto arguments = p.arguments(grid);
  auto make_params = [&](Flash_fwd_params const& args) {
    return c.chunk > 0 ? make_chunk_family_params<Causal>(args, 0, c.chunk)
                       : make_family_params<Causal>(args, 0, c.window);
  };
  auto linked_params = make_params(arguments);
  arguments.o_ptr = control_o.data;
  arguments.softmax_lse_ptr = control_lse.data;
  auto control_params = make_params(arguments);
  PhiWorkspace phi(c.phi, arguments, c.phi_params);
  std::unique_ptr<MsaWorkspace> msa;
  if (c.msa) msa = std::make_unique<MsaWorkspace>(opts.selection, c.seqlen);
  PackedParams linked_packed(meta), control_packed(meta);
  auto pack = [&](PackedParams& packed, FamilyParams<Causal> const& params) {
    if (msa) packed.put(meta.params[0], msa->load(params.load, p.q.data));
    else if (c.phi) packed.put(meta.params[0], phi.params(params.load));
    else packed.put(meta.params[0], params.load);
    packed.put(meta.params[1], params.core);
    if (msa) packed.put(meta.params[2], msa->partial());
    else packed.put(meta.params[2], params.epilogue);
    packed.put(meta.params[3], params.empty);
    for (int i = 4; i < 6; ++i) {
      if (meta.params[i].size != 0 || meta.params[i].alignment != 1)
        throw std::runtime_error("sm90 mask and score callees must have no parameter blob");
    }
  };
  pack(linked_packed, linked_params);
  pack(control_packed, control_params);
  size_t const reset_bytes = Causal && !c.msa ? sizeof(int) : 0;
  FamilyKernel linked(opts.linked_cubin, meta, grid, p.semaphore.data, reset_bytes);
  FamilyKernel control(opts.control_cubin, meta, grid, p.semaphore.data, reset_bytes);
  auto launch_linked = [&] {
    linked.launch(linked_packed);
    if (msa) msa->combine(p.o.data, p.lse.data, c.scale);
  };
  auto launch_control = [&] {
    control.launch(control_packed);
    if (msa) msa->combine(control_o.data, control_lse.data, c.scale);
  };
  auto launch = [&] {
    if (opts.mode == "linked") launch_linked();
    else launch_control();
  };
  if (r.action == "sanitize") {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    write_prefix(r, c);
    std::cout << "\"images_executed\":[" << json_quote(opts.mode) << "],\"timing\":false}\n";
    return;
  }
  p.reference(msa ? msa->selection() : nullptr);
  launch_linked();
  launch_control();
  CUDA_CHECK(cudaDeviceSynchronize());
  if (opts.seam_cost) {
    // Different compiler schedules are allowed here: each image must satisfy
    // the unchanged fp32 O/LSE oracle, not linked/control Gate A equality.
    p.check(p.o.data, p.lse.data, "score-identity");
    p.check(control_o.data, control_lse.data, "no-score-seam");
    auto timings = time_absolute_pair(
        [&] { linked.launch(linked_packed); }, [&] { control.launch(control_packed); });
    write_prefix(r, c);
    std::cout << "\"correctness\":{\"all_true\":true,\"checked_before_warmup\":true,"
              << "\"gate_a_bitwise\":null,\"gate_b_linked\":true,\"gate_b_control\":true,\"gate_c_lse\":true},"
              << "\"timing\":true,\"seam_cost\":true,"
              << "\"measurement_scope\":\"production-score-identity-flag-off\","
              << "\"order\":\"alternating-candidate-first-reference-first\","
              << "\"measurements\":{\"candidate\":{";
    write_absolute_timing(std::cout, timings[0]);
    std::cout << "\"image_label\":\"score-identity\"},\"reference\":{";
    write_absolute_timing(std::cout, timings[1]);
    std::cout << "\"image_label\":\"no-score-seam\"}},\"output_dtype\":\"bf16\",\"dtype\":\"bf16\",\"arch\":\"sm90\","
              << "\"oracle\":\"fp32-boundary-rows-all-heads\","
              << "\"work_definition\":\"attention O and natural-log LSE\"}\n";
    return;
  }
  CUDA_CHECK(cudaMemset(p.bad.data, 0, sizeof(int)));
  compare_bitwise<<<int((p.q_count + 255) / 256), 256>>>(p.o.data, control_o.data, p.q_count, p.bad.data);
  compare_bitwise<<<int((p.rows + 255) / 256), 256>>>(p.lse.data, control_lse.data, p.rows, p.bad.data);
  CUDA_CHECK(cudaGetLastError());
  int failed = 0;
  CUDA_CHECK(cudaMemcpy(&failed, p.bad.data, sizeof(int), cudaMemcpyDeviceToHost));
  if (failed) throw std::runtime_error("Gate A: linked and control O/LSE are not bitwise equal");
  p.check(p.o.data, p.lse.data, "linked");
  p.check(control_o.data, control_lse.data, "control");
  constexpr char family_correctness[] =
      "\"correctness\":{\"all_true\":true,\"checked_before_warmup\":true,"
      "\"gate_a_bitwise\":true,\"gate_b_linked\":true,\"gate_b_control\":true,\"gate_c_lse\":true},";
  if (r.action == "warmup") {
    for (int i = 0; i < r.warmups; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    write_prefix(r, c);
    std::cout << family_correctness << "\"timing\":false,\"warmups_completed\":" << r.warmups << "}\n";
    return;
  }
  measure(r, opts, c, family_correctness, launch);
}
} // namespace

int main(int argc, char** argv) {
  try {
    Options opts = parse_options(argc, argv);
    Case c = resolve_case(opts.cells_json, opts.cell);
    Request r = read_request();
    if (opts.seam_cost && r.action != "slot")
      throw std::runtime_error("--seam-cost requires action=slot");
    if (r.case_id != c.id) throw std::runtime_error("request case_id does not match --cell");
    if (opts.mode != "stock" && (opts.mask == "window") != (c.window > 0))
      throw std::runtime_error("sliding_window cells require --mask=window; causal cells require --mask=default");
    if (opts.mode != "stock" && ((opts.load == "block") != c.phi || (opts.mask == "phi") != c.phi))
      throw std::runtime_error("Phi cells require --load=block --mask=phi");
    if (opts.mode != "stock" && (opts.load == "msa") != c.msa)
      throw std::runtime_error("MSA cells require --load=msa --selection=<path>");
    if (opts.mode == "stock" && (c.softcap > 0 || c.chunk > 0 || c.phi || c.msa))
      throw std::runtime_error("slot stock images support only causal/window cells without softcap");
    if (opts.seam_cost && (c.softcap > 0 || c.chunk > 0 || c.phi || c.msa))
      throw std::runtime_error("--seam-cost requires an identity causal/window cell");
    Metadata meta{};
    LinkMetadata family_meta{};
    if (opts.mode == "stock") meta = load_metadata(opts.meta, c);
    else family_meta = load_link_metadata(opts, c);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
    CU_CHECK(cuInit(0));
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    if (device.major != 9 || device.minor != 0) throw std::runtime_error("sm90 slot requires an sm90 GPU");
    if (opts.mode != "stock") {
      process_family<true>(r, opts, family_meta, c, device);
    } else if (c.window == 0) {
      process<true, false>(r, opts, meta, c, device);
    } else {
      process<false, true>(r, opts, meta, c, device);
    }
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "attention-slot-sm90: " << e.what() << '\n';
    return 2;
  }
}
