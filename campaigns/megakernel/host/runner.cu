#include "mk_host_abi.h"

#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void cuda_check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + ": " +
      (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name) + " (" +
      (description == nullptr ? "no description" : description) + ")");
}

uint64_t parse_u64(const std::string& text, const std::string& what) {
  if (text.empty() || text.front() == '-') {
    fail("invalid " + what + ": " + text);
  }
  size_t consumed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &consumed, 10);
  } catch (const std::exception&) {
    fail("invalid " + what + ": " + text);
  }
  if (consumed != text.size()) {
    fail("invalid " + what + ": " + text);
  }
  return static_cast<uint64_t>(value);
}

uint32_t parse_u32(const std::string& text, const std::string& what) {
  const uint64_t value = parse_u64(text, what);
  if (value > std::numeric_limits<uint32_t>::max()) {
    fail(what + " exceeds uint32 range: " + text);
  }
  return static_cast<uint32_t>(value);
}

size_t parse_size(const std::string& text, const std::string& what) {
  const uint64_t value = parse_u64(text, what);
  if (value > std::numeric_limits<size_t>::max()) {
    fail(what + " exceeds size_t range: " + text);
  }
  return static_cast<size_t>(value);
}

double parse_double(const std::string& text, const std::string& what) {
  size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception&) {
    fail("invalid " + what + ": " + text);
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    fail("invalid " + what + ": " + text);
  }
  return value;
}

plk_digest_v2 parse_digest(const std::string& text, const char* option) {
  if (text.size() != 64) fail(std::string(option) + " requires 64 hex digits");
  plk_digest_v2 digest{};
  auto nibble = [&](char c) -> uint8_t {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    fail(std::string(option) + " requires 64 hex digits");
  };
  for (size_t i = 0; i < sizeof(digest.sha256); ++i)
    digest.sha256[i] = (nibble(text[2 * i]) << 4) | nibble(text[2 * i + 1]);
  return digest;
}

plk_dim3_v2 parse_cluster(const std::string& text) {
  std::istringstream input(text);
  std::string part;
  plk_dim3_v2 result{};
  uint32_t* dimensions[] = {&result.x, &result.y, &result.z};
  for (auto dimension : dimensions) {
    if (!std::getline(input, part, ',')) fail("--cluster requires X,Y,Z");
    *dimension = parse_u32(part, "--cluster");
    if (!*dimension) fail("--cluster dimensions must be positive");
  }
  if (std::getline(input, part, ',')) fail("--cluster requires X,Y,Z");
  return result;
}

size_t checked_mul(size_t lhs, size_t rhs, const std::string& what) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    fail("size overflow for " + what);
  }
  return lhs * rhs;
}

size_t dtype_bytes(const std::string& dtype) {
  if (dtype == "bf16") {
    return 2;
  }
  if (dtype == "f32" || dtype == "i32" || dtype == "u32") {
    return 4;
  }
  fail("unsupported manifest dtype: " + dtype);
}

struct Options {
  fs::path cubin;
  std::string kernel;
  uint32_t dynamic_smem = 0;
  uint32_t threads = 0;
  plk_dim3_v2 cluster{};
  size_t params_bytes = 0;
  size_t residue_offset = 0;
  size_t residue_bytes = 0;
  size_t params_align = 0;
  fs::path provider;
  uint32_t owner_id = 0;
  plk_digest_v2 dtypes_digest{};
  plk_digest_v2 owner_contract{};
  fs::path bundle;
  fs::path out;
  std::string mode;
  std::map<int32_t, int32_t> opcode_map;
  bool ntok_set = false;
  uint32_t ntok = 0;
  bool pos_id_first_set = false;
  uint32_t pos_id_first = 0;
  bool sync_each = false;
  bool bench_seconds_set = false;
  double bench_seconds = 0.0;
  bool bench_discard_set = false;
  double bench_discard = 0.0;
  bool bench_batch_tokens_set = false;
  uint32_t bench_batch_tokens = 0;
  bool bench_pos_id_set = false;
  uint32_t bench_pos_id = 0;
};

void usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s --cubin PATH --kernel SYMBOL --dyn-smem N --threads N "
      "--cluster X,Y,Z --params-bytes N --params-align N "
      "--residue-offset N --residue-bytes N --owner-id N "
      "--dtypes-digest HEX --owner-contract HEX --provider PATH.so --bundle DIR --out DIR "
      "--mode check|bench --opcode-map 0:0,...,7:7 "
      "[--ntok N] [--pos-id-first N] [--sync-each 0|1] "
      "[--bench-seconds S --bench-discard S --bench-batch-tokens N "
      "--pos-id P]\n",
      argv0);
}

std::map<int32_t, int32_t> parse_opcode_map(const std::string& text) {
  std::map<int32_t, int32_t> result;
  std::istringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    const size_t colon = item.find(':');
    if (item.empty() || colon == std::string::npos ||
        item.find(':', colon + 1) != std::string::npos) {
      fail("invalid --opcode-map item: " + item);
    }
    const uint32_t source = parse_u32(item.substr(0, colon), "opcode source");
    const uint32_t target = parse_u32(item.substr(colon + 1), "opcode target");
    if (source > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        target > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      fail("opcode value exceeds int32 range: " + item);
    }
    if (!result.emplace(static_cast<int32_t>(source),
                        static_cast<int32_t>(target))
             .second) {
      fail("duplicate opcode source in --opcode-map: " +
           std::to_string(source));
    }
  }
  if (result.empty()) {
    fail("--opcode-map must not be empty");
  }
  return result;
}

Options parse_options(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    usage(argv[0]);
    std::exit(0);
  }
  if (argc < 2 || argc % 2 == 0) {
    usage(argv[0]);
    fail("all command-line options are named key/value pairs");
  }

  const std::set<std::string> allowed{
      "--cubin",          "--kernel",          "--dyn-smem",
      "--threads",        "--cluster",         "--params-bytes",
      "--residue-offset", "--residue-bytes",   "--params-align",
      "--owner-id",       "--dtypes-digest",   "--owner-contract",
      "--provider",       "--bundle",          "--out",
      "--mode",           "--opcode-map",      "--ntok",
      "--pos-id-first",   "--sync-each",       "--bench-seconds",
      "--bench-discard",  "--bench-batch-tokens", "--pos-id"};
  std::map<std::string, std::string> values;
  for (int i = 1; i < argc; i += 2) {
    const std::string key = argv[i];
    if (allowed.count(key) == 0) {
      fail("unknown option: " + key);
    }
    if (!values.emplace(key, argv[i + 1]).second) {
      fail("duplicate option: " + key);
    }
  }

  const std::vector<std::string> required{
      "--cubin",          "--kernel",       "--dyn-smem",
      "--threads",        "--cluster",       "--params-bytes",
      "--residue-offset", "--residue-bytes", "--params-align",
      "--owner-id",       "--dtypes-digest", "--owner-contract",
      "--provider",       "--bundle",        "--out",
      "--mode",           "--opcode-map"};
  for (const std::string& key : required) {
    if (values.count(key) == 0 || values.at(key).empty()) {
      fail("missing required option: " + key);
    }
  }

  Options options;
  options.cubin = values.at("--cubin");
  options.kernel = values.at("--kernel");
  options.dynamic_smem = parse_u32(values.at("--dyn-smem"), "--dyn-smem");
  options.threads = parse_u32(values.at("--threads"), "--threads");
  options.cluster = parse_cluster(values.at("--cluster"));
  options.params_bytes = parse_size(values.at("--params-bytes"), "--params-bytes");
  options.residue_offset = parse_size(values.at("--residue-offset"), "--residue-offset");
  options.residue_bytes = parse_size(values.at("--residue-bytes"), "--residue-bytes");
  options.params_align = parse_size(values.at("--params-align"), "--params-align");
  options.provider = values.at("--provider");
  options.owner_id = parse_u32(values.at("--owner-id"), "--owner-id");
  options.dtypes_digest = parse_digest(values.at("--dtypes-digest"), "--dtypes-digest");
  options.owner_contract = parse_digest(values.at("--owner-contract"), "--owner-contract");
  options.bundle = values.at("--bundle");
  options.out = values.at("--out");
  options.mode = values.at("--mode");
  options.opcode_map = parse_opcode_map(values.at("--opcode-map"));

  if (values.count("--ntok") != 0) {
    options.ntok_set = true;
    options.ntok = parse_u32(values.at("--ntok"), "--ntok");
  }
  if (values.count("--pos-id-first") != 0) {
    options.pos_id_first_set = true;
    options.pos_id_first =
        parse_u32(values.at("--pos-id-first"), "--pos-id-first");
  }
  if (values.count("--sync-each") != 0) {
    const uint32_t value = parse_u32(values.at("--sync-each"), "--sync-each");
    if (value > 1) {
      fail("--sync-each must be 0 or 1");
    }
    options.sync_each = value == 1;
  }
  if (values.count("--bench-seconds") != 0) {
    options.bench_seconds_set = true;
    options.bench_seconds =
        parse_double(values.at("--bench-seconds"), "--bench-seconds");
  }
  if (values.count("--bench-discard") != 0) {
    options.bench_discard_set = true;
    options.bench_discard =
        parse_double(values.at("--bench-discard"), "--bench-discard");
  }
  if (values.count("--bench-batch-tokens") != 0) {
    options.bench_batch_tokens_set = true;
    options.bench_batch_tokens = parse_u32(
        values.at("--bench-batch-tokens"), "--bench-batch-tokens");
  }
  if (values.count("--pos-id") != 0) {
    options.bench_pos_id_set = true;
    options.bench_pos_id = parse_u32(values.at("--pos-id"), "--pos-id");
  }

  if (options.mode != "check" && options.mode != "bench") {
    fail("--mode must be check or bench");
  }
  if (options.threads == 0 || options.params_bytes == 0 || options.owner_id == 0) {
    fail("--threads, --params-bytes, and --owner-id must be positive");
  }
  if (options.dynamic_smem >
      static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    fail("--dyn-smem exceeds CUDA function attribute range");
  }
  if (options.params_align < alignof(void*) ||
      (options.params_align & (options.params_align - 1)) != 0 ||
      options.params_align % alignof(void*) != 0) {
    fail("--params-align must be a power of two and a multiple of pointer alignment");
  }
  if (options.residue_offset > options.params_bytes ||
      options.residue_bytes > options.params_bytes - options.residue_offset) {
    fail("residue exceeds the params aggregate");
  }
  if (options.residue_bytes == 0) {
    fail("--residue-bytes must be positive");
  }
  if (options.mode == "check") {
    if (options.ntok_set && options.ntok == 0) {
      fail("--ntok must be positive");
    }
  } else {
    if (!options.bench_seconds_set || !options.bench_discard_set ||
        !options.bench_batch_tokens_set || !options.bench_pos_id_set) {
      fail("bench mode requires --bench-seconds, --bench-discard, "
           "--bench-batch-tokens, and --pos-id");
    }
    if (options.bench_seconds <= 0.0 || options.bench_discard < 0.0 ||
        options.bench_batch_tokens == 0) {
      fail("bench seconds and batch tokens must be positive; discard must be nonnegative");
    }
  }
  return options;
}

struct TensorSpec {
  std::string name;
  std::string dtype;
  std::vector<size_t> dims;
  fs::path path;
  size_t bytes = 0;
};

using Manifest = std::map<std::string, TensorSpec>;
using Scalars = std::map<std::string, std::string>;

Manifest read_manifest(const fs::path& bundle) {
  const fs::path manifest_path = bundle / "manifest.txt";
  std::ifstream input(manifest_path);
  if (!input) {
    fail("cannot open manifest: " + manifest_path.string());
  }

  Manifest result;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream row(line);
    TensorSpec tensor;
    std::string ndim_text;
    if (!(row >> tensor.name >> tensor.dtype >> ndim_text)) {
      fail("malformed manifest line " + std::to_string(line_number));
    }
    dtype_bytes(tensor.dtype);
    const uint32_t ndim = parse_u32(ndim_text, "manifest ndim");
    if (ndim == 0) {
      fail("zero-dimensional tensor is not supported: " + tensor.name);
    }
    tensor.dims.reserve(ndim);
    size_t elements = 1;
    for (uint32_t axis = 0; axis < ndim; ++axis) {
      std::string dim_text;
      if (!(row >> dim_text)) {
        fail("missing dimension for manifest tensor " + tensor.name);
      }
      const size_t dim = parse_size(dim_text, "manifest dimension");
      if (dim == 0) {
        fail("zero dimension for manifest tensor " + tensor.name);
      }
      tensor.dims.push_back(dim);
      elements = checked_mul(elements, dim, tensor.name);
    }
    std::string relative_path;
    if (!(row >> relative_path)) {
      fail("missing file for manifest tensor " + tensor.name);
    }
    std::string trailing;
    if (row >> trailing) {
      fail("trailing fields on manifest line " + std::to_string(line_number));
    }
    tensor.path = bundle / relative_path;
    tensor.bytes = checked_mul(elements, dtype_bytes(tensor.dtype), tensor.name);
    std::error_code error;
    const uintmax_t file_bytes = fs::file_size(tensor.path, error);
    if (error) {
      fail("cannot stat tensor file " + tensor.path.string() + ": " +
           error.message());
    }
    if (file_bytes != tensor.bytes) {
      fail("tensor file size mismatch for " + tensor.name + ": expected " +
           std::to_string(tensor.bytes) + ", got " +
           std::to_string(file_bytes));
    }
    if (!result.emplace(tensor.name, std::move(tensor)).second) {
      fail("duplicate manifest tensor: " + tensor.name);
    }
  }
  return result;
}

Scalars read_scalars(const fs::path& bundle) {
  const fs::path scalars_path = bundle / "scalars.txt";
  std::ifstream input(scalars_path);
  if (!input) {
    fail("cannot open scalars: " + scalars_path.string());
  }
  Scalars result;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream row(line);
    std::string key;
    std::string value;
    std::string trailing;
    if (!(row >> key >> value) || (row >> trailing)) {
      fail("malformed scalars line " + std::to_string(line_number));
    }
    if (!result.emplace(key, value).second) {
      fail("duplicate scalar: " + key);
    }
  }
  return result;
}

const std::string& scalar_text(const Scalars& scalars, const std::string& key) {
  const auto iterator = scalars.find(key);
  if (iterator == scalars.end()) {
    fail("missing scalar: " + key);
  }
  return iterator->second;
}

int32_t scalar_i32(const Scalars& scalars, const std::string& key) {
  const uint32_t value = parse_u32(scalar_text(scalars, key), "scalar " + key);
  if (value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    fail("scalar exceeds int32 range: " + key);
  }
  return static_cast<int32_t>(value);
}

float scalar_f32(const Scalars& scalars, const std::string& key) {
  const double value = parse_double(scalar_text(scalars, key), "scalar " + key);
  if (value < -std::numeric_limits<float>::max() ||
      value > std::numeric_limits<float>::max()) {
    fail("scalar exceeds float range: " + key);
  }
  return static_cast<float>(value);
}

const TensorSpec& tensor_spec(const Manifest& manifest, const std::string& name) {
  const auto iterator = manifest.find(name);
  if (iterator == manifest.end()) {
    fail("missing manifest tensor: " + name);
  }
  return iterator->second;
}

std::string shape_string(const std::vector<size_t>& dims) {
  std::ostringstream output;
  output << '[';
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      output << ',';
    }
    output << dims[i];
  }
  output << ']';
  return output.str();
}

const TensorSpec& expect_tensor(const Manifest& manifest,
                                const std::string& name,
                                const std::string& dtype,
                                std::vector<size_t> dims) {
  const TensorSpec& tensor = tensor_spec(manifest, name);
  if (tensor.dtype != dtype || tensor.dims != dims) {
    fail("manifest mismatch for " + name + ": expected " + dtype + " " +
         shape_string(dims) + ", got " + tensor.dtype + " " +
         shape_string(tensor.dims));
  }
  return tensor;
}

std::vector<unsigned char> read_bytes(const TensorSpec& tensor) {
  std::vector<unsigned char> bytes(tensor.bytes);
  std::ifstream input(tensor.path, std::ios::binary);
  if (!input || !input.read(reinterpret_cast<char*>(bytes.data()),
                            static_cast<std::streamsize>(bytes.size()))) {
    fail("cannot read tensor file: " + tensor.path.string());
  }
  return bytes;
}

class Context {
 public:
  explicit Context(CUdevice device) {
    CUctxCreateParams params{};
    cuda_check(cuCtxCreate(&context_, &params, 0, device), "cuCtxCreate");
  }
  ~Context() {
    if (context_ != nullptr) {
      cuCtxDestroy(context_);
    }
  }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

 private:
  CUcontext context_ = nullptr;
};

class Module {
 public:
  Module(const fs::path& path, const std::string& symbol) {
    cuda_check(cuModuleLoad(&module_, path.c_str()), "cuModuleLoad");
    try {
      cuda_check(cuModuleGetFunction(&function_, module_, symbol.c_str()),
                 "cuModuleGetFunction");
    } catch (...) {
      cuModuleUnload(module_);
      module_ = nullptr;
      throw;
    }
  }
  ~Module() {
    if (module_ != nullptr) {
      cuModuleUnload(module_);
    }
  }
  CUfunction function() const { return function_; }
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

 private:
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
};

class Stream {
 public:
  Stream() { cuda_check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate"); }
  ~Stream() {
    if (stream_ != nullptr) {
      cuStreamDestroy(stream_);
    }
  }
  operator CUstream() const { return stream_; }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

 private:
  CUstream stream_ = nullptr;
};

class Event {
 public:
  Event() { cuda_check(cuEventCreate(&event_, CU_EVENT_DEFAULT), "cuEventCreate"); }
  ~Event() {
    if (event_ != nullptr) {
      cuEventDestroy(event_);
    }
  }
  operator CUevent() const { return event_; }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

 private:
  CUevent event_ = nullptr;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) {
      fail("attempted zero-byte device allocation");
    }
    cuda_check(cuMemAlloc(&pointer_, bytes), "cuMemAlloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != 0) {
      cuMemFree(pointer_);
    }
  }
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(other.pointer_), bytes_(other.bytes_) {
    other.pointer_ = 0;
    other.bytes_ = 0;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  CUdeviceptr get() const { return pointer_; }
  size_t size() const { return bytes_; }

 private:
  CUdeviceptr pointer_ = 0;
  size_t bytes_ = 0;
};

class PinnedRow {
 public:
  explicit PinnedRow(size_t bytes) : bytes_(bytes) {
    cuda_check(cuMemAllocHost(&pointer_, bytes), "cuMemAllocHost(embedding row)");
  }
  ~PinnedRow() {
    if (pointer_ != nullptr) {
      cuMemFreeHost(pointer_);
    }
  }
  void* get() const { return pointer_; }
  size_t size() const { return bytes_; }
  PinnedRow(const PinnedRow&) = delete;
  PinnedRow& operator=(const PinnedRow&) = delete;

 private:
  void* pointer_ = nullptr;
  size_t bytes_ = 0;
};

class AlignedParams {
 public:
  AlignedParams(size_t bytes, size_t alignment) : bytes_(bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
      fail("params allocation size overflow");
    }
    const size_t allocation_bytes = (bytes + alignment - 1) & ~(alignment - 1);
    pointer_ = std::aligned_alloc(alignment, allocation_bytes);
    if (pointer_ == nullptr) {
      fail("aligned_alloc failed for params aggregate");
    }
    std::memset(pointer_, 0, allocation_bytes);
  }
  ~AlignedParams() { std::free(pointer_); }
  unsigned char* get() const { return static_cast<unsigned char*>(pointer_); }
  size_t size() const { return bytes_; }
  AlignedParams(const AlignedParams&) = delete;
  AlignedParams& operator=(const AlignedParams&) = delete;

 private:
  void* pointer_ = nullptr;
  size_t bytes_ = 0;
};

class SharedLibrary {
 public:
  explicit SharedLibrary(const fs::path& path) {
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) fail("cannot load provider " + path.string() + ": " + dlerror());
    try {
      abi = symbol<plk_abi_fn_v2>("pipelink_params_abi_interpreter");
      query = symbol<plk_query_fn_v2>("pipelink_params_query_interpreter");
      build = symbol<plk_build_fn_v2>("pipelink_params_build_interpreter");
      release = symbol<plk_release_fn_v2>("pipelink_params_release_interpreter");
    } catch (...) {
      dlclose(handle_);
      throw;
    }
  }
  ~SharedLibrary() { dlclose(handle_); }
  plk_abi_fn_v2 abi = nullptr;
  plk_query_fn_v2 query = nullptr;
  plk_build_fn_v2 build = nullptr;
  plk_release_fn_v2 release = nullptr;
  SharedLibrary(const SharedLibrary&) = delete;
  SharedLibrary& operator=(const SharedLibrary&) = delete;

 private:
  template <typename F>
  F symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle_, name);
    const char* error = dlerror();
    if (error || !address) fail(std::string("cannot resolve ") + name);
    F function;
    static_assert(sizeof(function) == sizeof(address));
    std::memcpy(&function, &address, sizeof(function));
    return function;
  }
  void* handle_ = nullptr;
};

DeviceBuffer upload_tensor(const TensorSpec& tensor) {
  DeviceBuffer device(tensor.bytes);
  std::ifstream input(tensor.path, std::ios::binary);
  if (!input) {
    fail("cannot open tensor for upload: " + tensor.path.string());
  }
  constexpr size_t kChunkBytes = 16u * 1024u * 1024u;
  std::vector<unsigned char> chunk(std::min(kChunkBytes, tensor.bytes));
  size_t offset = 0;
  while (offset < tensor.bytes) {
    const size_t count = std::min(chunk.size(), tensor.bytes - offset);
    if (!input.read(reinterpret_cast<char*>(chunk.data()),
                    static_cast<std::streamsize>(count))) {
      fail("short read while uploading tensor: " + tensor.path.string());
    }
    cuda_check(cuMemcpyHtoD(device.get() + offset, chunk.data(), count),
               ("cuMemcpyHtoD(" + tensor.name + ")").c_str());
    offset += count;
  }
  return device;
}

DeviceBuffer upload_instructions(const TensorSpec& tensor,
                                 const std::map<int32_t, int32_t>& opcode_map) {
  if (tensor.bytes % sizeof(int32_t) != 0 || tensor.dims.size() != 3 ||
      tensor.dims[2] != 32) {
    fail("instructions tensor does not contain 32-word int32 rows");
  }
  std::vector<int32_t> words(tensor.bytes / sizeof(int32_t));
  std::ifstream input(tensor.path, std::ios::binary);
  if (!input || !input.read(reinterpret_cast<char*>(words.data()),
                            static_cast<std::streamsize>(tensor.bytes))) {
    fail("cannot read instructions tensor: " + tensor.path.string());
  }
  const size_t rows = checked_mul(tensor.dims[0], tensor.dims[1], "instruction rows");
  for (size_t row = 0; row < rows; ++row) {
    const int32_t source = words[row * 32];
    const auto mapping = opcode_map.find(source);
    if (mapping == opcode_map.end()) {
      fail("unknown instruction opcode " + std::to_string(source) +
           " at row " + std::to_string(row));
    }
    words[row * 32] = mapping->second;
  }
  DeviceBuffer device(tensor.bytes);
  cuda_check(cuMemcpyHtoD(device.get(), words.data(), tensor.bytes),
             "cuMemcpyHtoD(instructions)");
  return device;
}

void zero_buffer(const DeviceBuffer& buffer, const std::string& name) {
  cuda_check(cuMemsetD8(buffer.get(), 0, buffer.size()),
             ("cuMemsetD8(" + name + ")").c_str());
}

void wait_with_watchdog(CUevent event, const char* phase) {
  static const uint32_t timeout_seconds = [] {
    const char* value = std::getenv("MK_WATCHDOG_SECONDS");
    const uint32_t seconds =
        value == nullptr ? 30 : parse_u32(value, "MK_WATCHDOG_SECONDS");
    if (seconds == 0) {
      fail("MK_WATCHDOG_SECONDS must be positive");
    }
    return seconds;
  }();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  while (true) {
    const CUresult result = cuEventQuery(event);
    if (result == CUDA_SUCCESS) {
      return;
    }
    if (result != CUDA_ERROR_NOT_READY) {
      cuda_check(result, "cuEventQuery");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      std::fprintf(
          stderr,
          "watchdog timeout: %s did not finish within %u s; the MK kernel may "
          "be stalled, or instrumented execution may need a longer deadline\n",
          phase, timeout_seconds);
      std::fflush(stderr);
      std::fflush(stdout);
      std::_Exit(3);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void write_binary(const fs::path& path, const void* data, size_t bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output || !output.write(static_cast<const char*>(data),
                               static_cast<std::streamsize>(bytes))) {
    fail("cannot write output: " + path.string());
  }
}

std::vector<unsigned char> copy_from_device(CUdeviceptr source, size_t bytes,
                                            const std::string& name) {
  std::vector<unsigned char> output(bytes);
  cuda_check(cuMemcpyDtoH(output.data(), source, bytes),
             ("cuMemcpyDtoH(" + name + ")").c_str());
  return output;
}

uint64_t epoch_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

struct RuntimeBuffers {
  DeviceBuffer bar;
  DeviceBuffer instructions;
  DeviceBuffer timings;
  DeviceBuffer qkv_weights;
  DeviceBuffer attn_norm_weights;
  DeviceBuffer o_weights;
  DeviceBuffer mlp_norm_weights;
  DeviceBuffer up_weights;
  DeviceBuffer gate_weights;
  DeviceBuffer down_weights;
  DeviceBuffer lm_head_norm_weights;
  DeviceBuffer lm_head_weights;
  DeviceBuffer k_cache;
  DeviceBuffer v_cache;
  DeviceBuffer rope_cos;
  DeviceBuffer rope_sin;
  DeviceBuffer hidden_states;
  DeviceBuffer q_post_rope;
  DeviceBuffer attn_out;
  DeviceBuffer attn_lse_intermediates;
  DeviceBuffer attn_out_intermediates;
  DeviceBuffer silu_out;
  DeviceBuffer logits;
};

MkProviderArgs make_provider_args(const RuntimeBuffers& buffers,
                                  int32_t num_sms,
                                  int32_t num_instructions,
                                  int32_t max_seq_len,
                                  int32_t rope_rows,
                                  int32_t num_layers,
                                  int32_t bar_rows,
                                  int32_t vocab_size,
                                  float attn_scale,
                                  float rms_norm_eps,
                                  int32_t skip_attn_reduction) {
  MkProviderArgs args{};
  args.version = MK_PROVIDER_ARGS_VERSION;
  args.bar = buffers.bar.get();
  args.instructions = buffers.instructions.get();
  args.timings = buffers.timings.get();
  args.qkv_weights = buffers.qkv_weights.get();
  args.attn_norm_weights = buffers.attn_norm_weights.get();
  args.o_weights = buffers.o_weights.get();
  args.mlp_norm_weights = buffers.mlp_norm_weights.get();
  args.up_weights = buffers.up_weights.get();
  args.gate_weights = buffers.gate_weights.get();
  args.down_weights = buffers.down_weights.get();
  args.lm_head_norm_weights = buffers.lm_head_norm_weights.get();
  args.lm_head_weights = buffers.lm_head_weights.get();
  args.k_cache = buffers.k_cache.get();
  args.v_cache = buffers.v_cache.get();
  args.rope_cos = buffers.rope_cos.get();
  args.rope_sin = buffers.rope_sin.get();
  args.hidden_states = buffers.hidden_states.get();
  args.q_post_rope = buffers.q_post_rope.get();
  args.attn_out = buffers.attn_out.get();
  args.attn_lse_intermediates = buffers.attn_lse_intermediates.get();
  args.attn_out_intermediates = buffers.attn_out_intermediates.get();
  args.silu_out = buffers.silu_out.get();
  args.logits = buffers.logits.get();
  args.num_sms = num_sms;
  args.num_instructions = num_instructions;
  args.max_seq_len = max_seq_len;
  args.rope_rows = rope_rows;
  args.num_layers = num_layers;
  args.bar_rows = bar_rows;
  args.vocab_size = vocab_size;
  args.attn_scale = attn_scale;
  args.rms_norm_eps = rms_norm_eps;
  args.skip_attn_reduction = skip_attn_reduction;
  return args;
}

void provider_check(int status, const char* phase) {
  if (status != PLK_OK)
    fail(std::string("provider ") + phase + " failed with status " + std::to_string(status));
}

void verify_provider_abi(const SharedLibrary& provider, const Options& options) {
#define MK_ABI_ENTRY(id, T, ...) {id, 0, sizeof(T), alignof(T), {}},
#define MK_ROLE_ENTRY(id, T) {id, 0, sizeof(mk_host::T), alignof(mk_host::T), {}},
  const plk_layout_entry_v2 expected[] = {
      {1, 0, sizeof(MkProviderArgs), alignof(MkProviderArgs), {}},
      {2, 0, options.residue_bytes, MK_GLOBALS_ALIGN, {}},
      MK_HOST_ROLE_TYPES(MK_ROLE_ENTRY)
      PLK_HOST_ABI_TYPES(MK_ABI_ENTRY)};
#undef MK_ABI_ENTRY
#undef MK_ROLE_ENTRY
  plk_layout_entry_v2 entries[sizeof(expected) / sizeof(expected[0])];
  std::memcpy(entries, expected, sizeof(entries));
  plk_abi_report_v2 report{};
  report.header = {PLK_HOST_ABI_V2, sizeof(report)};
  report.entry_capacity = sizeof(entries) / sizeof(entries[0]);
  report.entries = entries;
  provider_check(provider.abi(&report), "abi");
  if (report.header.abi_version != PLK_HOST_ABI_V2 ||
      report.header.struct_bytes != sizeof(report) ||
      report.entry_capacity != sizeof(entries) / sizeof(entries[0]) ||
      report.entries != entries ||
      report.entry_count != report.entry_capacity ||
      std::memcmp(&report.profile.dtypes, &options.dtypes_digest, sizeof(plk_digest_v2)) ||
      std::memcmp(&report.owner_contract, &options.owner_contract, sizeof(plk_digest_v2)))
    fail("provider ABI identity differs from link.json");
  for (size_t i = 0; i < report.entry_count; ++i) {
    if (entries[i].abi_type_id != expected[i].abi_type_id ||
        entries[i].bytes != expected[i].bytes ||
        entries[i].alignment_bytes != expected[i].alignment_bytes)
      fail("provider ABI layout differs for type " + std::to_string(expected[i].abi_type_id));
  }
  const auto& residue = entries[1];
  if (options.params_align < residue.alignment_bytes ||
      options.residue_offset % residue.alignment_bytes ||
      options.params_bytes % options.params_align ||
      residue.bytes > options.params_bytes - options.residue_offset)
    fail("provider residue layout does not fit the params aggregate");
}

class ProviderSession {
 public:
  ProviderSession(const SharedLibrary& provider, CUstream stream)
      : provider_(provider), stream_(stream) {}
  ~ProviderSession() {
    if (attempted_) {
      // An exception after build must drain submitted work before owner release.
      const auto synchronized = cuStreamSynchronize(stream_);
      const int released = provider_.release(session_);
      if (synchronized != CUDA_SUCCESS || released != PLK_OK)
        std::fprintf(stderr, "provider session cleanup failed: sync=%d release=%d\n",
                     static_cast<int>(synchronized), released);
    }
  }
  void open(const plk_common_v2& common, const MkProviderArgs& args,
            const Options& options, uint32_t device_sms, AlignedParams& params) {
    plk_demands_v2 demands{};
    demands.header = {PLK_HOST_ABI_V2, sizeof(demands)};
    provider_check(provider_.query(&common, &args, &demands), "query");
    geometry_ = demands.geometry;
    auto equal = [](plk_dim3_v2 a, plk_dim3_v2 b) {
      return a.x == b.x && a.y == b.y && a.z == b.z;
    };
    if (demands.channel_count || demands.workspace_count ||
        geometry_.geometry_present != 1 || geometry_.launch_required != 1 ||
        !equal(geometry_.grid, {device_sms, 1, 1}) ||
        !equal(geometry_.block, common.block) ||
        !equal(geometry_.cluster, common.cluster) ||
        geometry_.dynamic_smem_bytes != common.dynamic_smem_bytes)
      fail("provider query geometry or resource demands violate the MK launch contract");
    plk_resources_v2 resources{};
    resources.header = {PLK_HOST_ABI_V2, sizeof(resources)};
    resources.cuda_stream_token = reinterpret_cast<uint64_t>(stream_);
    resources.geometry = geometry_;
    attempted_ = true;
    provider_check(provider_.build(&common, &args, &resources,
        params.get() + options.residue_offset, options.residue_bytes,
        nullptr, 0, &session_), "build");
  }
  void release_after_sync() {
    attempted_ = false;
    provider_check(provider_.release(session_), "release");
    session_ = nullptr;
  }
  const plk_geometry_v2& geometry() const { return geometry_; }
  ProviderSession(const ProviderSession&) = delete;
  ProviderSession& operator=(const ProviderSession&) = delete;

 private:
  const SharedLibrary& provider_;
  CUstream stream_;
  plk_geometry_v2 geometry_{};
  plk_session* session_ = nullptr;
  bool attempted_ = false;
};

void launch_kernel(CUfunction function, const plk_geometry_v2& geometry, CUstream stream,
                   AlignedParams* params) {
  CUlaunchConfig config{};
  config.gridDimX = geometry.grid.x;
  config.gridDimY = geometry.grid.y;
  config.gridDimZ = geometry.grid.z;
  config.blockDimX = geometry.block.x;
  config.blockDimY = geometry.block.y;
  config.blockDimZ = geometry.block.z;
  config.sharedMemBytes = static_cast<unsigned>(geometry.dynamic_smem_bytes);
  config.hStream = stream;
  CUlaunchAttribute cluster{};
  cluster.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  cluster.value.clusterDim = {geometry.cluster.x, geometry.cluster.y, geometry.cluster.z};
  config.attrs = &cluster;
  config.numAttrs = 1;

  size_t params_size = params->size();
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, params->get(),
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &params_size,
                   CU_LAUNCH_PARAM_END};
  cuda_check(cuLaunchKernelEx(&config, function, nullptr, extra),
             "cuLaunchKernelEx");
}

void stage_embedding(const std::vector<unsigned char>& embeddings,
                     size_t embedding_rows, size_t row,
                     PinnedRow* pinned_row, const DeviceBuffer& hidden_states,
                     CUstream stream) {
  if (row >= embedding_rows) {
    fail("embedding row out of range");
  }
  std::memcpy(pinned_row->get(), embeddings.data() + row * pinned_row->size(),
              pinned_row->size());
  cuda_check(cuMemcpyHtoDAsync(hidden_states.get(), pinned_row->get(),
                               pinned_row->size(), stream),
             "cuMemcpyHtoDAsync(hidden_states <- embeddings[t])");
}

void dump_outputs(const fs::path& out, uint32_t token, uint32_t pos_id,
                  const RuntimeBuffers& buffers, int32_t num_layers,
                  int32_t max_seq_len, int32_t num_sms, int32_t vocab_size) {
  const std::string prefix = "tok" + std::to_string(token) + "_";
  auto dump_contiguous = [&](const std::string& name, CUdeviceptr pointer,
                             size_t bytes) {
    const std::vector<unsigned char> host = copy_from_device(pointer, bytes, name);
    write_binary(out / (prefix + name + ".bin"), host.data(), host.size());
  };

  dump_contiguous("hidden_states", buffers.hidden_states.get(), 2048u * 2u);
  dump_contiguous("logits", buffers.logits.get(),
                  static_cast<size_t>(vocab_size) * 2u);
  dump_contiguous("q_post_rope", buffers.q_post_rope.get(), 2048u * 2u);
  dump_contiguous("attn_out", buffers.attn_out.get(), 2048u * 2u);
  dump_contiguous("silu_out", buffers.silu_out.get(), 8192u * 2u);

  constexpr size_t kKvRowElements = 8u * 64u;
  const size_t kv_layer_bytes = kKvRowElements * sizeof(uint16_t);
  std::vector<unsigned char> k_row(static_cast<size_t>(num_layers) * kv_layer_bytes);
  std::vector<unsigned char> v_row(static_cast<size_t>(num_layers) * kv_layer_bytes);
  for (int32_t layer = 0; layer < num_layers; ++layer) {
    const size_t element_offset =
        (static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) + pos_id) *
        kKvRowElements;
    cuda_check(cuMemcpyDtoH(k_row.data() + static_cast<size_t>(layer) * kv_layer_bytes,
                            buffers.k_cache.get() + element_offset * sizeof(uint16_t),
                            kv_layer_bytes),
               "cuMemcpyDtoH(k_row)");
    cuda_check(cuMemcpyDtoH(v_row.data() + static_cast<size_t>(layer) * kv_layer_bytes,
                            buffers.v_cache.get() + element_offset * sizeof(uint16_t),
                            kv_layer_bytes),
               "cuMemcpyDtoH(v_row)");
  }
  write_binary(out / (prefix + "k_row.bin"), k_row.data(), k_row.size());
  write_binary(out / (prefix + "v_row.bin"), v_row.data(), v_row.size());

  constexpr size_t kHeads = 32;
  constexpr size_t kHeadDim = 64;
  std::vector<float> attn_out_p0(kHeads * kHeadDim);
  std::vector<float> attn_lse_p0(kHeads);
  for (size_t head = 0; head < kHeads; ++head) {
    const size_t out_element_offset =
        (head * static_cast<size_t>(num_sms)) * kHeadDim;
    cuda_check(cuMemcpyDtoH(attn_out_p0.data() + head * kHeadDim,
                            buffers.attn_out_intermediates.get() +
                                out_element_offset * sizeof(float),
                            kHeadDim * sizeof(float)),
               "cuMemcpyDtoH(attn_out_p0)");
    const size_t lse_element_offset = head * static_cast<size_t>(num_sms);
    cuda_check(cuMemcpyDtoH(attn_lse_p0.data() + head,
                            buffers.attn_lse_intermediates.get() +
                                lse_element_offset * sizeof(float),
                            sizeof(float)),
               "cuMemcpyDtoH(attn_lse_p0)");
  }
  write_binary(out / (prefix + "attn_out_p0.bin"), attn_out_p0.data(),
               attn_out_p0.size() * sizeof(float));
  write_binary(out / (prefix + "attn_lse_p0.bin"), attn_lse_p0.data(),
               attn_lse_p0.size() * sizeof(float));
}

struct TokenTiming {
  uint32_t token;
  uint32_t pos_id;
  float kernel_ms;
};

void write_timings(const fs::path& out,
                   const std::vector<TokenTiming>& timings) {
  double total_ms = 0.0;
  for (const TokenTiming& timing : timings) {
    total_ms += timing.kernel_ms;
  }
  std::ofstream output(out / "timings.json", std::ios::trunc);
  if (!output) {
    fail("cannot write timings.json");
  }
  output << std::setprecision(9) << "{\"tokens\":[";
  for (size_t i = 0; i < timings.size(); ++i) {
    if (i != 0) {
      output << ',';
    }
    output << "{\"t\":" << timings[i].token << ",\"pos_id\":"
           << timings[i].pos_id << ",\"kernel_ms\":"
           << timings[i].kernel_ms << '}';
  }
  output << "],\"total_ms\":" << total_ms << "}\n";
  if (!output) {
    fail("cannot finish timings.json");
  }
}

struct BenchBatch {
  uint64_t start_ns;
  uint64_t end_ns;
  float cuda_ms;
  uint32_t tokens;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const Manifest manifest = read_manifest(options.bundle);
    const Scalars scalars = read_scalars(options.bundle);

    const int32_t num_sms = scalar_i32(scalars, "num_sms");
    const int32_t num_instructions = scalar_i32(scalars, "num_instructions");
    const int32_t max_seq_len = scalar_i32(scalars, "max_seq_len");
    const int32_t rope_rows = scalar_i32(scalars, "rope_rows");
    const int32_t num_layers = scalar_i32(scalars, "num_layers");
    const int32_t bar_rows = scalar_i32(scalars, "bar_rows");
    const int32_t vocab_size = scalar_i32(scalars, "vocab_size");
    const float attn_scale = scalar_f32(scalars, "attn_scale");
    const float rms_norm_eps = scalar_f32(scalars, "rms_norm_eps");
    const int32_t skip_attn_reduction =
        scalar_i32(scalars, "skip_attn_reduction");
    const uint32_t scalar_ntok =
        static_cast<uint32_t>(scalar_i32(scalars, "ntok"));
    const uint32_t scalar_pos_id_first =
        static_cast<uint32_t>(scalar_i32(scalars, "pos_id_first"));

    if (num_sms <= 0 || num_instructions <= 0 || max_seq_len <= 0 ||
        rope_rows <= 0) {
      fail("num_sms, num_instructions, max_seq_len, and rope_rows must be positive");
    }
    if (num_layers != 16 || bar_rows != 10 || vocab_size != 128256) {
      fail("bundle does not have the llama-1b fixed dimensions "
           "(num_layers=16, bar_rows=10, vocab_size=128256)");
    }
    if (skip_attn_reduction != 0 && skip_attn_reduction != 1) {
      fail("skip_attn_reduction must be 0 or 1");
    }

    const size_t layers = static_cast<size_t>(num_layers);
    const size_t sms = static_cast<size_t>(num_sms);
    const size_t instructions = static_cast<size_t>(num_instructions);
    const size_t max_seq = static_cast<size_t>(max_seq_len);
    const size_t rope = static_cast<size_t>(rope_rows);
    const size_t vocab = static_cast<size_t>(vocab_size);

    const TensorSpec& qkv_spec =
        expect_tensor(manifest, "qkv_weights", "bf16", {layers, 3072, 2048});
    const TensorSpec& attn_norm_spec = expect_tensor(
        manifest, "attn_norm_weights", "bf16", {layers, 2048});
    const TensorSpec& o_spec =
        expect_tensor(manifest, "o_weights", "bf16", {layers, 2048, 2048});
    const TensorSpec& mlp_norm_spec = expect_tensor(
        manifest, "mlp_norm_weights", "bf16", {layers, 2048});
    const TensorSpec& up_spec =
        expect_tensor(manifest, "up_weights", "bf16", {layers, 8192, 2048});
    const TensorSpec& gate_spec = expect_tensor(
        manifest, "gate_weights", "bf16", {layers, 8192, 2048});
    const TensorSpec& down_spec = expect_tensor(
        manifest, "down_weights", "bf16", {layers, 2048, 8192});
    const TensorSpec& lm_norm_spec = expect_tensor(
        manifest, "lm_head_norm_weights", "bf16", {2048});
    const TensorSpec& lm_head_spec = expect_tensor(
        manifest, "lm_head_weights", "bf16", {vocab, 2048});
    const TensorSpec& k_cache_spec = expect_tensor(
        manifest, "k_cache", "bf16", {layers, max_seq, 8, 64});
    const TensorSpec& v_cache_spec = expect_tensor(
        manifest, "v_cache", "bf16", {layers, max_seq, 8, 64});
    const TensorSpec& rope_cos_spec =
        expect_tensor(manifest, "rope_cos", "f32", {rope, 64});
    const TensorSpec& rope_sin_spec =
        expect_tensor(manifest, "rope_sin", "f32", {rope, 64});
    const TensorSpec& instruction_spec = expect_tensor(
        manifest, "instructions", "i32", {sms, instructions, 32});
    const TensorSpec& embedding_spec = expect_tensor(
        manifest, "embeddings", "bf16", {static_cast<size_t>(scalar_ntok), 2048});

    const uint32_t ntok = options.ntok_set ? options.ntok : scalar_ntok;
    const uint32_t pos_id_first =
        options.pos_id_first_set ? options.pos_id_first : scalar_pos_id_first;
    if (ntok == 0 || ntok > scalar_ntok) {
      fail("requested --ntok is zero or exceeds embeddings rows");
    }
    if (options.mode == "check") {
      const uint64_t final_pos =
          static_cast<uint64_t>(pos_id_first) + static_cast<uint64_t>(ntok) - 1;
      if (final_pos >= static_cast<uint64_t>(max_seq_len) ||
          final_pos >= static_cast<uint64_t>(rope_rows)) {
        fail("check token position exceeds KV cache or rope rows");
      }
    } else if (options.bench_pos_id >= static_cast<uint32_t>(max_seq_len) ||
               options.bench_pos_id >= static_cast<uint32_t>(rope_rows)) {
      fail("--pos-id exceeds KV cache or rope rows");
    }

    cuda_check(cuInit(0), "cuInit");
    CUdevice device = 0;
    cuda_check(cuDeviceGet(&device, 0), "cuDeviceGet");
    int device_sms = 0;
    cuda_check(cuDeviceGetAttribute(&device_sms,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    device),
               "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)");
    if (device_sms != num_sms) {
      fail("visible device SM count " + std::to_string(device_sms) +
           " does not equal bundle num_sms " + std::to_string(num_sms) +
           "; MK launch rejected because instructions are indexed by %smid");
    }

    Context context(device);
    Module module(options.cubin, options.kernel);
    cuda_check(cuFuncSetAttribute(
                   module.function(),
                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                   static_cast<int>(options.dynamic_smem)),
               "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
    SharedLibrary provider_library(options.provider);
    verify_provider_abi(provider_library, options);
    plk_common_v2 common{};
    common.header = {PLK_HOST_ABI_V2, sizeof(common)};
    common.owner_id = options.owner_id;
    common.block = {options.threads, 1, 1};
    common.cluster = options.cluster;
    common.dynamic_smem_bytes = options.dynamic_smem;
    common.profile.dtypes = options.dtypes_digest;
    common.owner_contract = options.owner_contract;
    Stream stream;
    Event start_event;
    Event stop_event;

    RuntimeBuffers buffers{
        DeviceBuffer(checked_mul(checked_mul(layers, static_cast<size_t>(bar_rows), "Bar"), 48u, "Bar") * sizeof(uint32_t)),
        upload_instructions(instruction_spec, options.opcode_map),
        DeviceBuffer(checked_mul(checked_mul(sms, instructions, "timings"), 128u, "timings") * sizeof(int32_t)),
        upload_tensor(qkv_spec),
        upload_tensor(attn_norm_spec),
        upload_tensor(o_spec),
        upload_tensor(mlp_norm_spec),
        upload_tensor(up_spec),
        upload_tensor(gate_spec),
        upload_tensor(down_spec),
        upload_tensor(lm_norm_spec),
        upload_tensor(lm_head_spec),
        upload_tensor(k_cache_spec),
        upload_tensor(v_cache_spec),
        upload_tensor(rope_cos_spec),
        upload_tensor(rope_sin_spec),
        DeviceBuffer(2048u * sizeof(uint16_t)),
        DeviceBuffer(2048u * sizeof(uint16_t)),
        DeviceBuffer(2048u * sizeof(uint16_t)),
        DeviceBuffer(checked_mul(checked_mul(32u, sms, "attn_lse_intermediates"), sizeof(float), "attn_lse_intermediates")),
        DeviceBuffer(checked_mul(checked_mul(checked_mul(32u, sms, "attn_out_intermediates"), 64u, "attn_out_intermediates"), sizeof(float), "attn_out_intermediates")),
        DeviceBuffer(8192u * sizeof(uint16_t)),
        DeviceBuffer(vocab * sizeof(uint16_t))};

    zero_buffer(buffers.bar, "Bar");
    zero_buffer(buffers.timings, "timings");
    zero_buffer(buffers.hidden_states, "hidden_states");
    zero_buffer(buffers.q_post_rope, "q_post_rope");
    zero_buffer(buffers.attn_out, "attn_out");
    zero_buffer(buffers.attn_lse_intermediates, "attn_lse_intermediates");
    zero_buffer(buffers.attn_out_intermediates, "attn_out_intermediates");
    zero_buffer(buffers.silu_out, "silu_out");
    zero_buffer(buffers.logits, "logits");

    const std::vector<unsigned char> embeddings = read_bytes(embedding_spec);
    PinnedRow embedding_row(2048u * sizeof(uint16_t));
    AlignedParams params(options.params_bytes, options.params_align);
    MkProviderArgs provider_args = make_provider_args(
        buffers, num_sms, num_instructions, max_seq_len, rope_rows, num_layers,
        bar_rows, vocab_size, attn_scale, rms_norm_eps,
        skip_attn_reduction);

    std::error_code directory_error;
    fs::create_directories(options.out, directory_error);
    if (directory_error) {
      fail("cannot create output directory " + options.out.string() + ": " +
           directory_error.message());
    }

    auto prepare_token = [&](size_t embedding_index) {
      cuda_check(cuMemsetD32Async(buffers.bar.get(), 0,
                                  buffers.bar.size() / sizeof(uint32_t), stream),
                 "cuMemsetD32Async(Bar)");
      stage_embedding(embeddings, scalar_ntok, embedding_index, &embedding_row,
                      buffers.hidden_states, stream);
    };

    if (options.mode == "check") {
      std::vector<TokenTiming> timings;
      timings.reserve(ntok);
      for (uint32_t token = 0; token < ntok; ++token) {
        const uint32_t pos_id = pos_id_first + token;
        prepare_token(token);
        provider_args.pos_id = pos_id;
        ProviderSession session(provider_library, stream);
        session.open(common, provider_args, options, device_sms, params);
        cuda_check(cuEventRecord(start_event, stream), "cuEventRecord(start)");
        launch_kernel(module.function(), session.geometry(), stream, &params);
        cuda_check(cuEventRecord(stop_event, stream), "cuEventRecord(stop)");
        wait_with_watchdog(stop_event, "decode token");
        cuda_check(cuStreamSynchronize(stream), "cuStreamSynchronize(token)");
        session.release_after_sync();
        float kernel_ms = 0.0f;
        cuda_check(cuEventElapsedTime(&kernel_ms, start_event, stop_event),
                   "cuEventElapsedTime(token)");
        dump_outputs(options.out, token, pos_id, buffers, num_layers,
                     max_seq_len, num_sms, vocab_size);
        timings.push_back(TokenTiming{token, pos_id, kernel_ms});
        std::printf("token %u pos_id=%u kernel_ms=%.6f\n", token, pos_id,
                    kernel_ms);
      }
      write_timings(options.out, timings);
      return 0;
    }

    auto run_bench_batch = [&](bool record) -> BenchBatch {
      const uint64_t start_ns = epoch_ns();
      provider_args.pos_id = options.bench_pos_id;
      ProviderSession session(provider_library, stream);
      session.open(common, provider_args, options, device_sms, params);
      cuda_check(cuEventRecord(start_event, stream), "cuEventRecord(batch start)");
      for (uint32_t token = 0; token < options.bench_batch_tokens; ++token) {
        const size_t embedding_index = token % static_cast<size_t>(scalar_ntok);
        prepare_token(embedding_index);
        launch_kernel(module.function(), session.geometry(), stream, &params);
        if (options.sync_each) {
          cuda_check(cuEventRecord(stop_event, stream),
                     "cuEventRecord(token stop)");
          wait_with_watchdog(stop_event, "bench token");
          cuda_check(cuStreamSynchronize(stream),
                     "cuStreamSynchronize(bench token)");
        }
      }
      cuda_check(cuEventRecord(stop_event, stream), "cuEventRecord(batch stop)");
      wait_with_watchdog(stop_event, record ? "bench batch" : "discard batch");
      cuda_check(cuStreamSynchronize(stream), "cuStreamSynchronize(batch)");
      session.release_after_sync();
      const uint64_t end_ns = epoch_ns();
      float cuda_ms = 0.0f;
      cuda_check(cuEventElapsedTime(&cuda_ms, start_event, stop_event),
                 "cuEventElapsedTime(batch)");
      return BenchBatch{start_ns, end_ns, cuda_ms,
                        options.bench_batch_tokens};
    };

    const auto discard_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         discard_start)
               .count() < options.bench_discard) {
      run_bench_batch(false);
    }

    std::vector<BenchBatch> batches;
    const auto measure_start = std::chrono::steady_clock::now();
    do {
      batches.push_back(run_bench_batch(true));
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           measure_start)
                 .count() < options.bench_seconds);

    std::cout << std::setprecision(9) << "{\"batches\":[";
    for (size_t i = 0; i < batches.size(); ++i) {
      if (i != 0) {
        std::cout << ',';
      }
      std::cout << "{\"start_ns\":" << batches[i].start_ns
                << ",\"end_ns\":" << batches[i].end_ns
                << ",\"cuda_ms\":" << batches[i].cuda_ms
                << ",\"tokens\":" << batches[i].tokens << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "runner error: %s\n", error.what());
    return 1;
  }
}
