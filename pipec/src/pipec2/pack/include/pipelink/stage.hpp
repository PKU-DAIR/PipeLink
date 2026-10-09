#pragma once
#include "tensor_source.h"
#include "host_view.h"

// The device object never includes registration machinery or evaluates annotations.
#if defined(PIPELINK_EMIT)
#include <pipelink_host_abi_v2.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
struct __half;
struct __nv_bfloat16;
#endif

namespace pl {
// Strings and initializer-list aggregates (including nested trees) are borrowed
// views. Named constexpr annotations need static backing lists at every nesting
// level; inline PL_* arguments live until registration copies their data.
// Device builders own scalar fields and endpoint binds without allocating.
using strings = std::initializer_list<const char*>;
// A sub-byte channel can name an alias of host_storage<uint8_t, 2>: two
// logical storage elements in one compiler-sized C storage unit.
template <typename T, unsigned Elements> struct host_storage {};
struct coord { const char* name; int lo, hi; };
struct values { const char* name; const char* sort; std::initializer_list<int> elements; };
struct binding_input { const char* name; const char* sort = "int"; };
constexpr binding_input bind(const char* name, const char* sort = "int") { return {name, sort}; }
struct bound {
  int value;
  constexpr operator int() const { return value; }
};
struct identity {
  const char* result; const char* work; const char* version; const char* dtype;
  const char* interpretation; std::initializer_list<coord> coords;
};
struct readiness { const char* after; const char* reuse_after; };
// Event-bounded claim storage, relative to this stage's tensor export/import.
// An omitted end lasts through kernel end. Before arriving at `to`, the owner
// completes all accesses in every proxy (including TMA, cp.async, WGMMA/tcgen05),
// and invalidates embedded mbarriers only after a collective point following
// every owner's wait/arrive. release() must not touch the ended range.
// Accessing threads outside the generated event cohort also issue
// fence.proxy.async.shared::cta after their last generic access; the linker
// fences cohort threads. No access may precede observing `from`.
// Readiness is published once per CTA. Export backing must remain live through
// completion; dead operand storage is not backing for register/TMEM exports.
struct during { const char* tensor; const char* from; const char* to = nullptr; };
struct calls { const char* produce; const char* release = nullptr; };
struct sync { int generations, capacity; };
struct lend {
  const char* name; const char* domain; int extent, align;
  const char* scope; const char* binding; const char* handle;
  std::optional<int> base; const char* owner; const char* from; const char* to;
  std::optional<pl::sync> sync = std::nullopt;
};
using lends = std::initializer_list<lend>;
constexpr lend smem(const char* name, int extent, int align, const char* handle,
                    const char* owner, const char* from, const char* to) {
  return {name, "smem", extent, align, "cta", "relocatable", handle,
          std::nullopt, owner, from, to};
}
constexpr lend regs(const char* name, int extent, const char* handle,
                    const char* owner, const char* from, const char* to) {
  return {name, "reg", extent, 1, "thread", "fixed", handle, std::nullopt, owner, from, to};
}
constexpr lend frame(const char* name, int words, const char* handle,
                     const char* owner, const char* from, const char* to) {
  return {name, "reg", words, 1, "thread", "caller_frame", handle, std::nullopt, owner, from, to};
}
constexpr lend metadata(const char* name, int extent, const char* handle,
                        const char* owner, const char* from, const char* to) {
  return {name, "metadata", extent, 1, "warp", "fixed", handle, std::nullopt, owner, from, to};
}
constexpr lend cta_barrier(const char* name, int base, const char* owner,
                           const char* from, const char* to, pl::sync sync) {
  return {name, "barrier", 1, 1, "cta", "fixed", nullptr, base, owner, from, to, sync};
}
struct arg { const char* name; const char* meaning; };
using args = std::initializer_list<arg>;
// "enter_once": one site entry per CTA. "single_completion": the import's
// completion helper is called exactly once per entry, with no later import call.
using guarantees = strings;
struct fixed_binding { const char* name; int value; };
using fixed_bindings = std::initializer_list<fixed_binding>;
inline constexpr const char* produce = "produce";
inline constexpr const char* consume = "consume";
struct all_threads_t {};
inline constexpr all_threads_t all_threads{};
struct launch { int x_multiple_of = 1; std::array<int, 3> max{1, 1, 1}; };
struct dynamic_t {};
inline constexpr dynamic_t dynamic{};
struct cluster {
  std::array<int, 3> compiled{1, 1, 1};
  pl::launch launch{};
  bool is_dynamic = false;
  constexpr cluster() = default;
  constexpr cluster(std::array<int, 3> dims, pl::launch limits = {}) : compiled(dims), launch(limits) {}
  constexpr cluster(dynamic_t, pl::launch limits = {}) : launch(limits), is_dynamic(true) {}
};
struct topology {
  int cta_group = 1; pl::cluster cluster{}; bool persistent = false;
  const char* scheduler = "static"; strings multicast{};
};

struct tree {
  int value = 0;
  bool is_leaf = false;
  std::initializer_list<tree> children{};
  constexpr tree() = default;
  constexpr tree(int value) : value(value), is_leaf(true) {}
  constexpr tree(std::initializer_list<tree> children) : children(children) {}
};
struct swizzle { int B, M, S; const char* unit = nullptr; };
struct affine {
  tree shape, stride;
  int elem_bits;
  const char* atom_name = nullptr;
  std::optional<pl::swizzle> swizzle_value{};
  std::optional<int> origin{}, pointer_flags{};
  constexpr affine(tree shape, tree stride, int elem_bits) : shape(shape), stride(stride), elem_bits(elem_bits) {}
  constexpr affine& atom(const char* name) { atom_name = name; return *this; }
  constexpr affine& swizzle(int B, int M, int S, const char* unit = nullptr) {
    swizzle_value = std::optional<pl::swizzle>{{B, M, S, unit}}; return *this;
  }
  constexpr affine& origin_bits(int value) { origin = std::optional<int>{value}; return *this; }
  constexpr affine& pointer_flag_bits(int value) { pointer_flags = std::optional<int>{value}; return *this; }
};
struct tmem { int lanes_lo, lanes_hi, cols, elem_bits; };
struct integer_or_text {
  int value = 0;
  const char* text = nullptr;
  constexpr integer_or_text(int value) : value(value) {}
  constexpr integer_or_text(const char* text) : text(text) {}
};
struct arrive {
  const char* by;
  integer_or_text count;
  constexpr arrive(const char* by, integer_or_text count) : by(by), count(count) {}
};
struct tile_t {};
inline constexpr tile_t tile{};
struct sum {
  strings endpoints;
  constexpr sum(strings endpoints) : endpoints(endpoints) {}
};
struct tx_bytes {
  enum class kind { constant, tile, sum };
  kind tag;
  int value = 0;
  strings endpoints{};
  constexpr tx_bytes(int value) : tag(kind::constant), value(value) {}
  constexpr tx_bytes(tile_t) : tag(kind::tile) {}
  constexpr tx_bytes(pl::sum value) : tag(kind::sum), endpoints(value.endpoints) {}
};
struct barrier_storage { const char* handle; int offset_bytes; int stride_bytes; };
struct side {
  const char* slots; pl::arrive arrive; pl::tx_bytes tx_bytes;
  std::optional<barrier_storage> storage{};
  constexpr side& native(barrier_storage value) {
    storage = std::optional<barrier_storage>{value}; return *this;
  }
};
struct member { const char* member_of; pl::tx_bytes tx_bytes; };
struct full_side {
  const char* slots = nullptr;
  pl::arrive arrive{nullptr, 0};
  pl::tx_bytes tx_bytes;
  const char* member_of = nullptr;
  std::optional<barrier_storage> storage{};
  constexpr full_side(side value)
      : slots(value.slots), arrive(value.arrive), tx_bytes(value.tx_bytes), storage(value.storage) {}
  constexpr full_side(member value) : tx_bytes(value.tx_bytes), member_of(value.member_of) {}
};
struct init {
  const char* owner;
  integer_or_text phase;
  const char* materializer;
  std::optional<int> warp{};
};
struct mbarrier {
  full_side full;
  std::optional<side> empty;
  pl::init init;
  const char* release = "once-per-acquire";
  constexpr mbarrier(full_side full, side empty, pl::init init, const char* release = "once-per-acquire")
      : full(full), empty(empty), init(init), release(release) {}
  constexpr mbarrier(full_side full, std::nullopt_t, pl::init init, const char* release)
      : full(full), empty(std::nullopt), init(init), release(release) {}
};
struct named_barrier { const char* id; int count; };
struct counter {
  const char* slot;
  const char* release = "red.release.gpu";
  const char* acquire = "ld.acquire.gpu";
};
// System-scope counter (peer-visible release/acquire); the endpoint must use .scope("system").
struct counter_sys : counter {
  constexpr explicit counter_sys(const char* slot) : counter{slot, "red.release.sys", "ld.acquire.sys"} {}
};
struct fixed { int value; constexpr explicit fixed(int value) : value(value) {} };
struct handle { const char* name; constexpr explicit handle(const char* name) : name(name) {} };
struct ring {
  int depth = 0;
  const char* depth_handle = nullptr;
  int stage_stride_bytes;
  constexpr ring(fixed depth, int stride) : depth(depth.value), stage_stride_bytes(stride) {}
  constexpr ring(handle depth, int stride) : depth_handle(depth.name), stage_stride_bytes(stride) {}
};

namespace detail {
struct unspecified {};
struct no_layout {};
struct endpoint_fields {
  const char* name; const char* direction; const char* role;
  const char* space = nullptr; const char* scope = nullptr; const char* proxy = nullptr;
  const char* dtype = nullptr;
  std::initializer_list<int> shape{};
  std::optional<pl::ring> ring{};
  strings exports{};
};
struct endpoint_bind { const char* key; const char* handle; };
}

// Layout/protocol and bind types follow the builder, so missing protocols are a
// compile-time error and bind chains have no fixed capacity or borrowed nodes.
template <typename Layout = detail::unspecified, typename Protocol = detail::unspecified,
          typename Binds = std::tuple<>> struct endpoint {
  detail::endpoint_fields fields;
  Layout layout_value;
  Protocol protocol_value;
  Binds binds;
  constexpr endpoint(const char* name, const char* direction, const char* role)
      : fields{name, direction, role}, layout_value{}, protocol_value{}, binds{} {}
  // Role-less form for non-register endpoints inside exported callables.
  constexpr endpoint(const char* name, const char* direction)
      : fields{name, direction, nullptr}, layout_value{}, protocol_value{}, binds{} {}
  constexpr endpoint(detail::endpoint_fields fields, Layout layout, Protocol protocol, Binds binds)
      : fields(fields), layout_value(layout), protocol_value(protocol), binds(binds) {}
  constexpr endpoint& space(const char* value) { fields.space = value; return *this; }
  constexpr endpoint& scope(const char* value) { fields.scope = value; return *this; }
  constexpr endpoint& proxy(const char* value) { fields.proxy = value; return *this; }
  constexpr endpoint& tile(const char* dtype, std::initializer_list<int> shape) {
    static_assert(!std::is_same_v<Layout, detail::no_layout>, "pl::signal_endpoint does not accept .tile()");
    fields.dtype = dtype; fields.shape = shape; return *this;
  }
  template <typename T> constexpr auto layout(T value) const {
    static_assert(!std::is_same_v<Layout, detail::no_layout>, "pl::signal_endpoint does not accept .layout()");
    static_assert(std::is_same_v<T, affine> || std::is_same_v<T, tmem>, "PL_ENDPOINT layout requires pl::affine or pl::tmem");
    return endpoint<T, Protocol, Binds>{fields, value, protocol_value, binds};
  }
  template <typename T> constexpr auto protocol(T value) const {
    if constexpr (std::is_same_v<T, counter_sys>) {
      return protocol(static_cast<counter>(value));
    } else {
      static_assert(std::is_same_v<T, mbarrier> || std::is_same_v<T, named_barrier> || std::is_same_v<T, counter>,
                    "PL_ENDPOINT protocol requires pl::mbarrier, pl::named_barrier, pl::counter or pl::counter_sys");
      static_assert(!std::is_same_v<Layout, detail::no_layout> || std::is_same_v<T, mbarrier>,
                    "pl::signal_endpoint requires pl::mbarrier");
      return endpoint<Layout, T, Binds>{fields, layout_value, value, binds};
    }
  }
  constexpr endpoint& ring(pl::ring value) { fields.ring = std::optional<pl::ring>{value}; return *this; }
  // Provided exports of this stage whose bodies use this endpoint: it leaves the interface
  // of a pipeline in which every one of them is unreferenced.
  constexpr endpoint& exports(strings names) { fields.exports = names; return *this; }
  template <typename Depth> constexpr endpoint& ring(Depth depth, int stride) { return ring(pl::ring(depth, stride)); }
  constexpr auto bind(const char* key, const char* handle) const {
    auto next = std::tuple_cat(binds, std::make_tuple(detail::endpoint_bind{key, handle}));
    return endpoint<Layout, Protocol, decltype(next)>{fields, layout_value, protocol_value, next};
  }
};
endpoint(const char*, const char*, const char*) -> endpoint<>;

// Completion-only endpoints have no data footprint; the full mbarrier is their
// only resource. Release "never" is one-shot; "external" delegates reuse ordering.
// Inside an exported callable the role may be omitted (runs in the caller's role).
constexpr auto signal_endpoint(const char* name, const char* direction, const char* role = nullptr) {
  return endpoint<detail::no_layout>{name, direction, role}
      .space("none").scope("cta").proxy("generic").ring(pl::fixed(1), 0);
}

struct claim {
  const char* handle; const char* domain;
  int constant = 0, alignment = 1;
  fixed_bindings coefficients{}, table_values{};
  bool is_table = false;
  const char* lifetime_value = "kernel";
  std::optional<during> tensor_lifetime{};
  std::optional<int> absolute_base{}, offset_value{};
  const char* alias_value = nullptr;
  strings role_names{};
  constexpr claim(const char* handle, const char* domain) : handle(handle), domain(domain) {}
  constexpr claim& extent(int value, fixed_bindings per = {}) {
    constant = value; coefficients = per; is_table = false; return *this;
  }
  constexpr claim& table(fixed_bindings values) { table_values = values; is_table = true; return *this; }
  constexpr claim& align(int value) { alignment = value; return *this; }
  constexpr claim& lifetime(const char* value) {
    lifetime_value = value; tensor_lifetime = std::optional<during>{}; return *this;
  }
  constexpr claim& lifetime(during value) {
    tensor_lifetime = std::optional<during>{value}; return *this;
  }
  constexpr claim& at(int value) { absolute_base = std::optional<int>{value}; return *this; }
  constexpr claim& offset(int value) { offset_value = std::optional<int>{value}; return *this; }
  constexpr claim& alias(const char* value) { alias_value = value; return *this; }
  constexpr claim& roles(strings values) { role_names = values; return *this; }
};

namespace detail {
// nvcc's host lowering cannot name an inline lambda type in a CTAD result.
// Captureless lambdas have a constexpr function-pointer conversion; named
// stateful callables keep their concrete type and are copied into the registry.
template <typename T, typename = void> struct map_type { using type = T; };
template <typename T> struct map_type<T, std::void_t<decltype(+std::declval<T>())>> {
  using type = decltype(+std::declval<T>());
};
template <typename T> using map_type_t = typename map_type<T>::type;
}

template <typename Fn, typename... Inputs> struct placement {
  const char* space; const char* units; strings outputs; Fn map; std::tuple<Inputs...> inputs;
  constexpr placement(const char* space, const char* units, strings outputs, Fn map, Inputs... inputs)
      : space(space), units(units), outputs(outputs), map(map), inputs(inputs...) {}
};
template <typename Fn, typename... Inputs>
placement(const char*, const char*, strings, Fn, Inputs...) -> placement<detail::map_type_t<Fn>, Inputs...>;
// The final argument is the map; preceding arguments are its ordered domains.
template <typename... Parts> struct demand_positions {
  std::tuple<Parts...> parts;
  constexpr demand_positions(Parts... parts) : parts(parts...) {}
};
template <typename... Parts> demand_positions(Parts...) -> demand_positions<detail::map_type_t<Parts>...>;
// Domains first, followed by space, units, and the map.
template <typename... Parts> struct demand_view {
  std::tuple<Parts...> parts;
  constexpr demand_view(Parts... parts) : parts(parts...) {}
};
template <typename... Parts> demand_view(Parts...) -> demand_view<detail::map_type_t<Parts>...>;
template <typename Domain, typename Fn> struct participants {
  Domain domain; Fn map;
  constexpr participants(Domain domain, Fn map) : domain(domain), map(map) {}
};
template <typename Domain, typename Fn> participants(Domain, Fn) -> participants<Domain, detail::map_type_t<Fn>>;
struct rendezvous {
  const char* event;
  const char* barrier;
  bool all = false;
  std::vector<int> cohort;
  rendezvous(const char* event, const char* barrier, all_threads_t)
      : event(event), barrier(barrier), all(true) {}
  template <typename Domain, typename Fn>
  rendezvous(const char* event, const char* barrier, const participants<Domain, Fn>& members)
      : event(event), barrier(barrier) {
    auto append = [&](int i) {
      if constexpr (std::is_same_v<decltype(members.map(i)), bool>) {
        if (members.map(i)) cohort.push_back(i);
      } else cohort.push_back(members.map(i));
    };
    if constexpr (std::is_same_v<Domain, coord>)
      for (int i = members.domain.lo; i < members.domain.hi; ++i) append(i);
    else for (int i : members.domain.elements) append(i);
  }
};

namespace detail {
static const char translation_unit = 0;
inline std::string quote(const std::string& value) {
  std::string out = "\"";
  for (unsigned char c : value) {
    if (c == '\\' || c == '"') { out += '\\'; out += c; }
    else if (c < 32) {
      constexpr char hex[] = "0123456789abcdef";
      out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
    } else out += c;
  }
  return out + '"';
}
inline std::string json(strings items) {
  std::string out = "[";
  for (const auto* item : items) { if (out.size() > 1) out += ','; out += quote(item); }
  return out + ']';
}
inline std::string json(const std::vector<std::string>& items) {
  std::string out = "[";
  for (const auto& item : items) { if (out.size() > 1) out += ','; out += quote(item); }
  return out + ']';
}
template <typename Range> std::string integers(const Range& items) {
  std::string out = "[";
  for (auto item : items) { if (out.size() > 1) out += ','; out += std::to_string(item); }
  return out + ']';
}
inline std::string field(const char* name, const std::string& value) { return quote(name) + ':' + value; }
inline std::string text(const char* name, const char* value) { return field(name, quote(value)); }
template <typename Int> std::string number(const char* name, Int value) { return field(name, std::to_string(value)); }
struct record {
  std::string kind, name, fields;
  std::function<void(std::ostream&)> write;
};
struct domain {
  std::string name, sort;
  int lo = 0, hi = 0;
  std::vector<int> values;
  bool finite = false, binding = false, known = true;
};
inline domain as_domain(coord x) { return {x.name, "int", x.lo, x.hi, {}, false, false, true}; }
inline domain as_domain(values x) { return {x.name, x.sort, 0, 0, x.elements, true, false, true}; }
inline domain as_domain(binding_input x) { return {x.name, x.sort, 0, 0, {}, false, true, false}; }
inline std::string json(const domain& d, bool role = false) {
  auto out = "{" + field("name", quote(d.name)) + ',' + field("sort", quote(d.sort));
  if (role) out += ',' + text("role", d.binding ? "bind" : "coord");
  if (d.finite) out += ',' + field("values", integers(d.values));
  else out += ",\"lo\":" + std::to_string(d.lo) + ",\"hi\":" + std::to_string(d.hi);
  return out + '}';
}
struct params_state { bool declared = false; };
static params_state translation_unit_params;
struct registry_data {
  std::vector<record> records;
  std::vector<std::pair<const void*, domain>> bindings;
  std::vector<std::pair<std::string, int>> roles;
  std::exception_ptr error;
};
inline registry_data& registry() { static registry_data value; return value; }
inline int add(const char* kind, const std::string& name, std::string fields) {
  registry().records.push_back({kind, name, std::move(fields), {}});
  return 0;
}
// Preserve registration errors until main(), where the runner can report them
// without aborting during static initialization.
template <typename Fn> int checked_registration(Fn fn) {
  try { return fn(); }
  catch (...) {
    if (!registry().error) registry().error = std::current_exception();
    return 0;
  }
}
inline int hint(const char* key, const char* value) {
  return add("hint", key, text("key", key) + ',' + text("value", value));
}
template <typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
int hint(const char* key, Int value) {
  if constexpr (std::is_same_v<Int, bool>)
    return add("hint", key, text("key", key) + ',' + field("value", value ? "true" : "false"));
  else return add("hint", key, text("key", key) + ',' + number("value", value));
}
inline std::string json(const identity& x) {
  auto out = "{" + text("result", x.result) + ',' + text("work", x.work) + ',' + text("version", x.version)
      + ',' + text("dtype", x.dtype) + ',' + text("interpretation", x.interpretation) + ",\"coords\":[";
  bool first = true;
  for (auto c : x.coords) {
    if (!first) out += ','; first = false;
    out += json(as_domain(c));
  }
  return out + "]}";
}
inline std::string json(lends items) {
  std::string out = "[";
  for (const auto& x : items) {
    if (out.size() > 1) out += ',';
    out += "{" + text("name", x.name) + ',' + text("domain", x.domain) + ',' + number("extent", x.extent)
        + ',' + number("align", x.align) + ',' + text("scope", x.scope) + ',' + text("binding", x.binding)
        + ",\"location\":{\"handle\":" + (x.handle ? quote(x.handle) : "null")
        + ",\"base\":" + (x.base ? std::to_string(*x.base) : "null") + "}," + text("owner", x.owner)
        + ",\"lifetime\":{" + text("from", x.from) + ',' + text("to", x.to) + "},\"sync\":";
    if (x.sync) out += "{" + number("generations", x.sync->generations) + ',' + number("capacity", x.sync->capacity) + '}';
    else out += "null";
    out += '}';
  }
  return out + ']';
}

inline std::string json(const tree& x) {
  if (x.is_leaf) return std::to_string(x.value);
  std::string out = "[";
  for (const auto& child : x.children) { if (out.size() > 1) out += ','; out += json(child); }
  return out + ']';
}
inline std::string json(const affine& x) {
  auto out = text("encoding", "pl-layout-v2") + ',' + field("shape", json(x.shape))
      + ',' + field("stride", json(x.stride)) + ',' + number("elem_bits", x.elem_bits);
  if (x.atom_name) out += ',' + text("atom", x.atom_name);
  if (x.swizzle_value) {
    const auto& s = *x.swizzle_value;
    out += ",\"swizzle\":{" + number("B", s.B) + ',' + number("M", s.M) + ',' + number("S", s.S);
    if (s.unit) out += ',' + text("unit", s.unit);
    out += '}';
  }
  if (x.origin) out += ',' + number("origin_bits", *x.origin);
  if (x.pointer_flags) out += ',' + number("pointer_flag_bits", *x.pointer_flags);
  return '{' + out + '}';
}
inline std::string json(const tmem& x) {
  return "{" + text("encoding", "pl-layout-v2") + ",\"lanes\":[" + std::to_string(x.lanes_lo)
      + ',' + std::to_string(x.lanes_hi) + "]," + number("cols", x.cols) + ',' + number("elem_bits", x.elem_bits) + '}';
}
inline std::string json(integer_or_text x) { return x.text ? quote(x.text) : std::to_string(x.value); }
inline std::string json(const tx_bytes& x) {
  if (x.tag == tx_bytes::kind::tile) return quote("tile");
  if (x.tag == tx_bytes::kind::sum) return "{" + field("sum", json(x.endpoints)) + '}';
  return std::to_string(x.value);
}
inline std::string json(const arrive& x) {
  if (!x.by) throw std::runtime_error("PL_ENDPOINT mbarrier side requires arrive(by, count)");
  return "{" + text("by", x.by) + ',' + field("count", json(x.count)) + '}';
}
inline std::string json(const barrier_storage& x) {
  return "{" + text("base", x.handle) + ',' + number("offset_bytes", x.offset_bytes)
      + ',' + number("stride_bytes", x.stride_bytes) + '}';
}
inline std::string json(const full_side& x) {
  if (x.member_of) {
    if (x.slots || x.arrive.by)
      throw std::runtime_error("PL_ENDPOINT mbarrier full: slots/arrive and member_of are mutually exclusive; use pl::side or pl::member");
    if (x.tx_bytes.tag == tx_bytes::kind::sum)
      throw std::runtime_error("PL_ENDPOINT mbarrier member_of: tx_bytes must be an integer or pl::tile, not pl::sum");
    return "{" + text("member_of", x.member_of) + ',' + field("tx_bytes", json(x.tx_bytes)) + '}';
  }
  if (!x.slots) throw std::runtime_error("PL_ENDPOINT mbarrier side requires slots or member_of");
  return "{" + text("slots", x.slots) + ',' + field("arrive", json(x.arrive))
      + ',' + field("tx_bytes", json(x.tx_bytes))
      + (x.storage ? ',' + field("native", json(*x.storage)) : "") + '}';
}
inline std::string json(const mbarrier& x) {
  if (!x.init.owner && !x.init.warp)
    throw std::runtime_error("PL_ENDPOINT mbarrier init requires owner or warp");
  auto out = text("kind", "mbarrier") + ',' + field("full", json(x.full));
  if (x.empty) out += ',' + field("empty", json(full_side{*x.empty}));
  out += ",\"init\":{";
  if (x.init.owner) out += text("owner", x.init.owner) + ',';
  out += field("phase", json(x.init.phase)) + ',' + text("materializer", x.init.materializer);
  if (x.init.warp) out += ',' + number("warp", *x.init.warp);
  return '{' + out + "}," + text("release", x.release) + '}';
}
inline std::string json(const named_barrier& x) {
  return "{" + text("kind", "named_barrier") + ',' + text("id", x.id) + ',' + number("count", x.count) + '}';
}
inline std::string json(const counter& x) {
  return "{" + text("kind", "counter") + ',' + text("slot", x.slot)
      + ',' + text("release", x.release) + ',' + text("acquire", x.acquire) + '}';
}
inline std::string json(const pl::ring& x) {
  return "{\"depth\":{" + (x.depth_handle ? text("handle", x.depth_handle) : number("fixed", x.depth))
      + "}," + number("stage_stride_bytes", x.stage_stride_bytes) + '}';
}
inline std::string json(fixed_bindings items) {
  std::string out = "{";
  for (auto x : items) { if (out.size() > 1) out += ','; out += number(x.name, x.value); }
  return out + '}';
}
template <typename Layout, typename Protocol, typename Binds>
int endpoint(const char* name, const pl::endpoint<Layout, Protocol, Binds>& x) {
  static_assert(!std::is_same_v<Layout, unspecified>, "PL_ENDPOINT requires .layout(pl::affine{...}) or .layout(pl::tmem{...})");
  static_assert(!std::is_same_v<Protocol, unspecified>, "PL_ENDPOINT requires .protocol(pl::mbarrier{...}), .protocol(pl::named_barrier{...}) or .protocol(pl::counter{...})");
  static_assert(!std::is_same_v<Layout, no_layout> || std::is_same_v<Protocol, mbarrier>,
                "pl::signal_endpoint requires pl::mbarrier");
  if constexpr (!std::is_same_v<Layout, unspecified> && !std::is_same_v<Protocol, unspecified>) {
    const auto& f = x.fields;
    if (!f.name || name != std::string(f.name))
      throw std::runtime_error(std::string("PL_ENDPOINT ") + name + ": macro name must match endpoint builder name");
    // An endpoint inside an exported callable may omit its role (nullptr): it runs in the caller's
    // role. Register tiles keep their role, which fixes their thread-value layout.
    const bool roleless = !f.role && f.space && std::strcmp(f.space, "reg") != 0;
    if (!f.direction || (!f.role && !roleless) || !f.space || !f.scope || !f.proxy)
      throw std::runtime_error(std::string("PL_ENDPOINT ") + name + ": requires direction, role, space, scope and proxy");
    auto out = text("name", name) + ',' + text("direction", f.direction) + ','
        + (roleless ? field("role", "null") : text("role", f.role))
        + ',' + text("space", f.space) + ',' + text("scope", f.scope) + ',' + text("proxy", f.proxy);
    if constexpr (std::is_same_v<Layout, no_layout>) {
      if (std::strcmp(f.space, "none") != 0 || f.dtype || f.shape.size())
        throw std::runtime_error(std::string("PL_ENDPOINT ") + name + ": signal endpoint requires space none and no tile");
    } else {
      if (!f.dtype || !f.shape.size())
        throw std::runtime_error(std::string("PL_ENDPOINT ") + name + ": requires tile");
      out += ",\"tile\":{" + text("dtype", f.dtype) + ',' + field("shape", integers(f.shape)) + "}"
          + ',' + field("layout", json(x.layout_value));
    }
    out += ',' + field("protocol", json(x.protocol_value));
    if (f.ring) out += ',' + field("ring", json(*f.ring));
    if (f.exports.size()) out += ',' + field("exports", json(f.exports));
    std::string binds = "{";
    std::apply([&](const auto&... b) {
      auto append = [&](const auto& value) {
        if (binds.size() > 1) binds += ',';
        binds += text(value.key, value.handle);
      };
      (append(b), ...);
    }, x.binds);
    return add("endpoint", name, out + ',' + field("binds", binds + '}'));
  }
  return 0;
}

template <typename T> struct registered_type {
  static constexpr const char* name = nullptr;
  static constexpr const char* members = nullptr;
};
template <typename T> int register_type(const char* name) {
  static_assert(std::is_class_v<T>, "PL_TYPE requires a class or struct type");
  static_assert(std::is_standard_layout_v<T>, "PL_TYPE requires a standard-layout type");
  static_assert(std::is_trivially_copyable_v<T>, "PL_TYPE requires a trivially copyable type");
  return add("type", name, text("name", name) + ',' + number("bytes", sizeof(T)) + ',' + number("align", alignof(T))
      + ',' + text("members", registered_type<T>::members));
}
template <typename T> void type_prelude(std::string& out) {
  if constexpr (std::is_reference_v<T> || std::is_const_v<T>) {
    type_prelude<std::remove_const_t<std::remove_reference_t<T>>>(out);
  } else if constexpr (std::is_pointer_v<T>) {
    type_prelude<std::remove_pointer_t<T>>(out);
  } else if constexpr (registered_type<T>::name != nullptr) {
    std::string name = registered_type<T>::name;
    auto split = name.rfind("::");
    auto definition = "struct " + (split == std::string::npos ? name : name.substr(split + 2))
        + " { " + registered_type<T>::members + " };";
    if (split != std::string::npos) definition = "namespace " + name.substr(0, split) + " { " + definition + " }";
    std::string guard = name;
    for (auto& c : guard) {
      if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
      else if (!(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9')) c = '_';
    }
    guard += "_DEFINED";
    definition = "#ifndef " + guard + "\n#define " + guard + '\n' + definition + "\n#endif\n";
    if (out.find(definition) == std::string::npos) out += definition;
  }
}
template <typename T, int Index = -1> std::string type_name() {
  if constexpr (std::is_lvalue_reference_v<T>) return type_name<std::remove_reference_t<T>, Index>() + "&";
  else if constexpr (std::is_rvalue_reference_v<T>) return type_name<std::remove_reference_t<T>, Index>() + "&&";
  else if constexpr (std::is_const_v<T>) {
    if constexpr (std::is_pointer_v<std::remove_const_t<T>>) return type_name<std::remove_const_t<T>, Index>() + " const";
    else return "const " + type_name<std::remove_const_t<T>, Index>();
  } else if constexpr (std::is_pointer_v<T>) return type_name<std::remove_pointer_t<T>, Index>() + "*";
  else if constexpr (std::is_same_v<T, void>) return "void";
  else if constexpr (std::is_same_v<T, pipelink_tensor_source>) return "pipelink_tensor_source";
  else if constexpr (std::is_same_v<T, bool>) return "bool";
  else if constexpr (std::is_same_v<T, char>) return "char";
  else if constexpr (std::is_same_v<T, float>) return "float";
  else if constexpr (std::is_same_v<T, double>) return "double";
  else if constexpr (std::is_integral_v<T> && !std::is_volatile_v<T>
      && !std::is_same_v<T, wchar_t> && !std::is_same_v<T, char16_t> && !std::is_same_v<T, char32_t>) {
    return std::string(std::is_signed_v<T> ? "int" : "uint") + std::to_string(sizeof(T) * 8) + "_t";
  }
#if defined(__CUDACC__)
  else if constexpr (std::is_same_v<T, __half>) return "__half";
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) return "__nv_bfloat16";
#endif
  else if constexpr (registered_type<T>::name != nullptr) return registered_type<T>::name;
  else {
    static_assert(!std::is_same_v<T, T>, "PipeLink unsupported parameter type; register class types with PL_TYPE(T, members...); parameter index is Index in type_name<T, Index> (return type: -1)");
    return {};
  }
}
template <typename> struct declaration;
template <typename R, typename... Args> struct declaration<R (*)(Args...)> {
  static constexpr std::size_t arity = sizeof...(Args);
  template <std::size_t... I> static auto parameter_types(std::index_sequence<I...>) {
    return std::array<std::string, sizeof...(Args)>{type_name<Args, int(I)>()...};
  }
  static auto parameter_types() { return parameter_types(std::index_sequence_for<Args...>{}); }
  static std::string print(const char* symbol) {
    std::string out;
    type_prelude<R>(out);
    (type_prelude<Args>(out), ...);
    out += "extern \"C\" __device__ " + type_name<R>() + " " + symbol + "(";
    bool first = true;
    for (auto& type : parameter_types()) { if (!first) out += ", "; first = false; out += type; }
    return out + ");";
  }
};
template <typename R, typename... Args>
struct declaration<R (*)(Args...) noexcept> : declaration<R (*)(Args...)> {};
template <typename Fn> std::string callable(const char* name, const char* symbol) {
  return text("name", name) + ',' + text("symbol", symbol) + ',' + field("declaration", quote(declaration<Fn>::print(symbol)));
}
inline int add_params(params_state* state, const char* type, std::string fields) {
  if (state->declared) throw std::runtime_error("PL_PARAMS / PL_NO_PARAMS / PL_PARAMS_OPAQUE: declare exactly one per translation unit");
  state->declared = true;
  return add("params", type, std::move(fields));
}
inline int no_params(params_state* state) {
  return add_params(state, "void", "\"type\":\"void\",\"bytes\":0,\"align\":1,\"members\":[]");
}
template <typename T> int opaque_params(params_state* state, const char* type) {
  return add_params(state, type, text("type", type) + ',' + number("bytes", sizeof(T))
      + ',' + number("align", alignof(T)) + ",\"members\":[],\"opaque\":true");
}
struct member { const char* name; std::size_t offset, bytes, align; };
template <typename T> int params(params_state* state, const char* type, std::initializer_list<member> members) {
  static_assert(std::is_standard_layout_v<T>, "PL_PARAMS requires a standard-layout type");
  static_assert(std::is_trivially_copyable_v<T>, "PL_PARAMS requires a trivially copyable type");
  auto out = text("type", type) + ',' + number("bytes", sizeof(T)) + ',' + number("align", alignof(T)) + ",\"members\":[";
  bool first = true;
  for (const auto& m : members) {
    if (!first) out += ','; first = false;
    out += "{" + text("name", m.name) + ',' + number("offset", m.offset) + ',' + number("bytes", m.bytes) + ',' + number("align", m.align) + '}';
  }
  return add_params(state, type, out + ']');
}
// Keep builder declarations separate from compiler-reflected record fields.
inline std::string host_semantics(const std::string& declaration) {
  return field("declaration", quote(declaration));
}
inline int host_declaration(const char* kind, const char* name, const std::string& declaration) {
  return add(kind, name, text("name", name) + ',' + host_semantics(declaration));
}
inline void neutral_host_abi_types();
template <typename T>
int host_group(const char* owner, const char* group, const char* suffix, const char* type,
               std::uint64_t type_id, bool allocated) {
  neutral_host_abi_types();
  return add("host_group", owner, text("owner", owner) + ',' + text("params_group", group)
      + ',' + text("suffix", suffix) + ',' + text("type", type) + ',' + number("type_id", type_id)
      + ',' + number("bytes", allocated ? sizeof(T) : 0) + ',' + number("align", alignof(T)));
}
template <typename T>
int host_abi_type(std::uint64_t id, const char* name, std::initializer_list<member> members,
                  bool neutral = false) {
  if (!neutral && id >= PLK_ABI_TYPE_ID_MIN && id <= PLK_ABI_TYPE_ID_MAX)
    throw std::runtime_error("owner ABI type ID is reserved for the neutral inventory");
  auto out = number("id", id) + ',' + text("name", name) + ',' + number("bytes", sizeof(T))
      + ',' + number("align", alignof(T)) + ",\"members\":[";
  bool first = true;
  for (const auto& m : members) {
    if (!first) out += ','; first = false;
    out += "{" + text("name", m.name) + ',' + number("offset", m.offset) + ',' + number("bytes", m.bytes) + '}';
  }
  return add("host_abi_type", name, out + ']');
}
template <typename T>
int host_residue(std::uint64_t id, const char* name, const char* type, std::uint64_t type_id,
                 bool allocated, strings reads) {
  return add("host_residue", name, number("id", id) + ',' + text("name", name)
      + ',' + text("type", type) + ',' + number("type_id", type_id)
      + ',' + number("bytes", allocated ? sizeof(T) : 0) + ',' + number("align", alignof(T))
      + ',' + host_semantics(json(reads)));
}
template <typename T> struct host_storage_bits {
  static constexpr std::size_t value = sizeof(T) * 8;
};
template <typename T, unsigned Elements> struct host_storage_bits<pl::host_storage<T, Elements>> {
  static_assert(Elements > 0 && sizeof(T) * 8 % Elements == 0, "PL_HOST_CHANNEL storage must contain whole element bits");
  static constexpr std::size_t value = sizeof(T) * 8 / Elements;
};
template <typename Fn> int role(const char* name, const char* symbol, int warps) {
  registry().roles.emplace_back(name, warps);
  return add("role", name, callable<Fn>(name, symbol) + ',' + number("warps", warps));
}
template <typename Fn> int call(const char* name, const char* symbol, const char* direction,
                               const char* role_or_export, const char* exported = nullptr) {
  if (std::strcmp(direction, "consume") == 0) {
    if (exported) throw std::runtime_error("PL_CALL consume takes only an export endpoint, not a caller role");
    return add("call", name, callable<Fn>(name, symbol) + ',' + text("direction", direction)
        + ',' + field("role", "null") + ',' + text("export", role_or_export));
  }
  return add("call", name, callable<Fn>(name, symbol) + ',' + text("direction", direction)
      + ',' + text("role", role_or_export) + (exported ? ',' + text("export", exported) : ""));
}
inline std::string lifetime_json(const char* value) { return quote(value); }
inline std::string lifetime_json(during value) {
  auto out = "{" + text("tensor", value.tensor) + ',' + text("from", value.from);
  if (value.to) out += ',' + text("to", value.to);
  return out + '}';
}
template <typename Lifetime, typename... Roles>
int claim(const char* handle, const char* domain, int extent, int align, Lifetime lifetime, Roles... roles) {
  return add("claim", handle, text("handle", handle) + ',' + text("domain", domain) + ',' + number("extent", extent)
      + ',' + number("align", align) + ',' + field("lifetime", lifetime_json(lifetime))
      + ',' + field("roles", json(strings{roles...})));
}
inline int claim(const pl::claim& x) {
  if (x.is_table && !x.table_values.size())
    throw std::runtime_error(std::string("PL_CLAIM_EX ") + x.handle + ": extent table must not be empty");
  auto extent = x.is_table ? "{" + field("table", json(x.table_values)) + '}'
      : "{" + number("const", x.constant) + ',' + field("per", json(x.coefficients)) + '}';
  auto out = text("handle", x.handle) + ',' + text("domain", x.domain) + ',' + field("extent", extent)
      + ',' + number("align", x.alignment) + ',' + field("lifetime",
          x.tensor_lifetime ? lifetime_json(*x.tensor_lifetime) : lifetime_json(x.lifetime_value));
  if (x.role_names.size()) out += ',' + field("roles", json(x.role_names));
  if (x.absolute_base) out += ',' + number("absolute_base", *x.absolute_base);
  if (x.offset_value) out += ',' + number("offset", *x.offset_value);
  if (x.alias_value) out += ',' + text("alias", x.alias_value);
  return add("claim", x.handle, std::move(out));
}
template <typename... Roles>
int claim_regs(const char* handle, const char* reg_group, std::initializer_list<int> values, Roles... roles) {
  if (!values.size()) throw std::runtime_error(std::string("PL_CLAIM_REGS ") + handle + ": requires non-empty values");
  return add("claim_regs", handle, text("handle", handle) + ',' + text("reg_group", reg_group)
      + ',' + field("values", integers(values)) + ',' + field("roles", json(strings{roles...})));
}
inline int variant(int index, fixed_bindings key) {
  return add("variant", std::to_string(index), number("index", index) + ',' + field("key", json(key)));
}
template <typename Fn> int variant_entry(int index, fixed_bindings key, const char* symbol) {
  return add("variant", std::to_string(index), number("index", index) + ',' + field("key", json(key))
      + ',' + text("entry", symbol) + ',' + field("declaration", quote(declaration<Fn>::print(symbol))));
}
template <typename T> int descriptor() {
  return add("descriptor", "", number("bytes", sizeof(T)) + ',' + number("align", alignof(T)));
}
template <typename T, typename Lifetime, typename... Roles>
int claim_smem(const char* handle, const char* type, Lifetime lifetime, Roles... roles) {
  return add("claim_smem", handle, text("handle", handle) + ',' + text("type", type) + ',' + number("bytes", sizeof(T))
      + ',' + number("align", alignof(T)) + ',' + field("lifetime", lifetime_json(lifetime))
      + ',' + field("roles", json(strings{roles...})));
}
inline int topology(const pl::topology& x) {
  return add("topology", "", number("cta_group", x.cta_group) + ",\"cluster\":{\"compiled\":"
      + (x.cluster.is_dynamic ? quote("dynamic") : integers(x.cluster.compiled))
      + ",\"launch\":{" + number("x_multiple_of", x.cluster.launch.x_multiple_of)
      + ',' + field("max", integers(x.cluster.launch.max)) + "}},\"persistent\":" + (x.persistent ? "true" : "false")
      + ',' + text("scheduler", x.scheduler) + ',' + field("multicast", json(x.multicast)));
}
inline int scheduler_strategy(const char* id, const char* inputs_json) {
  return add("scheduler_strategy", "", text("id", id) + ',' + field("inputs", inputs_json));
}
template <typename Domain> int binding(const void* unit, const char* name, const char* sort, Domain input,
    const char* owner, const char* units, const char* scope, const char* lifetime) {
  auto d = as_domain(input); d.name = name; d.sort = sort; d.binding = true;
  registry().bindings.emplace_back(unit, d);
  return add("binding", name, text("name", name) + ',' + text("sort", sort) + ',' + field("domain", json(d))
      + ',' + text("owner", owner) + ',' + text("units", units) + ',' + text("observation_scope", scope) + ',' + text("lifetime", lifetime));
}

template <typename Input> auto map_argument(int value) {
  if constexpr (std::is_same_v<Input, binding_input>) return bound{value};
  else return value;
}
template <typename Fn, typename... Inputs, std::size_t... I>
auto evaluate(Fn& fn, const std::array<int, sizeof...(Inputs)>& row, std::index_sequence<I...>) {
  return fn(map_argument<Inputs>(row[I])...);
}
// Resolution happens at emission, after every translation unit has registered.
template <typename Fn, typename... Inputs>
void map(const void* unit, const std::string& name, std::vector<std::string> outputs, Fn fn, const std::tuple<Inputs...>& inputs) {
  using result_type = std::invoke_result_t<Fn&, decltype(map_argument<Inputs>(0))...>;
  static_assert(std::is_same_v<result_type, std::array<int, std::tuple_size_v<result_type>>>,
                "PL map must return std::array<int, N>");
  auto domains = std::apply([](const auto&... input) { return std::vector<domain>{as_domain(input)...}; }, inputs);
  registry().records.push_back({"map", name, {},
    [unit, name, outputs = std::move(outputs), fn, domains = std::move(domains)](std::ostream& out) mutable {
      if (std::tuple_size_v<result_type> != outputs.size())
        throw std::runtime_error("P-MAP: output arity mismatch for " + name);
      for (auto& d : domains) if (d.binding && !d.known) {
        for (const auto& candidate : registry().bindings) if (candidate.first == unit && candidate.second.name == d.name) {
          d = candidate.second; break;
        }
        if (!d.known) throw std::runtime_error("P-MAP: " + name + ": undeclared binding '" + d.name + "'; add PL_BINDING in the same translation unit");
      }
      out << text("name", name.c_str()) << ",\"inputs\":[";
      bool first = true;
      for (const auto& d : domains) { if (!first) out << ','; first = false; out << json(d, true); }
      out << "],\"outputs\":" << json(outputs) << ",\"rows\":[";
      std::array<int, sizeof...(Inputs)> row{};
      bool first_row = true;
      auto enumerate = [&](auto&& self, std::size_t depth) -> void {
        if (depth == domains.size()) {
          auto result = evaluate<Fn, Inputs...>(fn, row, std::index_sequence_for<Inputs...>{});
          if (!first_row) out << ','; first_row = false; out << '[';
          bool first_value = true;
          for (std::size_t i = 0; i < row.size(); ++i) {
            if (!first_value) out << ','; first_value = false; out << row[i];
          }
          for (auto value : result) { if (!first_value) out << ','; first_value = false; out << value; }
          out << ']'; return;
        }
        const auto& d = domains[depth];
        if (d.finite) for (int value : d.values) { row[depth] = value; self(self, depth + 1); }
        else for (int value = d.lo; value < d.hi; ++value) { row[depth] = value; self(self, depth + 1); }
      };
      try { enumerate(enumerate, 0); }
      catch (const std::exception& e) { throw std::runtime_error("P-MAP: " + name + ": " + e.what()); }
      out << ']';
    }});
}
template <typename Tuple, std::size_t... I>
auto take(const Tuple& tuple, std::index_sequence<I...>) { return std::make_tuple(std::get<I>(tuple)...); }
template <typename Tuple> std::string domains_json(const Tuple& inputs) {
  std::string out = "[";
  auto append = [&](const auto& input) {
    if constexpr (!std::is_same_v<std::decay_t<decltype(input)>, binding_input>)
      out += (out.size() > 1 ? "," : "") + json(as_domain(input));
  };
  std::apply([&](const auto&... input) {
    (append(input), ...);
  }, inputs);
  return out + ']';
}
template <typename... Parts>
std::string demand(const void* unit, const std::string& name, const identity& id, const demand_positions<Parts...>& x) {
  constexpr auto count = sizeof...(Parts);
  static_assert(count >= 1, "demand_positions requires a map");
  auto inputs = take(x.parts, std::make_index_sequence<count - 1>{});
  std::vector<std::string> outputs;
  for (auto c : id.coords) outputs.emplace_back(c.name);
  outputs.emplace_back("participant"); outputs.emplace_back("position");
  map(unit, name, outputs, std::get<count - 1>(x.parts), inputs);
  return "{\"kind\":\"positions\"," + field("domain", domains_json(inputs)) + ',' + field("map", quote(name))
      + ',' + field("outputs", json(outputs)) + '}';
}
template <typename... Parts>
std::string demand(const void* unit, const std::string& name, const identity& id, const demand_view<Parts...>& x) {
  constexpr auto count = sizeof...(Parts);
  static_assert(count >= 3, "demand_view requires space, units, and a map");
  auto inputs = take(x.parts, std::make_index_sequence<count - 3>{});
  std::vector<std::string> outputs;
  for (auto c : id.coords) outputs.emplace_back(c.name);
  outputs.emplace_back("address");
  map(unit, name, outputs, std::get<count - 1>(x.parts), inputs);
  return "{\"kind\":\"memory_view\"," + field("domain", domains_json(inputs)) + ',' + field("map", quote(name))
      + ',' + field("outputs", json(outputs)) + ',' + text("space", std::get<count - 3>(x.parts))
      + ',' + text("units", std::get<count - 2>(x.parts)) + '}';
}
template <typename Fn, typename... Inputs>
int export_tensor(const void* unit, const char* name, const identity& id, const placement<Fn, Inputs...>& x,
                   strings capabilities, readiness ready, fixed_bindings fixed, lends backing, calls endpoints) {
  std::string map_name = std::string(name) + ".placement";
  map(unit, map_name, std::vector<std::string>(x.outputs.begin(), x.outputs.end()), x.map, x.inputs);
  std::string fixed_json = "{";
  for (auto b : fixed) { if (fixed_json.size() > 1) fixed_json += ','; fixed_json += field(b.name, std::to_string(b.value)); }
  fixed_json += '}';
  return add("tensor_export", name, text("name", name) + ',' + field("identity", json(id))
      + ',' + field("read_capabilities", json(capabilities)) + ",\"readiness\":{" + text("after", ready.after)
      + ',' + text("reuse_after", ready.reuse_after) + "}," + field("fixed_bindings", fixed_json)
      + ',' + field("backing", json(backing)) + ",\"placement\":{" + text("space", x.space) + ',' + text("units", x.units)
      + ',' + field("components", json(x.outputs)) + ',' + field("map", quote(map_name)) + "}"
      + ",\"calls\":{" + text("produce", endpoints.produce) + ',' + field("release", endpoints.release ? quote(endpoints.release) : "null") + "}");
}
template <typename Fn, typename Demand>
int import_tensor(const void* unit, const char* name, const char* symbol, const identity& id, args arguments,
                   const Demand& x, strings sites, const char* completion, lends resources,
                   std::initializer_list<rendezvous> events) {
  auto types = declaration<Fn>::parameter_types();
  if (arguments.size() != types.size()) throw std::runtime_error("P-MAP: tensor argument count differs from callable");
  std::string arguments_json = "[";
  std::size_t i = 0;
  for (auto a : arguments) {
    if (i) arguments_json += ',';
    arguments_json += "{" + text("name", a.name) + ',' + text("meaning", a.meaning) + ',' + field("ctype", quote(types[i++])) + '}';
  }
  std::string events_json = "[";
  for (const auto& event : events) {
    if (events_json.size() > 1) events_json += ',';
    events_json += "{" + text("event", event.event) + ',' + text("barrier", event.barrier)
        + ',' + field("cohort", event.all ? "{\"all_threads\":true}" : "{\"set\":" + integers(event.cohort) + '}') + '}';
  }
  return add("tensor_import", name, callable<Fn>(name, symbol) + ',' + field("identity", json(id))
      + ',' + field("args", arguments_json + ']') + ',' + field("demand", demand(unit, std::string(name) + ".demand", id, x))
      + ',' + field("sites", json(sites)) + ',' + text("completion", completion) + ',' + field("lends", json(resources))
      + ',' + field("rendezvous", events_json + ']'));
}
template <typename Fn>
void write_participants(std::ostream& out, const std::string& name, domain d, Fn fn) {
  constexpr bool predicate = std::is_same_v<decltype(fn(0)), bool>;
  if constexpr (predicate) {
    std::vector<int> selected;
    if (d.finite) { for (int i : d.values) if (fn(i)) selected.push_back(i); }
    else { for (int i = d.lo; i < d.hi; ++i) if (fn(i)) selected.push_back(i); }
    if (selected.empty()) throw std::runtime_error("P-MAP: " + name + ": participant predicate selects no threads");
    d.finite = true; d.values = std::move(selected);
  }
  out << field("name", quote(name)) << ",\"inputs\":[" << json(d, true)
      << "],\"outputs\":[\"participant\"],\"rows\":[";
  bool first = true;
  auto row = [&](int i) {
    if (!first) out << ','; first = false;
    out << '[' << i << ',';
    if constexpr (predicate) out << i; else out << fn(i);
    out << ']';
  };
  if (d.finite) for (int i : d.values) row(i);
  else for (int i = d.lo; i < d.hi; ++i) row(i);
  out << ']';
}
inline std::string site_participants(const char*, const char*, all_threads_t) { return "{\"all_threads\":true}"; }
template <typename Domain, typename Fn>
std::string site_participants(const char* name, const char*, const participants<Domain, Fn>& x) {
  std::string map_name = std::string(name) + ".participants";
  registry().records.push_back({"map", map_name, {}, [map_name, d = as_domain(x.domain), fn = x.map](std::ostream& out) {
    write_participants(out, map_name, d, fn);
  }});
  return "{" + field("map", quote(map_name)) + '}';
}
template <typename Fn> std::string site_participants(const char* name, const char* role, Fn fn) {
  // A bare lambda uses the role's complete thread domain, resolved after registration.
  std::string map_name = std::string(name) + ".participants";
  registry().records.push_back({"map", map_name, {}, [map_name, role = std::string(role), fn](std::ostream& out) {
    auto it = std::find_if(registry().roles.begin(), registry().roles.end(), [&](const auto& r) { return r.first == role; });
    if (it == registry().roles.end()) throw std::runtime_error("P-MAP: participant map references unknown role " + role);
    write_participants(out, map_name, as_domain(coord{"t", 0, it->second * 32}), fn);
  }});
  return "{" + field("map", quote(map_name)) + '}';
}
template <typename Participants>
int site(const char* name, const char* role, Participants participants, const char* invocation, guarantees guarantees, strings observes) {
  return add("site", name, text("name", name) + ',' + text("role", role) + ',' + field("participants", site_participants(name, role, participants))
      + ',' + text("invocation", invocation) + ',' + field("guarantees", json(guarantees)) + ',' + field("observes", json(observes)));
}
inline void emit(std::ostream& out) {
  if (registry().error) std::rethrow_exception(registry().error);
  auto& records = registry().records;
  std::sort(records.begin(), records.end(), [](const record& a, const record& b) {
    return std::tie(a.kind, a.name, a.fields) < std::tie(b.kind, b.name, b.fields);
  });
  out << "{\"records\":[";
  bool first = true;
  for (const auto& r : records) {
    if (!first) out << ','; first = false;
    out << '{' << field("kind", quote(r.kind)) << ',';
    if (r.write) r.write(out); else out << r.fields;
    out << '}';
  }
  out << "]}\n";
}
} // namespace detail

namespace host {
using roles = pl::strings;

// Host builders copy borrowed lists immediately; named builders own their values.
struct value {
  std::string encoded;
  template <typename Int, std::enable_if_t<std::is_integral_v<Int> && !std::is_same_v<Int, bool>, int> = 0>
  value(Int x) : encoded(std::to_string(x)) {}
  value(bool x) : encoded(x ? "true" : "false") {}
  value(double x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    encoded = "{" + detail::number("bits", bits) + '}';
  }
  value(const char* x) : encoded(detail::quote(x)) {}
  value(std::initializer_list<value> items) : encoded("[") {
    for (const auto& item : items) {
      if (encoded.size() > 1) encoded += ',';
      encoded += item.json();
    }
    encoded += ']';
  }
  std::string json() const { return encoded; }
};
struct source {
  std::string encoded;
  std::string json() const { return encoded; }
};
struct expr {
  std::string encoded;
  std::string json() const { return encoded; }
};
struct reference {
  std::string kind, name;
  operator expr() const { return {"{" + detail::field(kind.c_str(), detail::quote(name)) + '}'}; }
  operator source() const {
    return {"{" + detail::field("kind", detail::quote(kind)) + ',' + detail::field("name", detail::quote(name)) + '}'};
  }
  std::string json() const { return static_cast<expr>(*this).json(); }
};
struct integer_constant {
  std::string encoded;
  operator expr() const { return {encoded}; }
  operator source() const { return {"{\"kind\":\"constant\",\"value\":" + encoded + '}'}; }
  std::string json() const { return encoded; }
};
template <typename Int, std::enable_if_t<std::is_integral_v<Int> && !std::is_same_v<Int, bool>, int> = 0>
integer_constant constant(Int x) { return {std::to_string(x)}; }
inline source constant(value x) { return {"{\"kind\":\"constant\",\"value\":" + x.json() + '}'}; }
inline expr scalar_ref(const char* name) { return {"{" + detail::text("scalar", name) + '}'}; }
inline reference fact_ref(const char* name) { return {"fact", name}; }
inline reference binding_ref(const char* name) { return {"binding", name}; }
inline source caller_ref(const char* name) { return {"{\"kind\":\"caller\"," + detail::text("name", name) + '}'}; }
inline source hardware_ref(const char* name) { return {"{\"kind\":\"hardware\"," + detail::text("name", name) + '}'}; }
inline expr operator+(expr a, expr b) { return {"{\"op\":\"add\",\"args\":[" + a.json() + ',' + b.json() + "]}"}; }
inline expr operator*(expr a, expr b) { return {"{\"op\":\"mul\",\"args\":[" + a.json() + ',' + b.json() + "]}"}; }
inline source tuple(std::vector<source> items) {
  std::string out = "{\"kind\":\"tuple\",\"items\":[";
  bool first = true;
  for (const auto& item : items) {
    if (!first) out += ','; first = false;
    out += item.json();
  }
  return {out + "]}"};
}

struct scalar {
  unsigned id;
  std::string name, type;
  source source_value{"null"};
  scalar(unsigned id, const char* name, const char* type) : id(id), name(name), type(type) {}
  scalar& caller(const char* input) { source_value = caller_ref(input); return *this; }
  scalar& constant(std::int64_t x) { source_value = host::constant(x); return *this; }
  scalar& constant(double x) { source_value = host::constant(value(x)); return *this; }
  scalar& constant(bool x) { source_value = host::constant(value(x)); return *this; }
  template <typename Int, std::enable_if_t<std::is_integral_v<Int> && !std::is_same_v<Int, bool>, int> = 0>
  scalar& constant(Int x) { source_value = host::constant(x); return *this; }
  scalar& fact(const char* name) { source_value = fact_ref(name); return *this; }
  scalar& binding(const char* name) { source_value = binding_ref(name); return *this; }
  scalar& hardware(const char* name) { source_value = hardware_ref(name); return *this; }
  std::string json() const {
    return "{" + detail::number("id", id) + ',' + detail::field("name", detail::quote(name))
        + ',' + detail::field("type", detail::quote(type)) + ',' + detail::field("source", source_value.json()) + '}';
  }
};
struct scalars {
  std::string read_roles;
  std::vector<scalar> values;
  explicit scalars(roles reads) : read_roles(detail::json(reads)) {}
  scalars& field(scalar x) { values.push_back(std::move(x)); return *this; }
  scalars& fields(std::vector<scalar> x) { values = std::move(x); return *this; }
  std::string json() const {
    std::string out = "{\"read_roles\":" + read_roles + ",\"fields\":[";
    bool first = true;
    for (const auto& x : values) { if (!first) out += ','; first = false; out += x.json(); }
    return out + "]}";
  }
};
struct channel {
  std::string name, storage_dtype, logical_dtype;
  unsigned rank_value = 1;
  const char* access = "read";
  const char* domain_value = nullptr;
  bool multicast_view = false, peer_view = false;
  source source_value{"null"};
  std::string alias_value = "null", initializer_value = "{\"kind\":\"caller\"}", lifetime_value = "release";
  channel(const char* name, const char* storage) : name(name), storage_dtype(storage), logical_dtype(storage) {}
  channel& logical(const char* x) { logical_dtype = x; return *this; }
  channel& rank(unsigned x) { rank_value = x; return *this; }
  channel& read() { access = "read"; return *this; }
  channel& write() { access = "write"; return *this; }
  channel& read_write() { access = "read_write"; return *this; }
  // One caller-lent allocation symmetric over a declared PL_HOST_DOMAIN. domain() alone is
  // local access to it; peer()/multicast() add the views build receives.
  channel& domain(const char* x) {
    if (domain_value && std::strcmp(domain_value, x) != 0)
      throw std::runtime_error("pl::host::channel " + name + ": views name different domains");
    domain_value = x; return *this;
  }
  channel& peer(const char* x) { peer_view = true; return domain(x); }
  channel& multicast(const char* x) { multicast_view = true; return domain(x); }
  channel& caller(const char* input) { source_value = caller_ref(input); return *this; }
  channel& workspace(const char* name) {
    source_value = {"{\"kind\":\"workspace\"," + detail::text("name", name) + '}'}; return *this;
  }
  channel& alias(const char* name, expr bytes, expr bits = constant(0)) {
    alias_value = "{" + detail::text("channel", name) + ',' + detail::field("byte_offset", bytes.json())
        + ',' + detail::field("bit_offset", bits.json()) + '}'; return *this;
  }
  channel& initializer(const char* kind, const char* owner = nullptr) {
    initializer_value = "{" + detail::text("kind", kind);
    if (owner) initializer_value += ',' + detail::text("owner", owner);
    initializer_value += '}'; return *this;
  }
  channel& lifetime(const char* x) { lifetime_value = x; return *this; }
  std::string json() const {
    return "{" + detail::field("name", detail::quote(name)) + ',' + detail::field("storage_dtype", detail::quote(storage_dtype))
        + ',' + detail::field("logical_dtype", detail::quote(logical_dtype)) + ',' + detail::number("rank", rank_value)
        + ',' + detail::text("access", access)
        + ',' + detail::field("domain", domain_value ? detail::quote(domain_value) : std::string("null"))
        + ",\"views\":[" + (multicast_view ? std::string("\"multicast\"") : std::string())
        + (multicast_view && peer_view ? "," : "") + (peer_view ? "\"peer\"" : "") + ']'
        + ',' + detail::field("source", source_value.json())
        + ',' + detail::field("alias", alias_value) + ',' + detail::field("initializer", initializer_value)
        + ',' + detail::field("lifetime", detail::quote(lifetime_value)) + '}';
  }
};
struct workspace {
  std::string name, initializer = "{\"kind\":\"none\"}", group;
  const char* reset = "open";
  explicit workspace(const char* name) : name(name) {}
  workspace& owner_initialized(const char* owner) { initializer = "{\"kind\":\"owner\"," + detail::text("owner", owner) + '}'; return *this; }
  workspace& zero() { initializer = "{\"kind\":\"zero\"}"; return *this; }
  workspace& reset_open() { reset = "open"; return *this; }
  workspace& reset_launch() { reset = "launch"; return *this; }
  workspace& params_group(const char* x) { group = x; return *this; }
  std::string json() const {
    auto out = "{" + detail::field("name", detail::quote(name))
        + ",\"bytes\":\"query\",\"space\":\"gmem\",\"channel\":null,"
        + detail::field("initializer", initializer) + ',' + detail::text("reset", reset);
    if (!group.empty()) out += ',' + detail::field("params_group", detail::quote(group));
    return out + '}';
  }
};
struct transfer {
  std::string name, channel, endpoint = "null";
  const char* access = "read";
  transfer(const char* name, const char* channel) : name(name), channel(channel) {}
  transfer& device(const char* stage, const char* name) {
    endpoint = "{" + detail::text("stage", stage) + ',' + detail::text("endpoint", name) + '}'; return *this;
  }
  transfer& read() { access = "read"; return *this; }
  transfer& write() { access = "write"; return *this; }
  std::string json() const {
    return "{" + detail::field("name", detail::quote(name)) + ',' + detail::field("channel", detail::quote(channel))
        + ',' + detail::field("device_endpoint", endpoint) + ',' + detail::text("access", access) + '}';
  }
};
struct selection {
  std::string name, domain;
  selection(const char* name, std::initializer_list<int> values) : name(name), domain(detail::integers(values)) {}
};
struct fact_variant {
  std::string selection_json;
  host::value value;
  fact_variant(std::initializer_list<selection> selection, host::value value) : selection_json("{"), value(std::move(value)) {
    for (const auto& item : selection) {
      if (selection_json.size() > 1) selection_json += ',';
      selection_json += detail::field(item.name.c_str(), item.domain);
    }
    selection_json += '}';
  }
  std::string json() const { return "{\"selection\":" + selection_json + ",\"value\":" + value.json() + '}'; }
};
struct fact {
  std::string name, owner_value, stage_value, consumers_value = "[]", type = "null", value_json = "null";
  std::optional<std::vector<fact_variant>> variant_values;
  explicit fact(const char* name) : name(name) {}
  fact& owner(const char* x) { owner_value = x; return *this; }
  fact& stage(const char* x) { stage_value = x; return *this; }
  fact& consumers(roles x) { consumers_value = detail::json(x); return *this; }
  fact& i32(int x) { type = "\"i32\""; value_json = value(x).json(); variant_values.reset(); return *this; }
  fact& i64(std::int64_t x) { type = "\"i64\""; value_json = value(x).json(); variant_values.reset(); return *this; }
  fact& u64(std::uint64_t x) { type = "\"u64\""; value_json = value(x).json(); variant_values.reset(); return *this; }
  fact& boolean(bool x) { type = "\"bool\""; value_json = value(x).json(); variant_values.reset(); return *this; }
  fact& dtype(const char* x) { type = "\"dtype\""; value_json = value(x).json(); variant_values.reset(); return *this; }
  fact& tuple_i32(std::initializer_list<int> items) {
    type = "{\"kind\":\"tuple\",\"items\":[";
    bool first = true;
    for (auto item : items) {
      if (!first) type += ','; first = false;
      type += "{\"kind\":\"static\",\"type\":\"i32\"," + detail::number("value", item) + '}';
    }
    type += "]}"; value_json = detail::integers(items); variant_values.reset(); return *this;
  }
  fact& variants(std::vector<fact_variant> x) { variant_values = std::move(x); return *this; }
  std::string json() const {
    auto out = "{" + detail::field("name", detail::quote(name)) + ',' + detail::field("type", type)
        + ',' + detail::field("owner", detail::quote(owner_value)) + ',' + detail::field("stage", detail::quote(stage_value))
        + ',' + detail::field("consumers", consumers_value);
    if (!variant_values) return out + ",\"value\":" + value_json + '}';
    out += ",\"variants\":[";
    bool first = true;
    for (const auto& item : *variant_values) { if (!first) out += ','; first = false; out += item.json(); }
    return out + "]}";
  }
};
struct leaf {
  std::string encoded;
  static leaf static_(const char* type) { return {"{\"kind\":\"static\"," + detail::text("type", type) + '}'}; }
  static leaf runtime(const char* type) { return {"{\"kind\":\"runtime\"," + detail::text("type", type) + '}'}; }
  static leaf enumeration(roles values) { return {"{\"kind\":\"enum\",\"values\":" + detail::json(values) + '}'}; }
  static leaf tuple(std::vector<leaf> items) {
    std::string out = "{\"kind\":\"tuple\",\"items\":[";
    bool first = true;
    for (const auto& item : items) { if (!first) out += ','; first = false; out += item.json(); }
    return {out + "]}"};
  }
  static leaf dtype() { return {"{\"kind\":\"dtype\"}"}; }
  std::string json() const { return encoded; }
};
struct schema {
  std::string required_fields, optional_fields, pinned_fields, hardware_fields, runtime_fields = "[]";
  bool no_launch = false;
  schema& required(const char* name, leaf x) {
    if (!required_fields.empty()) required_fields += ',';
    required_fields += detail::field(name, x.json()); return *this;
  }
  schema& optional(const char* name, leaf x) {
    if (!optional_fields.empty()) optional_fields += ',';
    optional_fields += detail::field(name, x.json()); return *this;
  }
  schema& pinned(const char* name, value x) {
    if (!pinned_fields.empty()) pinned_fields += ',';
    pinned_fields += detail::field(name, x.json()); return *this;
  }
  schema& runtime_i32(roles x) { runtime_fields = detail::json(x); return *this; }
  schema& hardware_rule(const char* name, const char* kind, const char* source) {
    if (!hardware_fields.empty()) hardware_fields += ',';
    hardware_fields += detail::field(name, "{" + detail::text("kind", kind) + ',' + detail::text("name", source) + '}'); return *this;
  }
  schema& allow_no_launch(bool x) { no_launch = x; return *this; }
  std::string json() const {
    return "{\"required\":{" + required_fields + "},\"optional\":{" + optional_fields + "},\"pinned\":{" + pinned_fields
        + "},\"runtime_i32\":" + runtime_fields + ",\"hardware_rules\":{" + hardware_fields
        + "},\"allow_no_launch\":" + (no_launch ? "true" : "false") + '}';
  }
};
struct scheduler {
  std::string owner, strategy, schema_body, inputs, runtime = "[]", workspace_ids = "[]";
  explicit scheduler(const char* owner) : owner(owner) {}
  scheduler(const char* owner, const char* strategy, schema body) : owner(owner), strategy(strategy), schema_body(body.json()) {}
  scheduler& input(const char* name, source x) {
    if (!inputs.empty()) inputs += ',';
    inputs += "{" + detail::text("name", name) + ',' + detail::field("source", x.json()) + '}'; return *this;
  }
  scheduler& runtime_inputs(roles x) { runtime = detail::json(x); return *this; }
  scheduler& workspaces(std::initializer_list<unsigned> x) { workspace_ids = detail::integers(x); return *this; }
  std::string json() const {
    auto out = "{" + detail::field("owner", detail::quote(owner));
    if (schema_body.empty()) return out + '}';
    return out + ',' + detail::field("strategy", detail::quote(strategy)) + ",\"schema\":" + schema_body
        + ",\"inputs\":[" + inputs + "],\"runtime_inputs\":" + runtime + ",\"workspace_ids\":" + workspace_ids + '}';
  }
};
struct variants {
  std::string encoded = "[";
  variants(std::vector<pl::fixed_bindings> values) {
    unsigned index = 0;
    for (const auto& bindings : values) {
      if (index) encoded += ',';
      encoded += "{" + detail::number("index", index++) + ",\"entry\":null,\"key\":{";
      bool first = true;
      for (const auto& binding : bindings) {
        if (!first) encoded += ','; first = false;
        encoded += detail::number(binding.name, binding.value);
      }
      encoded += "}}";
    }
    encoded += ']';
  }
  std::string json() const { return encoded; }
};
} // namespace host

namespace detail {
template <typename Builder>
int host_builder(const char* kind, const std::string& name, unsigned id, const Builder& builder) {
  return add(kind, name, number("id", id) + ',' + field("name", quote(name)) + ',' + host_semantics(builder.json()));
}
template <typename StorageT>
int host_channel(unsigned id, const host::channel& builder) {
  return add("host_channel", builder.name, number("id", id) + ',' + field("name", quote(builder.name)) + ',' + number("storage_bits", host_storage_bits<StorageT>::value)
      + ',' + host_semantics(builder.json()));
}
inline int host_workspace(unsigned id, std::size_t align, const host::workspace& builder) {
  return add("host_workspace", builder.name, number("id", id) + ',' + field("name", quote(builder.name)) + ',' + number("align", align) + ',' + host_semantics(builder.json()));
}
inline int host_transfer(unsigned id, const host::transfer& builder) { return host_builder("host_transfer", builder.name, id, builder); }
inline int host_domain(unsigned id, const char* name) { return add("host_domain", name, number("id", id) + ',' + text("name", name)); }
inline int host_fact(const host::fact& builder) { return host_declaration("host_fact", builder.name.c_str(), builder.json()); }
} // namespace detail
} // namespace pl

#define PL_DETAIL_JOIN_I(a, b) a##b
#define PL_DETAIL_JOIN(a, b) PL_DETAIL_JOIN_I(a, b)
#define PL_DETAIL_REGISTER(...) static const int PL_DETAIL_JOIN(pl_registration_, __COUNTER__) = (__VA_ARGS__)
#define PL_DETAIL_CHECKED_REGISTER(...) PL_DETAIL_REGISTER(::pl::detail::checked_registration([] { return (__VA_ARGS__); }))
#define PL_DETAIL_FIELD(T, m) ::pl::detail::member{#m, offsetof(T, m), sizeof(((T*)nullptr)->m), __alignof__(((T*)nullptr)->m)}
#define PL_DETAIL_FIELDS_1(T, a) PL_DETAIL_FIELD(T, a)
#define PL_DETAIL_FIELDS_2(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_1(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_3(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_2(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_4(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_3(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_5(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_4(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_6(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_5(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_7(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_6(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_8(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_7(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_9(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_8(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_10(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_9(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_11(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_10(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_12(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_11(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_13(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_12(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_14(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_13(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_15(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_14(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_16(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_15(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_17(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_16(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_18(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_17(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_19(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_18(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_20(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_19(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_21(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_20(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_22(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_21(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_23(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_22(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_24(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_23(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_25(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_24(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_26(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_25(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_27(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_26(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_28(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_27(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_29(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_28(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_30(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_29(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_31(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_30(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_32(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_31(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_33(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_32(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_34(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_33(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_35(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_34(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_36(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_35(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_37(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_36(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_38(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_37(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_39(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_38(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_40(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_39(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_41(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_40(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_42(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_41(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_43(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_42(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_44(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_43(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_45(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_44(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_46(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_45(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_47(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_46(T, __VA_ARGS__)
#define PL_DETAIL_FIELDS_48(T, a, ...) PL_DETAIL_FIELD(T, a), PL_DETAIL_FIELDS_47(T, __VA_ARGS__)
#define PL_DETAIL_PARAMS_NARGS_I(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao,ap,aq,ar,av,at,au,aw,N,...) N
#define PL_DETAIL_PARAMS_NARGS(...) PL_DETAIL_PARAMS_NARGS_I(__VA_ARGS__,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
#define PL_DETAIL_NARGS_I(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,N,...) N
#define PL_DETAIL_NARGS(...) PL_DETAIL_NARGS_I(__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
#define PL_STAGE(id, version) PL_DETAIL_REGISTER(::pl::detail::add("stage", id, ::pl::detail::text("id", id) + ',' + ::pl::detail::text("version", version)))
#define PL_PARAMS(T, ...) PL_DETAIL_CHECKED_REGISTER(::pl::detail::params<T>(&::pl::detail::translation_unit_params, #T, {PL_DETAIL_JOIN(PL_DETAIL_FIELDS_, PL_DETAIL_PARAMS_NARGS(__VA_ARGS__))(T, __VA_ARGS__)}))
#define PL_NO_PARAMS() PL_DETAIL_CHECKED_REGISTER(::pl::detail::no_params(&::pl::detail::translation_unit_params))
#define PL_PARAMS_OPAQUE(T) PL_DETAIL_CHECKED_REGISTER(::pl::detail::opaque_params<T>(&::pl::detail::translation_unit_params, #T))
#define PL_DETAIL_TYPE_LAYOUT(T, shadow, ...) \
  struct shadow { __VA_ARGS__ }; \
  static_assert(sizeof(shadow) == sizeof(T) && alignof(shadow) == alignof(T), "PL_TYPE members do not restate " #T "'s layout")
#define PL_TYPE(T, ...) \
  PL_DETAIL_TYPE_LAYOUT(T, PL_DETAIL_JOIN(pl_type_shadow_, __COUNTER__), __VA_ARGS__); \
  template <> struct pl::detail::registered_type<T> { \
    static constexpr const char* name = #T; \
    static constexpr const char* members = #__VA_ARGS__; \
  }; \
  PL_DETAIL_REGISTER(::pl::detail::register_type<T>(#T))
#define PL_ROLE(name, fn, warps) PL_DETAIL_REGISTER(::pl::detail::role<decltype(&fn)>(#name, #fn, warps))
#define PL_EXPORT(name, fn) PL_DETAIL_REGISTER(::pl::detail::add("export", #name, ::pl::detail::callable<decltype(&fn)>(#name, #fn)))
#define PL_CALL(name, fn, ...) PL_DETAIL_REGISTER(::pl::detail::call<decltype(&fn)>(#name, #fn, __VA_ARGS__))
#define PL_ENDPOINT(name, ...) PL_DETAIL_CHECKED_REGISTER(::pl::detail::endpoint(#name, __VA_ARGS__))
// Keep lifetime in the variadic tail so pl::during{tensor, from, to} needs no parentheses.
#define PL_CLAIM(handle, domain, extent, align, ...) PL_DETAIL_REGISTER(::pl::detail::claim(handle, domain, extent, align, __VA_ARGS__))
#define PL_CLAIM_SMEM(handle, T, ...) PL_DETAIL_REGISTER(::pl::detail::claim_smem<T>(handle, #T, __VA_ARGS__))
#define PL_CLAIM_EX(...) PL_DETAIL_CHECKED_REGISTER(::pl::detail::claim(__VA_ARGS__))
#define PL_CLAIM_REGS(handle, reg_group, ...) PL_DETAIL_CHECKED_REGISTER(::pl::detail::claim_regs(handle, reg_group, __VA_ARGS__))
#define PL_VARIANT(index, ...) PL_DETAIL_CHECKED_REGISTER(::pl::detail::variant(index, __VA_ARGS__))
// Braces do not group macro arguments. Move the final function token ahead of
// the key fragments before stringizing it; named fixed_bindings also work.
#define PL_DETAIL_VARIANT_ENTRY(fn, index, ...) PL_DETAIL_CHECKED_REGISTER(::pl::detail::variant_entry<decltype(&fn)>(index, __VA_ARGS__, #fn))
#define PL_DETAIL_VARIANT_ENTRY_2(index, a1, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1)
#define PL_DETAIL_VARIANT_ENTRY_3(index, a1, a2, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2)
#define PL_DETAIL_VARIANT_ENTRY_4(index, a1, a2, a3, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3)
#define PL_DETAIL_VARIANT_ENTRY_5(index, a1, a2, a3, a4, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4)
#define PL_DETAIL_VARIANT_ENTRY_6(index, a1, a2, a3, a4, a5, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5)
#define PL_DETAIL_VARIANT_ENTRY_7(index, a1, a2, a3, a4, a5, a6, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6)
#define PL_DETAIL_VARIANT_ENTRY_8(index, a1, a2, a3, a4, a5, a6, a7, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7)
#define PL_DETAIL_VARIANT_ENTRY_9(index, a1, a2, a3, a4, a5, a6, a7, a8, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8)
#define PL_DETAIL_VARIANT_ENTRY_10(index, a1, a2, a3, a4, a5, a6, a7, a8, a9, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8, a9)
#define PL_DETAIL_VARIANT_ENTRY_11(index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
#define PL_DETAIL_VARIANT_ENTRY_12(index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
#define PL_DETAIL_VARIANT_ENTRY_13(index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
#define PL_DETAIL_VARIANT_ENTRY_14(index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)
#define PL_DETAIL_VARIANT_ENTRY_15(index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14)
#define PL_DETAIL_VARIANT_ENTRY_16(index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, fn) PL_DETAIL_VARIANT_ENTRY(fn, index, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)
#define PL_VARIANT_ENTRY(index, ...) PL_DETAIL_JOIN(PL_DETAIL_VARIANT_ENTRY_, PL_DETAIL_NARGS(__VA_ARGS__))(index, __VA_ARGS__)
#define PL_DESCRIPTOR(T) PL_DETAIL_CHECKED_REGISTER(::pl::detail::descriptor<T>())
#define PL_TOPOLOGY(...) PL_DETAIL_REGISTER(::pl::detail::topology(__VA_ARGS__))
#define PL_SCHEDULER_STRATEGY(id, inputs_json) PL_DETAIL_REGISTER(::pl::detail::scheduler_strategy(id, inputs_json))
#define PL_BINDING(...) PL_DETAIL_REGISTER(::pl::detail::binding(&::pl::detail::translation_unit, __VA_ARGS__))
#define PL_SITE(name, ...) PL_DETAIL_REGISTER(::pl::detail::site(#name, __VA_ARGS__))
#define PL_EXPORT_TENSOR(name, ...) PL_DETAIL_REGISTER(::pl::detail::export_tensor(&::pl::detail::translation_unit, #name, __VA_ARGS__))
#define PL_IMPORT_TENSOR(name, fn, ...) PL_DETAIL_REGISTER(::pl::detail::import_tensor<decltype(&fn)>(&::pl::detail::translation_unit, #name, #fn, __VA_ARGS__))
#define PL_HINT(key, value) PL_DETAIL_REGISTER(::pl::detail::hint(key, value))
#define PL_HOST_GROUP(owner, group, suffix, T, type_id, allocated) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_group<T>(owner, group, suffix, #T, type_id, allocated))
#define PL_HOST_ABI_TYPE(id, T, ...) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_abi_type<T>(id, #T, {PL_DETAIL_JOIN(PL_DETAIL_FIELDS_, PL_DETAIL_PARAMS_NARGS(__VA_ARGS__))(T, __VA_ARGS__)}))
#define PL_HOST_ABI_EMPTY(id, T) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_abi_type<T>(id, #T, {}))
#define PL_HOST_RESIDUE(id, name, T, type_id, allocated, roles) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_residue<T>(id, name, #T, type_id, allocated, roles))
#define PL_HOST_SCALARS(id, builder) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_builder("host_scalars", "", id, builder))
#define PL_HOST_CHANNEL(id, StorageT, builder) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_channel<StorageT>(id, builder))
#define PL_HOST_WORKSPACE(id, align, builder) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_workspace(id, align, builder))
#define PL_HOST_TRANSFER(id, builder) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_transfer(id, builder))
#define PL_HOST_DOMAIN(id, name) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_domain(id, name))
#define PL_HOST_FACT(builder) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_fact(builder))
#define PL_HOST_SCHEDULER(builder) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_declaration("host_scheduler", "", (builder).json()))
#define PL_HOST_VARIANTS(stage, builder) PL_DETAIL_CHECKED_REGISTER(::pl::detail::host_declaration("host_variants", stage, (builder).json()))
#define PL_HOST_TARGET() PL_DETAIL_CHECKED_REGISTER(::pl::detail::add("host_target", "", "\"name\":\"\""))
#define PL_HOST_DECLARATIONS(owner) PL_DETAIL_CHECKED_REGISTER(::pl::detail::add("host_declarations", "", ::pl::detail::text("owner", owner)))

namespace pl::detail {
inline void neutral_host_abi_types() {
#define PL_DETAIL_NEUTRAL_ABI(id, T, ...) \
  host_abi_type<T>(id, #T, {PL_DETAIL_JOIN(PL_DETAIL_FIELDS_, PL_DETAIL_PARAMS_NARGS(__VA_ARGS__))(T, __VA_ARGS__)}, true);
  PLK_HOST_ABI_TYPES(PL_DETAIL_NEUTRAL_ABI)
#undef PL_DETAIL_NEUTRAL_ABI
}
} // namespace pl::detail
#else
#define PL_STAGE(...)
#define PL_PARAMS(...)
#define PL_NO_PARAMS(...)
#define PL_PARAMS_OPAQUE(...)
#define PL_TYPE(...)
#define PL_ROLE(...)
#define PL_EXPORT(...)
#define PL_CALL(...)
#define PL_ENDPOINT(...)
#define PL_CLAIM(...)
#define PL_CLAIM_SMEM(...)
#define PL_CLAIM_EX(...)
#define PL_CLAIM_REGS(...)
#define PL_VARIANT(...)
#define PL_VARIANT_ENTRY(...)
#define PL_DESCRIPTOR(...)
#define PL_TOPOLOGY(...)
#define PL_SCHEDULER_STRATEGY(...)
#define PL_BINDING(...)
#define PL_SITE(...)
#define PL_EXPORT_TENSOR(...)
#define PL_IMPORT_TENSOR(...)
#define PL_HINT(...)
#define PL_HOST_GROUP(...)
#define PL_HOST_ABI_TYPE(...)
#define PL_HOST_ABI_EMPTY(...)
#define PL_HOST_RESIDUE(...)
#define PL_HOST_SCALARS(...)
#define PL_HOST_CHANNEL(...)
#define PL_HOST_WORKSPACE(...)
#define PL_HOST_TRANSFER(...)
#define PL_HOST_DOMAIN(...)
#define PL_HOST_FACT(...)
#define PL_HOST_SCHEDULER(...)
#define PL_HOST_VARIANTS(...)
#define PL_HOST_TARGET(...)
#define PL_HOST_DECLARATIONS(...)
#endif
