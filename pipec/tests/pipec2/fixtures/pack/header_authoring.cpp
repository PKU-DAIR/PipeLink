#include <pipelink/stage.hpp>

struct alignas(16) Packet { unsigned values[4]; };
namespace sample::nested { class Token { public: unsigned value; }; }
#if defined(MISSTATED_TYPE_MEMBERS)
PL_TYPE(Packet, unsigned values[4];);
#else
PL_TYPE(Packet, alignas(16) unsigned values[4];);
#endif
PL_TYPE(sample::nested::Token, unsigned value;);

extern "C" Packet plx_packet(const Packet&, Packet*, const Packet*, sample::nested::Token*);
extern "C" void plx_pointer(Packet*);
extern "C" Packet plx_value(Packet);
PL_EXPORT(packet, plx_packet);
PL_EXPORT(pointer, plx_pointer);
PL_EXPORT(value, plx_value);
PL_CALL(packet_call, plx_packet, pl::consume, "packet");
PL_IMPORT_TENSOR(input, plx_pointer,
    (pl::identity{"value", "work", "v1", "f32", "exact", {{"m", 0, 1}}}),
    (pl::args{{"fragment", "dest_fragment"}}),
    (pl::demand_positions{pl::coord{"t", 0, 1}, [](int t) { return std::array<int, 3>{t, t, 0}; }}),
    (pl::strings{"site"}), "done", pl::lends{}, {});

PL_STAGE("header.authoring", "1");
#if defined(TYPED_PARAMS)
struct Params {
  char first;
  alignas(16) int value;
};
PL_PARAMS(Params, first, value);
#elif defined(WIDE_PARAMS)
struct ThirtyTwo {
  int a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
  int q, r, s, t, u, v, w, x, y, z, aa, ab, ac, ad, ae, af;
};
PL_PARAMS(ThirtyTwo, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p,
    q, r, s, t, u, v, w, x, y, z, aa, ab, ac, ad, ae, af);
#elif defined(OPAQUE_PARAMS)
class Opaque {
public:
  int first;
  ~Opaque() {}
private:
  int second;
};
static_assert(!std::is_standard_layout_v<Opaque> && !std::is_trivially_copyable_v<Opaque>);
PL_PARAMS_OPAQUE(Opaque);
#else
PL_NO_PARAMS();
#endif
#if defined(DUPLICATE_PARAMS)
PL_NO_PARAMS();
#endif

#if defined(OMIT_INIT_WARP)
constexpr pl::init initializer{nullptr, 0, "stage"};
#else
constexpr pl::init initializer{nullptr, 0, "stage", 0};
#endif
PL_ENDPOINT(barrier, pl::endpoint("barrier", pl::produce, "worker")
    .space("smem").scope("cta").proxy("async").tile("u32", {1})
    .layout(pl::affine{pl::tree{1}, pl::tree{4}, 32})
    .protocol(pl::mbarrier{
        pl::side{"full", pl::arrive{"worker", 1}, 4},
        pl::side{"empty", pl::arrive{"worker", 1}, 0}, initializer}));
