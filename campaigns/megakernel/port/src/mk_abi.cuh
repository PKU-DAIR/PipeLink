#pragma once

#include "llama.cuh"
#include <pipelink/stage.hpp>

using mk_config = ::config;
using mk_globals = llama_1b_globals;
using mk_state = megakernel::state<mk_config>;

#define MK_OP_CALL(NAME, SLOT, FN)                                              \
    PL_EXPORT(FN, org_pipelink_megakernel_##NAME##_##FN);                         \
    PL_CALL(megakernel_slot##SLOT##_##FN, org_pipelink_megakernel_##NAME##_##FN,  \
            pl::consume, #FN);

#if defined(PIPELINK_EMIT)
// The upstream op bodies own the C facts; the wrapper supplies only their
// expressions and semantic usage. No unused instruction resource gets a claim.
#define MK_ANNOTATE_OP(NAME, SLOT, OP_T, PAGES, SEMAPHORES, SCRATCH, BAR, ADD)    \
    static const int mk_annotations_##NAME = [] {                              \
        std::string stage_id = "org.pipelink.megakernel." #NAME;                \
        std::replace(stage_id.begin(), stage_id.end(), '_', '-');               \
        const std::string phase = "phase:instr_" + stage_id;                    \
        PL_STAGE(stage_id.c_str(), "7309cec+pipelink.megakernel");              \
        PL_NO_PARAMS();                                                       \
        PL_DESCRIPTOR(mk_config::instruction_t);                               \
        PL_TOPOLOGY(pl::topology{mk_config::CLUSTER_BLOCKS,                     \
            pl::cluster{{mk_config::CLUSTER_BLOCKS, 1, 1}}, true});             \
        if constexpr (OP_T::opcode == 0) {                                    \
            PL_CLAIM_EX(pl::claim("pipelink_megakernel_opcode_slot" #SLOT,      \
                "opcode").extent(1).at(OP_T::opcode));                         \
        } else {                                                              \
            PL_CLAIM("pipelink_megakernel_opcode_slot" #SLOT, "opcode",        \
                     1, 1, "kernel");                                         \
        }                                                                     \
        if constexpr ((PAGES) > 0) {                                          \
            PL_CLAIM("pipelink_megakernel_" #NAME "_pages", "instr_page",     \
                     PAGES, 1, phase.c_str());                                 \
        }                                                                     \
        if constexpr ((SEMAPHORES) > 0) {                                     \
            PL_CLAIM("pipelink_megakernel_" #NAME "_semaphores",              \
                     "instr_semaphore", SEMAPHORES, 1, phase.c_str());         \
        }                                                                     \
        if constexpr ((SCRATCH) > 0) {                                        \
            PL_CLAIM("pipelink_megakernel_" #NAME "_scratch", "instr_scratch",\
                     SCRATCH, 1, phase.c_str());                               \
        }                                                                     \
        if constexpr ((BAR) > 0) {                                            \
            PL_CLAIM_EX(pl::claim("pipelink_megakernel_bar_" #BAR,              \
                "named_barrier").extent(1).at(BAR).alias("MkBar" #BAR));       \
        }                                                                     \
        if constexpr ((BAR) == 3) {                                           \
            PL_HINT("named_barrier_0",                                        \
                "consumer: fixed CTA bar.sync 0 outside named_barrier allocation"); \
        }                                                                     \
        PL_HINT("gmem_barrier_protocol",                                      \
            "Advisory: g.Bar in params: volatile poll + __nanosleep acquire; " \
            "atomicAdd after cp.async.bulk.wait_group release (F9)");          \
        PL_HINT("instruction_context_bases_unused", true);                     \
        PL_HINT("uses_tmem", false);                                           \
        PL_HINT("uses_store_add_async", ADD);                                  \
        return 0;                                                             \
    }();
#else
#define MK_ANNOTATE_OP(...)
#endif

#define MK_DEFINE_OP_EXPORTS(NAME, SLOT, OP_T, PAGES, SEMS, SCRATCH, BAR, ADD)   \
    MK_DEFINE_OP_EXPORTS_BODY(org_pipelink_megakernel_##NAME##_, OP_T)           \
    MK_OP_CALL(NAME, SLOT, release_lid)                                          \
    MK_OP_CALL(NAME, SLOT, init_semaphores)                                      \
    MK_OP_CALL(NAME, SLOT, loader)                                              \
    MK_OP_CALL(NAME, SLOT, launcher)                                            \
    MK_OP_CALL(NAME, SLOT, consumer)                                            \
    MK_OP_CALL(NAME, SLOT, storer)                                              \
    MK_ANNOTATE_OP(NAME, SLOT, OP_T, PAGES, SEMS, SCRATCH, BAR, ADD)

#define MK_DEFINE_OP_EXPORTS_BODY(PREFIX, OP_T)                                 \
    extern "C" __attribute__((used)) __device__ int PREFIX##release_lid(       \
        const void *g, int *instruction, int query) {                            \
        int q = query;                                                           \
        return OP_T::controller::release_lid(                                    \
            *static_cast<const mk_globals *>(g),                                 \
            *reinterpret_cast<mk_config::instruction_t *>(instruction), q);      \
    }                                                                            \
    extern "C" __attribute__((used)) __device__ int PREFIX##init_semaphores(   \
        const void *g, void *s) {                                                \
        return OP_T::controller::init_semaphores(                                \
            *static_cast<const mk_globals *>(g), *static_cast<mk_state *>(s));   \
    }                                                                            \
    extern "C" __attribute__((used)) __device__ void PREFIX##loader(           \
        const void *g, void *s) {                                                \
        OP_T::loader::run(*static_cast<const mk_globals *>(g),                   \
                          *static_cast<mk_state *>(s));                           \
    }                                                                            \
    extern "C" __attribute__((used)) __device__ void PREFIX##launcher(         \
        const void *g, void *s) {                                                \
        OP_T::launcher::run(*static_cast<const mk_globals *>(g),                 \
                            *static_cast<mk_state *>(s));                         \
    }                                                                            \
    extern "C" __attribute__((used)) __device__ void PREFIX##consumer(         \
        const void *g, void *s) {                                                \
        OP_T::consumer::run(*static_cast<const mk_globals *>(g),                 \
                            *static_cast<mk_state *>(s));                         \
    }                                                                            \
    extern "C" __attribute__((used)) __device__ void PREFIX##storer(           \
        const void *g, void *s) {                                                \
        OP_T::storer::run(*static_cast<const mk_globals *>(g),                   \
                          *static_cast<mk_state *>(s));                           \
    }
