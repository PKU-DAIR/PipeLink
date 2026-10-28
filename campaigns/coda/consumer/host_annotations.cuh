#pragma once
#include "params.h"
#include <pipelink/stage.hpp>
#ifndef CODA_READER
#define CODA_READER "coda.swiglu:epilogue"
#endif

PL_HOST_GROUP("coda.epilogue", "coda.epilogue.params", "coda_epilogue",
              plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_ABI_TYPE(200, CodaEpilogueParams, a, b, output, M, N, K);
#define CODA_ANNOTATE_VIEW(T) PL_HOST_ABI_TYPE(201, T, residue, scalars)
CODA_ANNOTATE_VIEW(CODA_ROLE_VIEW);
#undef CODA_ANNOTATE_VIEW
PL_HOST_RESIDUE(2, "coda.epilogue.residue", CodaEpilogueParams, 200, true,
                pl::host::roles({CODA_READER}));
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({CODA_READER}))
    .field(pl::host::scalar(1, "coda.M", "i64").caller("M"))
    .field(pl::host::scalar(2, "coda.N", "i64").caller("N"))
    .field(pl::host::scalar(3, "coda.K", "i64").caller("K")));
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("coda.x", "bf16").rank(2).read().caller("x"));
PL_HOST_CHANNEL(2, uint16_t, pl::host::channel("coda.w", "bf16").rank(2).read().caller("w"));
PL_HOST_CHANNEL(3, uint16_t, pl::host::channel("coda.out", "bf16").rank(2).write().caller("out"));
#if CODA_PROFILE == 103
PL_HOST_SCHEDULER(pl::host::scheduler("coda.sm103"));
#elif CODA_PROFILE == 90
PL_HOST_SCHEDULER(pl::host::scheduler("coda.sm90"));
#elif CODA_PROFILE == 120
PL_HOST_SCHEDULER(pl::host::scheduler("coda.sm120.owner"));
#else
#error "Define CODA_PROFILE as 103, 90, or 120"
#endif
