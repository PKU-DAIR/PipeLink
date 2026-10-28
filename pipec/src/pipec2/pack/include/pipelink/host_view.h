#pragma once

/* First argument of every export called on a role-less host-group stage: pointers
 * to the owner's residue and the shared scalar image inside the kernel params,
 * each NULL if the owner does not declare it. The caller passes its own params;
 * the linker-generated call trampoline builds this view from the same params
 * image, so the callee only sees its own owner's records. */
typedef struct pipelink_host_view {
  const void* residue;
  const void* scalars;
} pipelink_host_view;
