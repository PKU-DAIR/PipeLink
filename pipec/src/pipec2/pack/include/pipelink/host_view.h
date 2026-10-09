#pragma once

/* The view a role-less host-group stage receives as the first argument of each
 * called export: its owner's residue and shared scalar image inside the kernel
 * params, NULL for a record the owner does not declare. The caller passes its own
 * params view; the linker-generated call trampoline builds this view from the
 * same params image, so it never names another owner's records. */
typedef struct pipelink_host_view {
  const void* residue;
  const void* scalars;
} pipelink_host_view;
