# Contract-authored TileLang epilogue

The source walkthrough, manifest-derived facts, zero-hand-macro annotation table,
partner-invariance check and dated runtime evidence are in
[T6 — Author a stage in TileLang](../../../docs/tutorials/T6-tilelang.md).

## Lower and pack

Use the pinned Linux x86_64 TileLang environment from
[T6's prerequisites](../../../docs/tutorials/T6-tilelang.md#goal-and-prerequisites).
From the repository root, first prepare each profile's `tl-reg.plink` and
`rowscale.plink` with the [complete build recipe](../../../docs/tutorials/T6-tilelang.md#2-lower-pack-and-link-every-profile).
Then lower this recipe against those opaque packages and pack the result:

```sh
source tools/env.sh
export TMPDIR="$PIPELINK_SCRATCH_ROOT/t6"
mkdir -p "$TMPDIR"
export PIPELINK_TILELANG_PYTHON="$PWD/frontends/tilelang/.venv/bin/python"
export TILELANG_INCLUDE_DIR="$PWD/frontends/tilelang/.venv/lib/python3.11/site-packages/tilelang/src"
export CUTLASS_INCLUDE_DIR="$PWD/third_party/cutlass/include"
export EX=examples/one-epilogue-many-producers
for P in sm_103a sm_90a sm_120a; do
  O="$PWD/$EX/out/$P"
  PIPELINK_ACC_PARTNER="$O/packed/tl-reg.plink" PIPELINK_SCALE_PARTNER="$O/packed/rowscale.plink" \
    "$PIPELINK_TILELANG_PYTHON" -m pipelink_tilelang \
    --kernel "$EX/consumer-tilelang/epilogue_contract.py" --profile "$P" --out "$TMPDIR/$P/tl-reg"
  uv run --project pipec --locked pipec2 pack \
    "$TMPDIR/$P/tl-reg/generation/tilelang-epilogue/roles.cu" \
    -o "$TMPDIR/$P/tl-epilogue.plink" \
    -I "$CUTLASS_INCLUDE_DIR" -I "$TILELANG_INCLUDE_DIR" --profile "$P"
done
```

Lowering and packing are GPU-free. Linking, same-source controls and runtime
acceptance use the [parent example workflow](../README.md#commands).
