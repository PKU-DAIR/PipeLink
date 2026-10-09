"""Lower a declared TileLang recipe into plain annotated stage translation units."""
import argparse
import importlib.util
import os
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", help="pipec2 profile; defaults to the demo target or sm_103a")
    parser.add_argument("--demo", help="composition demo whose entry pins contract partners")
    parser.add_argument("--composition", type=Path, help="explicit composition registry JSON")
    args = parser.parse_args()
    if args.demo is not None and args.composition is None:
        parser.error("--demo requires --composition")
    cuda_home = os.environ["PIPELINK_CUDA_HOME"]
    os.environ["CUDA_HOME"] = cuda_home
    # Lowering is CPU-only; never create a CUDA context on a shared GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    from tilelang import tvm
    from tilelang.env import CUDA_HOME
    from . import demo_profile, load_composition, set_context
    from .capture import capture
    from .annotate import AnnotationError

    if Path(CUDA_HOME).resolve() != Path(cuda_home).resolve():
        raise AnnotationError("CUDA_HOME", "TileLang did not select PIPELINK_CUDA_HOME")
    composition = load_composition(args.composition) if args.composition is not None else None
    context = set_context(args.profile, args.demo, composition)
    if args.demo is not None:
        expected = demo_profile(context.composition, args.demo)
        if context.profile != expected:
            raise AnnotationError("profile", f"--profile {context.profile} disagrees with demo {args.demo} target {expected}")
    spec = importlib.util.spec_from_file_location("pipelink_kernel_recipe", args.kernel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "PIPELINK"):
        module.PIPELINK.validate()
    if hasattr(module, "STAGE"):
        for channel in module.STAGE.channels:
            if channel.partner.profile != context.profile:
                raise AnnotationError("profile", f"{channel.partner.path} is packed for {channel.partner.profile}, not {context.profile}")
    random.seed(module.SEED)
    np.random.seed(module.SEED)
    target = tvm.target.Target({"kind": "cuda", "arch": context.profile})

    def lower(*arguments):
        result = capture(module.program(*arguments), target=target, pass_configs=module.PASS_CONFIGS)
        generated = {"source": result.kernel_source, "tir": result.snapshots["pre_codegen"],
                     "facts": result.facts, "passes": result.passes, "snapshots": result.snapshots,
                     "modules": result.modules, "layout_objects": result.layout_objects}
        if hasattr(module, "STAGE"):
            generated["external"] = {
                role: [record for channel in module.STAGE.channels if channel.role == role
                       for record in channel.external]
                for role in module.STAGE.roles
            }
        return generated

    if hasattr(module, "TENSOR_EXPORT"):
        from .tensor_export import emit
        emit(args.out, module, lower())
    elif hasattr(module, "STAGE"):
        generated = {depth: lower(depth) for depth in module.DEPTHS} if module.DEPTHS else {None: lower()}
        module.EMITTER(args.out, module, generated)
    elif hasattr(module, "EMITTER"):
        module.EMITTER(args.out, module, lower())
    elif module.KIND == "ws_gemm":
        from .ws import emit
        generated = {depth: lower(depth) for depth in module.DEPTHS}
        emit(args.out, module, generated, target=target, planes=module.PLANES)
    else:
        raise AnnotationError("KIND", "unsupported recipe: " + module.KIND)
    print(args.out)


if __name__ == "__main__":
    main()
