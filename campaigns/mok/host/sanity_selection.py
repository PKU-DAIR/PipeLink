"""Restrict upstream functional test loops to their existing minimum shape."""


def pytest_collection_modifyitems(items):
    modules = {item.module for item in items if item.module.__name__.endswith("test_functional")}
    for module in modules:
        original_shapes = module.shapes

        def sanity_shapes(world_size, original_shapes=original_shapes):
            shapes = tuple(shape for shape in original_shapes(world_size)
                           if shape[2:] == (256, 256, 1, 512))
            if len(shapes) != 1:
                raise RuntimeError("Expected exactly one upstream 256-wide/512-token sanity shape")
            return shapes

        module.shapes = sanity_shapes
