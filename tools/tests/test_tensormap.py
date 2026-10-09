import unittest

from tools.tensormap import encode_tiled


class TensorMapShapeTests(unittest.TestCase):
    def test_rejects_inconsistent_rank_before_native_array_access(self):
        for dims, strides, box in (
            ((), (), ()),
            ((64,) * 6, (128,) * 5, (64,) * 6),
            ((64, 128), (), (64, 128)),
            ((64, 128), (128,), (64,)),
            ((64, 128), (128, 256), (64, 128)),
        ):
            with self.subTest(dims=dims, strides=strides, box=box):
                with self.assertRaises(ValueError):
                    encode_tiled(0, dtype=6, dims=dims, strides_bytes=strides,
                                 box=box, swizzle=3)
