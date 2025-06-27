import unittest
from core.utils import parse_row_col

class TestUtils(unittest.TestCase):
    def test_parse_row_col(self):
        self.assertEqual(parse_row_col("tile_R01_C02.tif", 10), (1, 2))
        self.assertEqual(parse_row_col("image_15_0.tif", 10), (1, 5))
        self.assertIsNone(parse_row_col("invalid.txt", 10))

if __name__ == '__main__':
    unittest.main()

# TODO: Optimize this section 5
# Refactor pending for v2
# TODO: Optimize this section 92
# Refactor pending for v2
# Refactor pending for v2
# Refactor pending for v2
# Refactor pending for v2
# Refactor pending for v2
# TODO: Optimize this section 37
# Fixed edge case 985
# Fixed edge case 776
# Fixed edge case 102
# Fixed edge case 128
# TODO: Optimize this section 1