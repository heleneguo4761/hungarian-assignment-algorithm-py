import math
import unittest

from hungarian_assignment_algorithm_py import HungarianError, solve


class TestSquare(unittest.TestCase):
    def test_2x2_basic(self):
        m = [[1, 2], [2, 1]]
        r = solve(m)
        self.assertEqual(len(r), 2)
        cols = {c for _, c, _ in r}
        self.assertEqual(cols, {0, 1})
        # Optimal: (0,0)=1, (1,1)=1 -> total 2
        d = {row: col for row, col, _ in r}
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 1)
        total = sum(c for _, _, c in r)
        self.assertEqual(total, 2)

    def test_3x3_classic(self):
        m = [[4, 1, 7], [2, 3, 5], [5, 8, 6]]
        r = solve(m)
        cols = [c for _, c, _ in r]
        self.assertEqual(sorted(cols), [0, 1, 2])
        # Optimal: (0,1)=1, (1,0)=2, (2,2)=6 -> total 9
        d = {row: col for row, col, _ in r}
        self.assertEqual(d[0], 1)
        self.assertEqual(d[1], 0)
        self.assertEqual(d[2], 2)
        self.assertEqual(sum(c for _, _, c in r), 9)

    def test_diagonal(self):
        m = [[5, 9, 1], [10, 3, 7], [8, 2, 4]]
        r = solve(m)
        cols = [c for _, c, _ in r]
        self.assertEqual(sorted(cols), [0, 1, 2])
        # Optimal: (0,2)=1, (1,1)=3, (2,0)=8 -> total 12
        d = {row: col for row, col, _ in r}
        self.assertEqual(d[0], 2)
        self.assertEqual(d[1], 1)
        self.assertEqual(d[2], 0)
        self.assertEqual(sum(c for _, _, c in r), 12)

    def test_all_zero(self):
        m = [[0, 0], [0, 0]]
        r = solve(m)
        self.assertEqual(sum(c for _, _, c in r), 0)
        self.assertEqual(sorted(c for _, c, _ in r), [0, 1])

    def test_single_element(self):
        m = [[42]]
        r = solve(m)
        self.assertEqual(r, [(0, 0, 42.0)])


class TestRectangular(unittest.TestCase):
    def test_more_rows_than_cols(self):
        m = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaises(HungarianError):
            solve(m)

    def test_more_cols_than_rows(self):
        m = [[1, 2, 3], [4, 5, 6]]
        r = solve(m)
        self.assertEqual(len(r), 2)
        cols = {c for _, c, _ in r}
        self.assertTrue(cols.issubset({0, 1, 2}))
        # Optimal: (0,0)=1, (1,1)=5 -> total 6
        d = {row: col for row, col, _ in r}
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 1)
        self.assertEqual(sum(c for _, _, c in r), 6)

    def test_one_row_many_cols(self):
        m = [[7, 3, 9, 1, 5]]
        r = solve(m)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 0)
        self.assertEqual(r[0][1], 3)  # min cost col
        self.assertEqual(r[0][2], 1.0)


class TestFloats(unittest.TestCase):
    def test_float_costs(self):
        m = [[1.5, 2.5], [2.5, 1.5]]
        r = solve(m)
        d = {row: col for row, col, _ in r}
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 1)
        self.assertAlmostEqual(sum(c for _, _, c in r), 3.0, places=9)

    def test_mixed_int_float(self):
        m = [[1, 2.5], [2, 1.5]]
        r = solve(m)
        d = {row: col for row, col, _ in r}
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 1)


class TestLarge(unittest.TestCase):
    def test_5x5(self):
        m = [[9, 2, 7, 8, 1],
             [6, 4, 3, 5, 8],
             [2, 5, 7, 9, 3],
             [1, 3, 9, 7, 4],
             [8, 4, 2, 6, 5]]
        r = solve(m)
        self.assertEqual(len(r), 5)
        self.assertEqual(sorted(c for _, c, _ in r), [0, 1, 2, 3, 4])
        # The algorithm returns a valid permutation; just verify minimality
        # by confirming no cheaper assignment exists via brute force.
        import itertools
        best = min(sum(m[i][p[i]] for i in range(5)) for p in itertools.permutations(range(5)))
        self.assertEqual(sum(c for _, _, c in r), best)


class TestErrors(unittest.TestCase):
    def test_empty_matrix(self):
        with self.assertRaises(HungarianError):
            solve([])

    def test_ragged_matrix(self):
        with self.assertRaises(HungarianError):
            solve([[1, 2], [3]])

    def test_nan_entry(self):
        with self.assertRaises(HungarianError):
            solve([[1, 2], [3, float('nan')]])

    def test_empty_row(self):
        with self.assertRaises(HungarianError):
            solve([[], []])

    def test_non_list_matrix(self):
        with self.assertRaises(HungarianError):
            solve("not a matrix")

    def test_non_list_row(self):
        with self.assertRaises(HungarianError):
            solve([[1, 2], "not a row"])


class TestTypes(unittest.TestCase):
    def test_tuples_accepted(self):
        m = ((1, 2), (2, 1))
        r = solve(m)
        self.assertEqual(sum(c for _, _, c in r), 2)

    def test_costs_are_float(self):
        m = [[1, 2], [2, 1]]
        r = solve(m)
        for _, _, c in r:
            self.assertIsInstance(c, float)


if __name__ == '__main__':
    unittest.main()
