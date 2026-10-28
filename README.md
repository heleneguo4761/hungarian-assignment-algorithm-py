# hungarian-assignment-algorithm-py

Solves the minimum-cost assignment problem for two sets using the Kuhn-Munkres augmenting-path method.

## Usage

```python
from hungarian_assignment_algorithm_py import solve

# Square matrix: 3 workers, 3 tasks
result = solve([[4, 1, 7], [2, 3, 5], [5, 8, 6]])
for row, col, cost in result:
    print(f"row {row} -> col {col}, cost {cost}")
# row 0 -> col 1, cost 1.0
# row 1 -> col 0, cost 2.0
# row 2 -> col 2, cost 6.0
```

For wider-than-tall matrices (fewer rows than columns), each row is still
assigned a distinct column:

```python
result = solve([[1, 2, 3], [4, 5, 6]])
# [(0, 0, 1.0), (1, 1, 5.0)]
```

## Why this library exists

The assignment problem shows up in scheduling, matching, and logistics work.
Most Python implementations either pull in NumPy (a heavy dependency for
what is fundamentally an O(n^3) algorithm on small inputs) or live inside a
larger package that obscures the algorithm itself. This library is a single,
dependency-free function that implements the Kuhn-Munkres augmenting-path
method directly, so the algorithm is auditable and the surface area is
exactly one name.

The trade-off: no vectorisation, so for very large matrices (n in the
thousands) a NumPy-backed solver will be faster. For the common case of n
under a few hundred, the difference is negligible and the zero-dependency
property is worth more.

## Edge cases worth knowing

- **Rectangular inputs**: the matrix must be wider than it is tall (more
columns than rows) or square. A matrix taller than wide raises
`HungarianError` because there is no feasible perfect assignment of rows to
the available columns. Padded columns beyond the original width never
appear in results; assignments always reference real columns.

- **Infinite costs**: entries of `float('inf')` are permitted and treated as
"this assignment is forbidden." If no feasible finite assignment exists,
`HungarianError` is raised.

- **NaN**: rejected up front with `HungarianError`, since NaN propagates
through comparisons and makes the slack update non-deterministic.

- **Tie handling**: when multiple columns achieve minimum slack during the
potential update, the lowest-indexed one is chosen. This makes output
deterministic but means the returned assignment is not necessarily unique;
another assignment with the same total cost may exist.

- **Cost types**: costs may be `int` or `float`; the result costs are always
`float` because the algorithm works with dual potentials that are
fractional in general.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
