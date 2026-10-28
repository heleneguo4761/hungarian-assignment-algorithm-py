from __future__ import annotations

import math

__all__ = ["solve", "HungarianError"]


class HungarianError(ValueError):
    """Raised when the cost matrix is empty, non-rectangular, or contains NaN."""


def solve(cost_matrix):
    """Find a minimum-cost assignment of rows to columns.

    Args:
        cost_matrix: A non-empty, rectangular sequence of sequences whose
            entries are ints or floats. NaN entries are rejected.

    Returns:
        A list of (row, col, cost) triples of length equal to the number of
        rows in the input. The col indices are always real (never padded),
        and represent a minimum-cost assignment of each row to a distinct
        column.

    Raises:
        HungarianError: If the matrix is empty, ragged, contains NaN, or has
            no feasible finite assignment.
    """
    matrix, real_rows, real_cols = _validate_and_normalize(cost_matrix)
    assignment = _kuhn_munkres(matrix, real_rows, real_cols)
    result = []
    for r in range(real_rows):
        c = assignment[r]
        if c >= real_cols:
            raise HungarianError(
                "No feasible assignment: at least one row has no finite cost"
            )
        cost = matrix[r][c]
        if math.isinf(cost):
            raise HungarianError(
                "No feasible assignment: at least one row has no finite cost"
            )
        result.append((r, c, cost))
    return result


def _validate_and_normalize(cost_matrix):
    if not isinstance(cost_matrix, (list, tuple)):
        raise HungarianError("cost_matrix must be a list or tuple of rows")
    if len(cost_matrix) == 0:
        raise HungarianError("cost_matrix must not be empty")
    width = None
    for row in cost_matrix:
        if not isinstance(row, (list, tuple)):
            raise HungarianError("each row must be a list or tuple")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise HungarianError("cost_matrix must be rectangular (no ragged rows)")
    if width == 0:
        raise HungarianError("cost_matrix rows must not be empty")

    real_rows = len(cost_matrix)
    real_cols = width
    if real_rows > real_cols:
        raise HungarianError(
            "cost_matrix must be wider than tall (or square); "
            "a matrix taller than wide has no feasible perfect assignment"
        )

    n = real_rows
    out = []
    for r in range(real_rows):
        new_row = []
        for cell in cost_matrix[r]:
            f = float(cell)
            if math.isnan(f):
                raise HungarianError("cost_matrix must not contain NaN values")
            new_row.append(f)
        out.append(new_row)
    return out, real_rows, real_cols


def _kuhn_munkres(matrix, real_rows, real_cols):
    INF = float("inf")
    # u_pot[i] is the potential for row i; v_pot[j] is the potential for col j.
    u_pot = [0.0] * real_rows
    v_pot = [0.0] * real_cols
    # match_row[i] = column assigned to row i, or -1.
    match_row = [-1] * real_rows
    # match_col[j] = row assigned to column j, or -1.
    match_col = [-1] * real_cols

    for root in range(real_rows):
        # BFS from unmatched row 'root' to find an augmenting path.
        slack = [INF] * real_cols          # slack[j] = min reduced cost edge to col j
        slack_row = [-1] * real_cols       # slack_row[j] = row achieving slack[j]
        visited = [False] * real_cols     # visited columns

        # Initialize slack from the root row.
        for j in range(real_cols):
            cost = matrix[root][j]
            if math.isinf(cost):
                slack[j] = INF
            else:
                slack[j] = cost - u_pot[root] - v_pot[j]
            slack_row[j] = root

        col_to_match = -1
        while True:
            # Find the unvisited column with minimum slack.
            min_slack = INF
            min_j = -1
            for j in range(real_cols):
                if not visited[j] and slack[j] < min_slack:
                    min_slack = slack[j]
                    min_j = j

            if min_j == -1:
                # No reachable unvisited column with finite slack: infeasible.
                raise HungarianError(
                    "No feasible assignment: at least one row has no finite cost"
                )

            # Update potentials by min_slack.
            u_pot[root] += min_slack
            for j in range(real_cols):
                if visited[j]:
                    v_pot[j] -= min_slack
                else:
                    slack[j] -= min_slack

            # The minimum-slack column is now tight; add it to the tree.
            visited[min_j] = True

            if match_col[min_j] == -1:
                # Found an augmenting path ending at an unmatched column.
                col_to_match = min_j
                break

            # The column is matched; add its matched row to the tree.
            r = match_col[min_j]
            for j in range(real_cols):
                if visited[j]:
                    continue
                cost = matrix[r][j]
                if math.isinf(cost):
                    continue
                s = cost - u_pot[r] - v_pot[j]
                if s < slack[j]:
                    slack[j] = s
                    slack_row[j] = r

        # Walk back along slack_row / match_row, flipping the matching.
        cur = col_to_match
        while cur != -1:
            r = slack_row[cur]
            next_match = match_row[r] if r != root else -1
            match_row[r] = cur
            match_col[cur] = r
            cur = next_match

    return match_row
