# hungarian-assignment-algorithm-py
#
# Package entry point. Re-exports the public API:
#   - solve      -> function: find minimum-cost assignment
#   - HungarianError -> exception: raised on invalid input
#
# Keeping __init__ thin avoids a heavy import cost for users who only
# need the top-level name; everything delegates to core.solve.

from .core import HungarianError, solve

__all__ = ["solve", "HungarianError"]
