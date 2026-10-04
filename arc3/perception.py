"""Grid perception: statistics, signatures, ASCII rendering, objects, diffs."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Sequence

import numpy as np

from arc3.types import GridObject

_ASCII = "0123456789abcdefghijklmnopqrstuvwxyz"


def to_grid(frame: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    return np.asarray(frame, dtype=np.int16)


def grid_stats(grid: np.ndarray) -> dict[str, object]:
    values, counts = np.unique(grid, return_counts=True)
    return {
        "shape": [int(s) for s in grid.shape],
        "unique_values": [int(v) for v in values],
        "value_counts": {int(v): int(c) for v, c in zip(values, counts)},
        "nonzero_count": int(np.count_nonzero(grid)),
    }


def grid_signature(grid: np.ndarray, border: int = 0) -> str:
    """Stable hash of the grid; `border` ignores a frame of edge cells (HUD ticks)."""
    core = grid
    if border > 0 and min(grid.shape) > 2 * border:
        core = grid[border:-border, border:-border]
    data = np.ascontiguousarray(core, dtype=np.int16)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(data.shape).encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


def to_ascii(grid: np.ndarray) -> str:
    return "\n".join(
        "".join(_ASCII[int(v)] if 0 <= int(v) < len(_ASCII) else "?" for v in row)
        for row in grid
    )


def background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def find_objects(
    grid: np.ndarray,
    background: int | None = None,
    min_size: int = 1,
) -> list[GridObject]:
    """4-connected single-colour components, excluding the background colour."""
    if grid.size == 0:
        return []
    bg = background_color(grid) if background is None else background
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects: list[GridObject] = []
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0] or int(grid[y0, x0]) == bg:
                continue
            color = int(grid[y0, x0])
            seen[y0, x0] = True
            queue = deque([(y0, x0)])
            cells: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(cells) < min_size:
                continue
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            cy = (min(ys) + max(ys)) / 2
            cx = (min(xs) + max(xs)) / 2
            anchor_y, anchor_x = min(
                cells, key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2
            )
            objects.append(
                GridObject(
                    color=color,
                    size=len(cells),
                    min_x=min(xs),
                    max_x=max(xs),
                    min_y=min(ys),
                    max_y=max(ys),
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                )
            )
    return objects


def frame_diff(before: np.ndarray, after: np.ndarray) -> int:
    """Number of changed cells; a shape change counts every cell of `after`."""
    if before.shape != after.shape:
        return int(after.size)
    return int(np.count_nonzero(before != after))


def changed_bbox(
    before: np.ndarray, after: np.ndarray
) -> tuple[int, int, int, int] | None:
    """(min_x, max_x, min_y, max_y) of changed cells, or None when nothing changed."""
    if before.shape != after.shape:
        return (0, after.shape[1] - 1, 0, after.shape[0] - 1)
    ys, xs = np.nonzero(before != after)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
