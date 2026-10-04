import numpy as np

from arc3.perception import (
    background_color,
    changed_bbox,
    find_objects,
    frame_diff,
    grid_signature,
    grid_stats,
    to_ascii,
)


def make_grid() -> np.ndarray:
    grid = np.zeros((6, 6), dtype=np.int16)
    grid[0:3, 2] = 1
    grid[2, 3:5] = 1
    grid[5, 5] = 3
    return grid


def test_grid_stats_and_ascii():
    grid = make_grid()
    stats = grid_stats(grid)
    assert stats["shape"] == [6, 6]
    assert stats["unique_values"] == [0, 1, 3]
    assert stats["nonzero_count"] == 6
    assert to_ascii(grid).splitlines()[2] == "001110"


def test_signature_is_stable_and_border_aware():
    a = make_grid()
    b = a.copy()
    assert grid_signature(a) == grid_signature(b)
    b[0, 0] = 7
    assert grid_signature(a) != grid_signature(b)
    assert grid_signature(a, border=1) == grid_signature(b, border=1)


def test_find_objects_connected_components():
    objects = find_objects(make_grid())
    assert background_color(make_grid()) == 0
    by_color = {o.color: o for o in objects}
    assert len(objects) == 2
    assert by_color[1].size == 5
    assert (
        by_color[1].min_x,
        by_color[1].max_x,
        by_color[1].min_y,
        by_color[1].max_y,
    ) == (2, 4, 0, 2)
    assert make_grid()[by_color[1].anchor_y, by_color[1].anchor_x] == 1
    assert by_color[3].size == 1


def test_frame_diff_and_bbox():
    a = make_grid()
    b = a.copy()
    assert frame_diff(a, b) == 0
    assert changed_bbox(a, b) is None
    b[4, 1] = 2
    b[5, 5] = 0
    assert frame_diff(a, b) == 2
    assert changed_bbox(a, b) == (1, 5, 4, 5)
    assert frame_diff(a, np.zeros((3, 3), dtype=np.int16)) == 9
