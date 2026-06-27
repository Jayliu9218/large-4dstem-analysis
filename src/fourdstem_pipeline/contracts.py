"""Unified data contract and coordinate conventions for the 4D-STEM pipeline.

All modules MUST follow these conventions to avoid x/y or bbox-order
confusion downstream (Stage 2 ROI, py4DSTEM, PNG overlays, orientation
preview, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AxisOrder = Literal["nav_y_nav_x_q_y_q_x"]
BBoxOrder = Literal["y0_y1_x0_x1"]
CenterOrder = Literal["y_x"]


@dataclass
class DataContract:
    """Coordinate conventions used throughout the pipeline.

    Attributes
    ----------
    axis_order:
        4D data shape convention: ``nav_y_nav_x_q_y_q_x`` means
        ``[navigation_y, navigation_x, detector_q_y, detector_q_x]``.
    bbox_order:
        Bounding-box list convention: ``y0_y1_x0_x1`` means
        ``[y_start, y_end, x_start, x_end]``.
    center_order:
        Point / centre convention: ``y_x`` means ``[y, x]``
        (row-major, consistent with ``np.argwhere``).
    array_shape:
        Optional resolved 4D shape ``(nav_y, nav_x, q_y, q_x)``.
    nav_shape:
        Optional 2D navigation shape ``(nav_y, nav_x)``.
    sig_shape:
        Optional 2D signal shape ``(q_y, q_x)``.
    """

    axis_order: AxisOrder = "nav_y_nav_x_q_y_q_x"
    bbox_order: BBoxOrder = "y0_y1_x0_x1"
    center_order: CenterOrder = "y_x"
    array_shape: tuple[int, int, int, int] | None = None
    nav_shape: tuple[int, int] | None = None
    sig_shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, str | list[int] | None]:
        """Serialize to a plain dict suitable for JSON / YAML reports."""
        result: dict[str, str | list[int] | None] = {
            "axis_order": self.axis_order,
            "bbox_order": self.bbox_order,
            "center_order": self.center_order,
        }
        if self.array_shape is not None:
            result["array_shape"] = list(self.array_shape)
        if self.nav_shape is not None:
            result["nav_shape"] = list(self.nav_shape)
        if self.sig_shape is not None:
            result["sig_shape"] = list(self.sig_shape)
        return result
