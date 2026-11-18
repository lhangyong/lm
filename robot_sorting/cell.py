# -*- coding: utf-8 -*-
"""分拣单元仿真：传送带、三轴机械臂、真空吸盘、剔除料箱。

现场机械臂只需替换 driver.py 里的驱动，状态机和安全逻辑不用改。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 工作坐标（毫米）：X 沿传送带，Y 横跨传送带（0=取料位，220=料箱），Z 向上
HOME = {"x": 20.0, "y": 0.0, "z": 60.0}
PICK_ABOVE = {"x": 200.0, "y": 0.0, "z": 60.0}
PICK_POS = {"x": 200.0, "y": 0.0, "z": 10.0}
BIN_ABOVE = {"x": 200.0, "y": 220.0, "z": 60.0}
BIN_POS = {"x": 200.0, "y": 220.0, "z": 20.0}


@dataclass
class SortStats:
    ok: int = 0
    ng: int = 0
    sorted: int = 0
    passed: int = 0
    manual: int = 0
    faults: int = 0


@dataclass
class Part:
    part_id: str
    image_name: str
    score: float = 0.0
    severity: str = "None"
    defect_ratio: float = 0.0


class SortingCell:
    """轻量物理模型：目标位姿 + 速度插值 + 真空/故障状态。"""

    def __init__(self, speed: float = 260.0):
        self.time_s = 0.0
        self.speed = float(speed)
        self.pose: Dict[str, float] = dict(HOME)
        self.pose["vacuum"] = 0.0
        self.target: Dict[str, float] = dict(HOME)
        self.guard_closed = True
        self.estop = False
        self.fault: Optional[str] = None
        self.vacuum_ok = True
        self.reject_bin = 0
        self.stats = SortStats()
        self.conveyor_pos = 0.0
        self.part_present = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.time_s = 0.0
        self.pose = dict(HOME)
        self.pose["vacuum"] = 0.0
        self.target = dict(HOME)
        self.estop = False
        self.fault = None
        self.vacuum_ok = True
        self.conveyor_pos = 0.0
        self.part_present = False

    def set_target(self, pose: Dict[str, float]) -> None:
        for key in ("x", "y", "z"):
            if key in pose:
                self.target[key] = float(pose[key])

    def set_vacuum(self, on: bool) -> None:
        self.pose["vacuum"] = 1.0 if on else 0.0
        if on and self.fault == "vacuum_loss":
            self.vacuum_ok = False
            self.pose["vacuum"] = 0.0
        elif not on:
            self.vacuum_ok = True

    def tick(self, dt: float) -> None:
        self.time_s += max(0.0, dt)
        if self.estop or self.fault == "jam":
            return
        step = self.speed * max(0.0, dt)
        for key in ("x", "y", "z"):
            delta = self.target[key] - self.pose[key]
            if abs(delta) <= step:
                self.pose[key] = self.target[key]
            else:
                self.pose[key] += math.copysign(step, delta)
        if self.fault == "vacuum_loss":
            self.vacuum_ok = False

    def at_target(self, tol: float = 1.5) -> bool:
        return all(abs(self.pose[k] - self.target[k]) <= tol for k in ("x", "y", "z"))

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        return {
            "arm": {k: round(float(self.pose[k]), 1) for k in ("x", "y", "z")},
            "vacuum": bool(self.pose["vacuum"]),
            "vacuum_ok": bool(self.vacuum_ok),
            "guard_closed": bool(self.guard_closed),
            "estop": bool(self.estop),
            "fault": self.fault,
            "reject_bin": int(self.reject_bin),
            "stats": {
                "ok": self.stats.ok, "ng": self.stats.ng,
                "sorted": self.stats.sorted, "passed": self.stats.passed,
                "manual": self.stats.manual, "faults": self.stats.faults,
            },
        }
