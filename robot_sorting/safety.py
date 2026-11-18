# -*- coding: utf-8 -*-
"""机械臂安全约束：工作空间、急停、防护门、真空确认。"""
from __future__ import annotations

from typing import Dict, List, Tuple


class ArmSafety:
    X_RANGE = (0.0, 400.0)
    Y_RANGE = (0.0, 250.0)
    Z_RANGE = (0.0, 100.0)

    def clamp_pose(self, pose: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        rejects: List[str] = []
        out = dict(pose)
        for key, (lo, hi) in (("x", self.X_RANGE), ("y", self.Y_RANGE), ("z", self.Z_RANGE)):
            value = float(out.get(key, 0.0))
            if value < lo or value > hi:
                rejects.append(f"{key}={value:.0f} 超出工作空间，已钳位到 {lo:.0f}-{hi:.0f}")
            out[key] = max(lo, min(hi, value))
        return out, rejects

    @staticmethod
    def can_start(cell) -> Tuple[bool, List[str]]:
        rejects: List[str] = []
        if cell.estop:
            rejects.append("急停未复位，禁止启动")
        if not cell.guard_closed:
            rejects.append("防护门未关闭，禁止机械臂动作")
        if cell.fault:
            rejects.append(f"分拣单元故障未清除：{cell.fault}")
        return (not rejects), rejects
