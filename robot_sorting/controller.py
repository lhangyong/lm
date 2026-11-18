# -*- coding: utf-8 -*-
"""机械臂分拣闭环控制器。

流程：检测结果 NG → 入剔除队列 → 传送到取料位 → 吸盘抓取 → 横移到废料箱
→ 放下 → 回原点。OK 件直接放行。急停、真空丢失、卡滞都会中止并留痕。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from .cell import (BIN_ABOVE, BIN_POS, HOME, PICK_ABOVE, PICK_POS, Part,
                   SortingCell)
from .driver import ArmDriver, SimulatorArmDriver
from .safety import ArmSafety

STATE_CN = {
    "IDLE": "待机",
    "CONVEY": "传送带送料",
    "APPROACH": "移到取料位",
    "DESCEND": "下降取料",
    "GRIP": "吸盘抓取",
    "LIFT": "提升",
    "TRANSFER": "横移料到废料箱",
    "PLACE": "下降放料",
    "RELEASE": "松开吸盘",
    "RETURN": "回原点",
    "ABORT": "异常回原点",
    "FAULT": "故障停机",
    "ESTOP": "急停",
}


class RobotController:
    def __init__(self, auto_start: bool = False, max_queue: int = 8,
                 cell: Optional[SortingCell] = None, driver: Optional[ArmDriver] = None):
        self.cell = cell or SortingCell()
        self.driver: ArmDriver = driver or SimulatorArmDriver(self.cell)
        self.safety = ArmSafety()
        self.max_queue = int(max_queue)
        self.mode = "auto"
        self.state = "IDLE"
        self.queue: Deque[Part] = deque()
        self.current: Optional[Part] = None
        self.last_action = "等待检测结果"
        self.fault_reason = ""
        self.events: Deque[Dict[str, str]] = deque(maxlen=30)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._timer = 0.0
        self._seq = 0
        self._cycle_timeout = 12.0
        if auto_start:
            self.start()

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="robot-sorting", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self.tick(0.1)
            except Exception:
                import logging
                logging.exception("robot sorting tick failed")

    # ------------------------------------------------------------------ 外部接口
    def ingest(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """把一次检测结果送进分拣单元；NG 自动入队，OK 直接放行。"""
        verdict = str(result.get("verdict", "")).strip().upper()
        image_name = str(result.get("image_name") or result.get("image_path") or "unknown")
        score = float(result.get("anomaly_score", 0.0) or 0.0)
        severity = str(result.get("severity", "None"))
        defect_ratio = float(result.get("defect_ratio", 0.0) or 0.0)

        with self._lock:
            if verdict == "OK":
                self.cell.stats.ok += 1
                self.cell.stats.passed += 1
                action, message = "pass", "OK，直接放行"
            elif verdict == "NG":
                self.cell.stats.ng += 1
                if self.mode != "auto":
                    action, message = "monitor", "NG，当前为监控模式，未驱动机械臂"
                elif self.cell.estop or self.cell.fault:
                    self.cell.stats.manual += 1
                    action, message = "blocked", "NG，但机械臂急停/故障中，转人工处理"
                elif len(self.queue) >= self.max_queue:
                    self.cell.stats.manual += 1
                    action, message = "buffer_full", "NG，剔除队列已满，转人工处理"
                else:
                    self._seq += 1
                    part = Part(part_id=f"NG-{self._seq:04d}", image_name=image_name,
                                score=score, severity=severity, defect_ratio=defect_ratio)
                    self.queue.append(part)
                    action, message = "queued", f"NG 已入剔除队列（第 {len(self.queue)} 件）"
            else:
                # 检测结果缺失或异常时不能当成 OK 放行，必须转人工
                self.cell.stats.manual += 1
                action, message = "invalid", "检测结果无效，转人工确认"
            self.last_action = message
            self._log("info" if action in ("pass", "queued") else "warning", message)
            return {
                "action": action,
                "message": message,
                "queue_len": len(self.queue),
                "mode": self.mode,
                "robot_state": self.state,
                "estimated_wait_s": round(len(self.queue) * 6.0, 1),
            }

    def set_mode(self, mode: str) -> str:
        mode = str(mode).lower()
        if mode not in ("auto", "monitor"):
            raise ValueError("mode 必须是 auto 或 monitor")
        with self._lock:
            self.mode = mode
            self._log("info", f"分拣模式切换为 {mode}")
        return mode

    def estop(self) -> None:
        with self._lock:
            self.cell.estop = True
            self.state = "ESTOP"
            self._timer = 0.0
            self._log("critical", "急停触发，机械臂停止动作")

    def reset(self) -> None:
        with self._lock:
            self.cell.reset()
            self.queue.clear()
            self.current = None
            self.state = "IDLE"
            self.fault_reason = ""
            self._timer = 0.0
            self._log("info", "分拣单元已复位（队列与故障已清除，计数保留）")

    def set_fault(self, name: Optional[str]) -> Optional[str]:
        with self._lock:
            self.cell.fault = name or None
            if name:
                self.cell.estop = False
                self.fault_reason = name
                self._log("warning", f"注入分拣故障：{name}")
            else:
                self.fault_reason = ""
                if self.state in ("FAULT", "ABORT"):
                    self.state = "IDLE"
                self._log("info", "分拣故障已清除")
            return self.cell.fault

    # ------------------------------------------------------------------ 主状态机
    def tick(self, dt: float = 0.1) -> None:
        with self._lock:
            self.cell.tick(dt)
            self._timer += max(0.0, dt)

            if self.cell.estop:
                self.state = "ESTOP"
                return
            if self.state == "ESTOP":
                return
            if self.state == "FAULT":
                return

            if self.state == "IDLE":
                self._start_next()
                return

            if self.state == "ABORT":
                if self.cell.at_target():
                    self.current = None
                    self.state = "FAULT" if self.cell.fault else "IDLE"
                    self._timer = 0.0
                return

            if self._timer > self._cycle_timeout:
                self._abort("单件分拣超时，机械臂未到达目标位")
                return

            handler = {
                "CONVEY": self._state_convey,
                "APPROACH": self._state_approach,
                "DESCEND": self._state_descend,
                "GRIP": self._state_grip,
                "LIFT": self._state_lift,
                "TRANSFER": self._state_transfer,
                "PLACE": self._state_place,
                "RELEASE": self._state_release,
                "RETURN": self._state_return,
            }.get(self.state)
            if handler:
                handler()

    def _start_next(self) -> None:
        if self.mode != "auto" or not self.queue:
            return
        ok, rejects = self.safety.can_start(self.cell)
        if not ok:
            self.last_action = "；".join(rejects)
            return
        self.current = self.queue.popleft()
        self.cell.part_present = True
        self._enter("CONVEY")
        self._log("info", f"开始分拣 {self.current.part_id}（{self.current.image_name}）")

    # ------------------------------------------------------------------ 各状态
    def _state_convey(self) -> None:
        self.cell.conveyor_pos = min(1.0, self._timer / 0.8)
        if self._timer >= 0.8:
            self._move(PICK_ABOVE, "APPROACH")

    def _state_approach(self) -> None:
        if self.cell.at_target() and self._timer > 0.15:
            self._move(PICK_POS, "DESCEND")

    def _state_descend(self) -> None:
        if self.cell.at_target():
            self._enter("GRIP")
            self.driver.write_pose(self.cell.pose, True)
            self._log("info", "吸盘已开启，等待真空确认")

    def _state_grip(self) -> None:
        if self._timer < 0.35:
            return
        if not self.cell.vacuum_ok:
            self._abort("真空未建立，工件未被吸住")
            return
        self._move(PICK_ABOVE, "LIFT")

    def _state_lift(self) -> None:
        if self.cell.at_target():
            self._move(BIN_ABOVE, "TRANSFER")

    def _state_transfer(self) -> None:
        if self.cell.at_target():
            self._move(BIN_POS, "PLACE")

    def _state_place(self) -> None:
        if self.cell.at_target():
            self._enter("RELEASE")
            self.driver.write_pose(self.cell.pose, False)
            self._log("info", "已到废料箱，松开吸盘")

    def _state_release(self) -> None:
        if self._timer < 0.3:
            return
        self.cell.stats.sorted += 1
        self.cell.reject_bin += 1
        self.cell.part_present = False
        self.last_action = f"已剔除 {self.current.part_id}" if self.current else "已剔除一件"
        self._move(HOME, "RETURN")

    def _state_return(self) -> None:
        if self.cell.at_target():
            self.current = None
            self.state = "IDLE"
            self._timer = 0.0
            self._log("info", "分拣完成，机械臂回到原点")

    # ------------------------------------------------------------------ 辅助
    def _move(self, target: Dict[str, float], state: str) -> None:
        safe, rejects = self.safety.clamp_pose(target)
        if rejects:
            self._log("warning", "；".join(rejects))
        self._enter(state)
        self.driver.write_pose(safe, bool(self.cell.pose.get("vacuum")))

    def _enter(self, state: str) -> None:
        self.state = state
        self._timer = 0.0

    def _abort(self, reason: str) -> None:
        self.cell.stats.faults += 1
        self.cell.stats.manual += 1
        self.fault_reason = reason
        self.cell.fault = self.cell.fault or "cycle_abort"
        self.cell.part_present = False
        self._log("critical", f"分拣中止：{reason}")
        self._enter("ABORT")
        self.driver.write_pose(HOME, False)
        self.cell.set_target(HOME)

    def _log(self, level: str, message: str) -> None:
        self.events.append({"ts": datetime.now().strftime("%H:%M:%S"),
                            "level": level, "message": message})

    def status(self) -> Dict[str, Any]:
        with self._lock:
            snap = self.cell.snapshot()
            return {
                "ok": True,
                "state": self.state,
                "state_cn": STATE_CN.get(self.state, self.state),
                "mode": self.mode,
                "queue": [asdict(p) for p in self.queue],
                "queue_len": len(self.queue),
                "current": asdict(self.current) if self.current else None,
                "last_action": self.last_action,
                "fault_reason": self.fault_reason,
                "events": list(self.events),
                **snap,
            }
