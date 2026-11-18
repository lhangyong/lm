# -*- coding: utf-8 -*-
"""机械臂驱动抽象：仿真驱动 + Modbus TCP 骨架。

现场接 PLC/机械臂时，只需实现 write_pose 和 read_status，控制器不用改。
Modbus 寄存器示例：
  40001 X（×10）、40002 Y（×10）、40003 Z（×10）、40004 真空（0/1）
  30001 就绪、30002 真空确认、30003 报警字
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .cell import SortingCell


class ArmDriver:
    def write_pose(self, pose: Dict[str, float], vacuum: bool) -> None:
        raise NotImplementedError

    def read_status(self) -> Dict[str, object]:
        raise NotImplementedError


class SimulatorArmDriver(ArmDriver):
    def __init__(self, cell: SortingCell):
        self.cell = cell

    def write_pose(self, pose: Dict[str, float], vacuum: bool) -> None:
        self.cell.set_target(pose)
        self.cell.set_vacuum(vacuum)

    def read_status(self) -> Dict[str, object]:
        return self.cell.snapshot()


@dataclass
class ModbusArmConfig:
    host: str = "127.0.0.1"
    port: int = 502
    unit_id: int = 1
    timeout: float = 2.0


class ModbusArmDriver(ArmDriver):
    """真实 PLC 骨架，未做过现场连通测试；上现场前必须核对点表与字节序。"""

    REGISTER_MAP = {
        "x": 40001,
        "y": 40002,
        "z": 40003,
        "vacuum": 40004,
        "ready": 30001,
        "vacuum_ok": 30002,
        "alarm": 30003,
    }

    def __init__(self, config: Optional[ModbusArmConfig] = None, dry_run: bool = True):
        self.config = config or ModbusArmConfig()
        self.dry_run = dry_run
        self._client = None
        self._last: Dict[str, object] = {"ready": True, "vacuum_ok": True, "alarm": 0}

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from pymodbus.client import ModbusTcpClient  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("未安装 pymodbus，无法连接 PLC；请先 pip install pymodbus") from exc
        self._client = ModbusTcpClient(self.config.host, port=self.config.port,
                                       timeout=self.config.timeout)
        if not self._client.connect():
            raise RuntimeError(f"机械臂 PLC 连接失败 {self.config.host}:{self.config.port}")
        return self._client

    def write_pose(self, pose: Dict[str, float], vacuum: bool) -> None:
        self._last.update(pose)
        self._last["vacuum"] = bool(vacuum)
        if self.dry_run:
            return
        client = self._ensure_client()
        for key in ("x", "y", "z"):
            if key in pose:
                addr = self.REGISTER_MAP[key] - 40001
                client.write_register(addr, int(round(float(pose[key]) * 10)),
                                      slave=self.config.unit_id)
        client.write_register(self.REGISTER_MAP["vacuum"] - 40001,
                              1 if vacuum else 0, slave=self.config.unit_id)

    def read_status(self) -> Dict[str, object]:
        if self.dry_run:
            return dict(self._last)
        client = self._ensure_client()
        rr = client.read_input_registers(self.REGISTER_MAP["ready"] - 30001, 3,
                                         slave=self.config.unit_id)
        if rr.isError():
            raise RuntimeError("读取机械臂状态寄存器失败")
        return {
            "ready": bool(rr.registers[0]),
            "vacuum_ok": bool(rr.registers[1]),
            "alarm": int(rr.registers[2]),
        }
