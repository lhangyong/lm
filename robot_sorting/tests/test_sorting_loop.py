# -*- coding: utf-8 -*-
"""机械臂分拣闭环验收：NG 剔除、OK 放行、急停、监控模式、真空故障。

跑法：D:\\anconda\\envs\\yolov8-gpu\\python.exe -m unittest robot_sorting.tests.test_sorting_loop -v
"""
from __future__ import annotations

import unittest

from robot_sorting.controller import RobotController


def ng(name="bad.png"):
    return {"verdict": "NG", "image_name": name, "anomaly_score": 1.92,
            "severity": "Major", "defect_ratio": 0.12}


def ok(name="good.png"):
    return {"verdict": "OK", "image_name": name, "anomaly_score": 1.20,
            "severity": "None", "defect_ratio": 0.0}


class TestRobotSorting(unittest.TestCase):
    def run_ticks(self, ctrl, n=80, dt=0.1):
        for _ in range(n):
            ctrl.tick(dt)

    def test_ng_part_is_sorted(self):
        ctrl = RobotController()
        r = ctrl.ingest(ng())
        self.assertEqual(r["action"], "queued")
        self.assertEqual(r["queue_len"], 1)
        self.run_ticks(ctrl, 80)
        s = ctrl.status()
        self.assertEqual(s["stats"]["sorted"], 1, "NG 件应被剔除")
        self.assertEqual(s["reject_bin"], 1)
        self.assertEqual(s["queue_len"], 0)
        self.assertEqual(s["state"], "IDLE")
        self.assertIsNone(s["current"])

    def test_ok_part_passes(self):
        ctrl = RobotController()
        r = ctrl.ingest(ok())
        self.assertEqual(r["action"], "pass")
        self.run_ticks(ctrl, 10)
        s = ctrl.status()
        self.assertEqual(s["stats"]["passed"], 1)
        self.assertEqual(s["stats"]["sorted"], 0)
        self.assertEqual(s["queue_len"], 0)

    def test_estop_stops_and_reset_recovers(self):
        ctrl = RobotController()
        ctrl.ingest(ng())
        self.run_ticks(ctrl, 5)
        self.assertNotEqual(ctrl.status()["state"], "IDLE")
        ctrl.estop()
        self.run_ticks(ctrl, 10)
        s = ctrl.status()
        self.assertEqual(s["state"], "ESTOP")
        self.assertTrue(s["estop"])
        pose = s["arm"]
        self.run_ticks(ctrl, 10)
        self.assertEqual(ctrl.status()["arm"], pose, "急停后机械臂不应继续动")
        ctrl.reset()
        self.run_ticks(ctrl, 5)
        s2 = ctrl.status()
        self.assertEqual(s2["state"], "IDLE")
        self.assertEqual(s2["queue_len"], 0)
        self.assertFalse(s2["estop"])

    def test_monitor_mode_does_not_sort(self):
        ctrl = RobotController()
        ctrl.set_mode("monitor")
        r = ctrl.ingest(ng())
        self.assertEqual(r["action"], "monitor")
        self.run_ticks(ctrl, 20)
        s = ctrl.status()
        self.assertEqual(s["stats"]["ng"], 1)
        self.assertEqual(s["stats"]["sorted"], 0)
        self.assertEqual(s["queue_len"], 0)
        ctrl.set_mode("auto")
        ctrl.ingest(ng("bad2.png"))
        self.assertEqual(ctrl.status()["queue_len"], 1)

    def test_vacuum_loss_aborts(self):
        ctrl = RobotController()
        ctrl.ingest(ng())
        self.run_ticks(ctrl, 12)          # 已进入取料/抓取阶段
        ctrl.set_fault("vacuum_loss")
        self.run_ticks(ctrl, 30)
        s = ctrl.status()
        self.assertIn(s["state"], ("ABORT", "FAULT"))
        self.assertGreaterEqual(s["stats"]["manual"], 1)
        self.assertEqual(s["reject_bin"], 0, "真空丢失时不能把工件算作已剔除")
        self.assertTrue(s["fault_reason"], "应记录中止原因")

    def test_invalid_result_does_not_pass(self):
        ctrl = RobotController()
        r = ctrl.ingest({"verdict": "", "image_name": "broken.png"})
        self.assertEqual(r["action"], "invalid")
        s = ctrl.status()
        self.assertEqual(s["stats"]["passed"], 0, "检测结果缺失时不能当成 OK 放行")
        self.assertEqual(s["stats"]["manual"], 1)

    def test_multiple_ng_queue(self):
        ctrl = RobotController(max_queue=4)
        for i in range(4):
            self.assertEqual(ctrl.ingest(ng(f"bad{i}.png"))["action"], "queued")
        r = ctrl.ingest(ng("overflow.png"))
        self.assertEqual(r["action"], "buffer_full", "队列满时应转人工，不能无限接料")
        self.run_ticks(ctrl, 400)
        s = ctrl.status()
        self.assertEqual(s["stats"]["sorted"], 4, "队列里的 NG 应全部剔除")
        self.assertEqual(s["queue_len"], 0)
        self.assertEqual(s["state"], "IDLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
