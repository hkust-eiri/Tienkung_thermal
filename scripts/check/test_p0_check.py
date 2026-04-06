import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p0_check as pc


class P0CheckParsingTests(unittest.TestCase):
    def test_inspect_motor_status_interface_text(self) -> None:
        text = """
std_msgs/Header header
bodyctrl_msgs/msg/MotorStatus[] status
uint16 name
float32 pos
float32 speed
float32 current
float32 temperature
"""
        info = pc.inspect_motor_status_interface_text(text)
        self.assertTrue(info["status_is_array"])
        self.assertFalse(info["temperature_is_array"])
        self.assertTrue(info["name_is_integer_like"])

    def test_summarize_timing_payload_from_deltas(self) -> None:
        summary = pc.summarize_timing_payload({"stamp_deltas_sec": [0.0025, 0.0026, 0.0024]})
        self.assertAlmostEqual(summary["median_dt"], 0.0025)
        self.assertAlmostEqual(summary["avg_hz"], 1.0 / ((0.0025 + 0.0026 + 0.0024) / 3.0))


if __name__ == "__main__":
    unittest.main()
