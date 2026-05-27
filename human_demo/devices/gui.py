"""
PyBullet GUI 滑块输入设备
==========================

利用 PyBullet 自带的 GUI 滑块作为输入设备。
每个自由度对应一个滑块，适合调试和精细控制。

改进说明：
  1. 不再每次读取后归零滑块（改为手动拖动控制）
  2. 若 resetSimulation() 导致滑块失效，自动重建所有滑块
"""

import pybullet as p

from human_demo.devices.base import InputDevice

LABELS = ["X", "Y", "Z", "RX", "RY", "RZ"]


class PyBulletGUI(InputDevice):
    """PyBullet GUI 滑块控制器"""

    @property
    def codename(self) -> str:
        return "pbg"

    @property
    def clsname(self) -> str:
        return self.__class__.__name__.lower()

    # ------------------------------------------------------------------
    # 内部辅助：创建 / 重建全部滑块
    # ------------------------------------------------------------------

    def _create_sliders(self):
        """创建（或重建）6 个滑块，初始值均为 0。"""
        self._params = {}
        for label in LABELS:
            # addUserDebugParameter(name, rangeMin, rangeMax, startValue)
            self._params[label] = p.addUserDebugParameter(label, -1, 1, 0)

    # ------------------------------------------------------------------
    # InputDevice 接口
    # ------------------------------------------------------------------

    def _connect(self):
        self._create_sliders()

    def _disconnect(self):
        pass

    def _update(self):
        action = []

        for label in LABELS:
            try:
                val = p.readUserDebugParameter(self._params[label])
            except Exception:
                # resetSimulation() 会使旧滑块 ID 失效，捕获后重建
                self._create_sliders()
                try:
                    val = p.readUserDebugParameter(self._params[label])
                except Exception:
                    val = 0.0
            action.append(val)

        # 分别应用位置 / 姿态缩放
        for i in range(6):
            action[i] *= self.pos_scaling if i < 3 else self.orn_scaling

        self.pose = action
