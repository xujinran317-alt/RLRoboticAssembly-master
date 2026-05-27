"""

nothing.py - 空输入设备（随机/零动作）

当不需要人工输入时使用此设备，比如程序自动测试环境。
可以输出随机动作（用于探索）或零动作。

"""

import random

from devices.device import InputDevice


class Nothing(InputDevice):
    """空输入设备（程序控制模式）"""

    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)
        self.is_random = True  # True=输出随机动作，False=输出零动作

    @property
    def codename(self):
        return 'nth'  # 设备代号

    def _connect(self):
        """虚拟连接（不需要实际硬件）"""
        self._is_connected = True

    def _disconnect(self):
        """虚拟断开"""
        self._is_connected = False

    def _update(self):
        """生成随机动作或零动作"""
        action = []
        for i in range(6):
            a = random.uniform(-1.0, 1.0) if self.is_random else 0.0
            a *= self.pos_scaling if i < 3 else self.orn_scaling
            action.append(a)
        self.pose = action
