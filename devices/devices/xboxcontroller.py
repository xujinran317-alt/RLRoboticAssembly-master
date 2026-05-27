"""

xboxcontroller.py - XBox 手柄输入设备

通过 PyGame 读取 XBox 手柄的摇杆输入，映射为 6 自由度动作：
  - 左摇杆：X（左右）、Y（前后）
  - 右摇杆：RX、RY
  - 扳机键：Z+、RZ+（受按钮 8/9 控制方向）
  - 按钮 8（左肩键）：翻转 Z 方向
  - 按钮 9（右肩键）：翻转 RZ 方向

"""

import pygame
from pygame.locals import *

from devices.device import InputDevice


class XBoxController(InputDevice):
    """XBox 手柄控制器"""

    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)

    @property
    def codename(self):
        return 'xbc'  # 设备代号

    def _connect(self):
        """连接手柄（初始化 PyGame 手柄模块）"""
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, '未找到手柄！'
        self._device = pygame.joystick.Joystick(0)
        self._device.init()
        self._z_directions = [1, 1]  # Z 和 RZ 的方向（默认正）
        self._is_connected = True

    def _disconnect(self):
        pass

    def _update(self):
        """读取手柄最新输入并映射为动作"""
        action = []
        for event in pygame.event.get():
            if event.type == JOYBUTTONUP or event.type == JOYBUTTONDOWN:
                button_state = [self._device.get_button(i) for i in range(15)]
                # 按钮 8 翻转 Z 方向，按钮 9 翻转 RZ 方向
                self._z_directions[0] *= -1 if button_state[8] else 1
                self._z_directions[1] *= -1 if button_state[9] else 1
            if event.type == JOYAXISMOTION:
                action = [self._device.get_axis(i) for i in range(6)]
                # PyGame 的轴映射到手柄 6 个自由度
                x = action[0]           # 左摇杆左右
                y = action[1] * -1      # 左摇杆前后（取反）
                z = (action[4] + 1) / 2 * self._z_directions[0]  # 左扳机（映射到0~1）
                rx = action[2]          # 右摇杆左右
                ry = action[3] * -1     # 右摇杆前后（取反）
                rz = (action[5] + 1) / 2 * self._z_directions[1]  # 右扳机（映射到0~1）
                action = [x, y, z, rx, ry, rz]
        if len(action) == 0:
            return
        for i in range(6):
            # 高通滤波：忽略极小值（摇杆死区）
            action[i] = 0.0 if abs(action[i]) < 0.001 else action[i]
            # 应用缩放系数
            action[i] *= self.pos_scaling if i < 3 else self.orn_scaling
            action[i] = round(action[i], 5)
        self.pose = action  # 设置动作数据
