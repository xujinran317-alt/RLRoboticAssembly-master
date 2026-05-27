"""
XBox 手柄输入设备
=================

通过 PyGame 读取 XBox 手柄摇杆输入，映射为 6 自由度动作。
"""

import pygame
from pygame.locals import JOYBUTTONUP, JOYBUTTONDOWN, JOYAXISMOTION

from human_demo.devices.base import InputDevice


class XBoxController(InputDevice):
    """XBox 手柄控制器"""

    @property
    def codename(self) -> str:
        return "xbc"

    @property
    def clsname(self) -> str:
        return self.__class__.__name__.lower()

    def _connect(self):
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, "未找到手柄！"
        self._device = pygame.joystick.Joystick(0)
        self._device.init()
        self._z_directions = [1, 1]

    def _disconnect(self):
        pass

    def _update(self):
        action = []
        for event in pygame.event.get():
            if event.type in (JOYBUTTONUP, JOYBUTTONDOWN):
                buttons = [self._device.get_button(i) for i in range(15)]
                self._z_directions[0] *= -1 if buttons[8] else 1
                self._z_directions[1] *= -1 if buttons[9] else 1
            if event.type == JOYAXISMOTION:
                axes = [self._device.get_axis(i) for i in range(6)]
                x = axes[0]
                y = axes[1] * -1
                z = (axes[4] + 1) / 2 * self._z_directions[0]
                rx = axes[2]
                ry = axes[3] * -1
                rz = (axes[5] + 1) / 2 * self._z_directions[1]
                action = [x, y, z, rx, ry, rz]

        if not action:
            return

        for i in range(6):
            action[i] = 0.0 if abs(action[i]) < 0.001 else action[i]
            action[i] *= self.pos_scaling if i < 3 else self.orn_scaling
            action[i] = round(action[i], 5)

        self.pose = action
