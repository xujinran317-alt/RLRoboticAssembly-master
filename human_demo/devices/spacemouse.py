"""
SpaceMouse 3D 鼠标输入设备
===========================

通过 PyGame 读取 SpaceMouse 的 6 维输入。
适合给机器人提供实时人工示范。
"""

import pygame
from pygame.constants import JOYAXISMOTION

from human_demo.devices.base import InputDevice


class SpaceMouse(InputDevice):
    """SpaceMouse 3D 鼠标控制器"""

    @property
    def codename(self) -> str:
        return "spm"

    @property
    def clsname(self) -> str:
        return self.__class__.__name__.lower()

    def _connect(self):
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, "未找到摇杆设备！"
        self._device = pygame.joystick.Joystick(0)
        self._device.init()

    def _disconnect(self):
        pass

    def _update(self):
        action = []
        for event in pygame.event.get():
            if event.type == JOYAXISMOTION:
                action = [self._device.get_axis(i) for i in range(6)]

        if not action:
            return

        for i in range(6):
            action[i] *= self.pos_scaling if i < 3 else self.orn_scaling
        action[0] = -action[0]  # SpaceMouse X 方向取反
        self.pose = tuple(action)
