#! usr/bin/env python
"""

spacemouse.py - SpaceMouse 3D 鼠标输入设备

通过 PyGame 读取 SpaceMouse 3D 鼠标的 6 维输入。
SpaceMouse 是一个可以同时推/拉/旋转的 6 自由度输入设备，
很适合用来给机器人提供实时示范。

"""

import pygame
from pygame.constants import JOYAXISMOTION

from devices.device import InputDevice


class SpaceMouse(InputDevice):
    """SpaceMouse 3D 鼠标控制器"""

    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)

    @property
    def codename(self):
        return 'spm'  # 设备代号

    def _connect(self):
        """连接 SpaceMouse（通过 PyGame 手柄接口）"""
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, '未找到摇杆设备！'
        self._device = pygame.joystick.Joystick(0)
        self._device.init()
        self._is_connected = True

    def _disconnect(self, *args, **kwargs):
        pass

    def _update(self):
        """读取 SpaceMouse 的 6 轴输入"""
        action = []
        for event in pygame.event.get():
            if event.type == JOYAXISMOTION:
                action = [self._device.get_axis(i) for i in range(6)]
        if len(action) == 0:
            return
        for i in range(6):
            # 应用缩放系数
            action[i] *= self.pos_scaling if i < 3 else self.orn_scaling
        action[0] = -action[0]  # SpaceMouse 的 X 方向是反的，取反
        self.pose = tuple(action)
