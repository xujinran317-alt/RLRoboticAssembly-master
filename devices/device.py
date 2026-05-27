#! usr/bin/env python
"""

device.py - 输入设备的抽象基类

定义了所有输入设备（手柄/鼠标/滑块等）的通用接口。
子类需要实现：
  - _connect()    连接设备
  - _disconnect() 断开设备
  - _update()     读取设备最新输入
  - codename      设备代号

"""

from abc import ABC

from devices.action import Action
from devices.pconnect import ConnectPattern
from devices.pdisconnect import DisconnectPattern
from devices.pupdate import UpdatePattern


class InputDevice(ConnectPattern, DisconnectPattern, UpdatePattern, Action, ABC):
    """输入设备的抽象基类，组合了连接/断开/更新/动作四种能力"""

    def __init__(self, pos_scaling=1.0, orn_scaling=1.0):
        """
        初始化输入设备
        pos_scaling: 位置输入的缩放系数
        orn_scaling: 姿态输入的缩放系数
        """
        ConnectPattern.__init__(self)
        DisconnectPattern.__init__(self)
        UpdatePattern.__init__(self)
        Action.__init__(self)
        self.action = Action()  # 动作数据对象
        self.pos_scaling = pos_scaling
        self.orn_scaling = orn_scaling
        self._device = None  # 设备的具体连接对象

    @property
    def clsname(self):
        """获取设备类名（小写）"""
        return self.__class__.__name__.lower()

    @property
    def codename(self):
        """获取设备代号（子类必须实现）"""
        raise NotImplementedError

    def start(self):
        """启动设备（如果未连接则自动连接）"""
        if not self._is_connected:
            self.connect()
