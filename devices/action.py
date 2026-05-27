#! usr/bin/env python
"""

action.py - 动作数据结构

定义了一个 6 自由度动作的数据结构，包含：
  - 位置：x, y, z（平移）
  - 姿态：rx, ry, rz（旋转）

使用线程事件（Event）来通知新数据到达。
支持 setter/getter 属性访问。
"""

from threading import Event

# 六个自由度的名称常量
X = 'x'
Y = 'y'
Z = 'z'
RX = 'rx'
RY = 'ry'
RZ = 'rz'


class Action:
    """6自由度动作数据容器"""

    def __init__(self):
        """初始化：所有轴归零，创建事件对象"""
        self._action = {}
        for name in [X, Y, Z, RX, RY, RZ]:
            self._action[name] = 0.0
        self._event = Event()  # 用于线程间通知"有新数据"

    def await_new(self, timeout=None):
        """
        阻塞等待新数据到达
        timeout: 超时时间（秒），None 表示无限等待
        返回: True=有数据，False=超时
        """
        return self._event.wait(timeout)

    @property
    def has_new(self):
        """检查是否有新数据"""
        return self._event.isSet()

    # ---- 以下为各个轴的属性读写（x, y, z, rx, ry, rz）----
    # 每次 set 时触发事件，每次 get 时清除事件

    @property
    def x(self):
        self._event.clear()
        return self._action[X]

    @x.setter
    def x(self, value):
        self._event.set()
        self._action[X] = float(value)

    @property
    def y(self):
        self._event.clear()
        return self._action[Y]

    @y.setter
    def y(self, value):
        self._event.set()
        self._action[Y] = float(value)

    @property
    def z(self):
        self._event.clear()
        return self._action[Z]

    @z.setter
    def z(self, value):
        self._event.set()
        self._action[Z] = float(value)

    @property
    def rx(self):
        self._event.clear()
        return self._action[RX]

    @rx.setter
    def rx(self, value):
        self._event.set()
        self._action[RX] = float(value)

    @property
    def ry(self):
        self._event.clear()
        return self._action[RY]

    @ry.setter
    def ry(self, value):
        self._event.set()
        self._action[RY] = float(value)

    @property
    def rz(self):
        self._event.clear()
        return self._action[RZ]

    @rz.setter
    def rz(self, value):
        self._event.set()
        self._action[RZ] = float(value)

    @property
    def pose(self):
        """获取完整位姿 [x, y, z, rx, ry, rz]"""
        self._event.clear()
        return [self._action[name] for name in [X, Y, Z, RX, RY, RZ]]

    @pose.setter
    def pose(self, tup):
        """设置完整位姿"""
        for i, name in enumerate([X, Y, Z, RX, RY, RZ]):
            self._action[name] = float(tup[i])
        self._event.set()

    @property
    def pos(self):
        """获取位置 [x, y, z]"""
        self._event.clear()
        return [self._action[name] for name in [X, Y, Z]]

    @pos.setter
    def pos(self, tup):
        """设置位置"""
        for i, name in enumerate([X, Y, Z]):
            self._action[name] = float(tup[i])
        self._event.set()

    @property
    def orn(self):
        """获取姿态 [rx, ry, rz]"""
        self._event.clear()
        return [self._action[name] for name in [RX, RY, RZ]]

    @orn.setter
    def orn(self, tup):
        """设置姿态"""
        for i, name in enumerate([RX, RY, RZ]):
            self._action[name] = float(tup[i])
        self._event.set()

