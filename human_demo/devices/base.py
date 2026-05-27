"""
输入设备基类
============

所有输入设备（手柄/鼠标/滑块等）的通用接口。
使用 Mixin 模式组合连接/断开/更新/动作功能。
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable


class InputDevice(ABC):
    """输入设备抽象基类"""

    def __init__(self, pos_scaling: float = 1.0, orn_scaling: float = 1.0):
        self._action = ActionData()
        self.pos_scaling = pos_scaling
        self.orn_scaling = orn_scaling
        self._device = None
        self._is_connected = False

    @property
    def clsname(self) -> str:
        """获取设备类名（小写）"""
        return self.__class__.__name__.lower()

    @property
    @abstractmethod
    def codename(self) -> str:
        """设备代号（如 xbc, spm, pbg）"""
        ...

    @property
    def action(self) -> "ActionData":
        return self._action

    @action.setter
    def action(self, value):
        self._action = value

    @property
    def pose(self):
        """获取当前位姿 [x, y, z, rx, ry, rz]"""
        return self._action.pose

    @pose.setter
    def pose(self, value):
        self._action.pose = value

    def start(self):
        """启动设备"""
        if not self._is_connected:
            self.connect()

    def connect(self):
        """连接设备"""
        self._connect()
        self._is_connected = True

    def disconnect(self):
        """断开设备"""
        self._disconnect()
        self._is_connected = False

    def update(self):
        """更新设备状态"""
        self._update()

    @abstractmethod
    def _connect(self):
        ...

    @abstractmethod
    def _disconnect(self):
        ...

    @abstractmethod
    def _update(self):
        ...


class ActionData:
    """
    6自由度动作数据容器
    使用 Event 实现线程间的"有新数据"通知
    """

    def __init__(self):
        from threading import Event

        self._action = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
        self._event = Event()

    def await_new(self, timeout: Optional[float] = None) -> bool:
        """等待新数据到达"""
        return self._event.wait(timeout)

    @property
    def has_new(self) -> bool:
        return self._event.is_set()

    def _set(self, name: str, value: float):
        self._event.set()
        self._action[name] = float(value)

    def _get(self, name: str) -> float:
        self._event.clear()
        return self._action[name]

    @property
    def x(self): return self._get("x")
    @x.setter
    def x(self, v): self._set("x", v)

    @property
    def y(self): return self._get("y")
    @y.setter
    def y(self, v): self._set("y", v)

    @property
    def z(self): return self._get("z")
    @z.setter
    def z(self, v): self._set("z", v)

    @property
    def rx(self): return self._get("rx")
    @rx.setter
    def rx(self, v): self._set("rx", v)

    @property
    def ry(self): return self._get("ry")
    @ry.setter
    def ry(self, v): self._set("ry", v)

    @property
    def rz(self): return self._get("rz")
    @rz.setter
    def rz(self, v): self._set("rz", v)

    @property
    def pose(self) -> list:
        self._event.clear()
        return [self._action[k] for k in ["x", "y", "z", "rx", "ry", "rz"]]

    @pose.setter
    def pose(self, values: list):
        for i, k in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            self._action[k] = float(values[i])
        self._event.set()

    @property
    def pos(self) -> list:
        self._event.clear()
        return [self._action[k] for k in ["x", "y", "z"]]

    @pos.setter
    def pos(self, values: list):
        for i, k in enumerate(["x", "y", "z"]):
            self._action[k] = float(values[i])
        self._event.set()

    @property
    def orn(self) -> list:
        self._event.clear()
        return [self._action[k] for k in ["rx", "ry", "rz"]]

    @orn.setter
    def orn(self, values: list):
        for i, k in enumerate(["rx", "ry", "rz"]):
            self._action[k] = float(values[i])
        self._event.set()
