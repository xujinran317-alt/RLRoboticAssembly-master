"""
pconnect.py - 连接模式的抽象基类

定义了设备的"连接"行为模板。
子类需要实现 _connect() 方法来执行实际的连接逻辑。
"""

from abc import ABC, abstractmethod

from devices.callbacks import handle, validate


class ConnectPattern(ABC):
    """连接模式抽象类"""

    def __init__(self):
        self._on_connect = None  # 连接成功的回调函数
        if '_is_connected' not in self.__dict__:
            self._is_connected = False  # 连接状态标志
    @property
    def is_connected(self):
        """获取连接状态"""
        return self._is_connected

    def connect(self, *args, **kwargs):
        """
        公开的连接方法
        1. 调用子类实现的 _connect()
        2. 更新连接状态
        3. 执行回调函数
        """
        result = self._connect(*args, **kwargs)
        self._is_connected = result
        handle(self._on_connect, self)
        return result

    @abstractmethod
    def _connect(self, *args, **kwargs):
        """子类实现的连接逻辑"""
        pass

    @property
    def on_connect(self):
        """获取连接回调"""
        return self._on_connect

    @on_connect.setter
    def on_connect(self, func):
        """设置连接回调"""
        validate(func, allow_args=True, allow_return=True)
        self._on_connect = func

