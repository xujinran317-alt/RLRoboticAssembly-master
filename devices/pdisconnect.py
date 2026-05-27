"""
pdisconnect.py - 断开模式的抽象基类

定义了设备的"断开"行为模板。
子类需要实现 _disconnect() 方法来执行实际的断开逻辑。
"""

from abc import ABC, abstractmethod

from devices.callbacks import handle, validate


class DisconnectPattern(ABC):
    """断开模式抽象类"""

    def __init__(self):
        self._on_disconnect = None  # 断开成功的回调函数
        if '_is_connected' not in self.__dict__:
            self._is_connected = False  # 连接状态标志

    def disconnect(self, *args, **kwargs):
        """
        公开的断开方法
        1. 调用子类实现的 _disconnect()
        2. 更新连接状态
        3. 执行回调函数
        """
        result = self._disconnect(*args, **kwargs)
        self._is_connected = result
        handle(self._on_disconnect, self)
        return result

    @abstractmethod
    def _disconnect(self, *args, **kwargs):
        """子类实现的断开逻辑"""
        pass

    @property
    def on_disconnect(self):
        """获取断开回调"""
        return self._on_disconnect

    @on_disconnect.setter
    def on_disconnect(self, func):
        """设置断开回调"""
        validate(func, allow_args=True, allow_return=True)
        self._on_disconnect = func