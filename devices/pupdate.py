"""
pupdate.py - 更新模式的抽象基类

定义了设备的"更新"行为模板。
子类需要实现 _update() 方法来执行实际的数据读取逻辑。
"""

from abc import ABC, abstractmethod

from devices.callbacks import handle, validate


class UpdatePattern(ABC):
    """更新模式抽象类"""

    def __init__(self):
        self._on_update = None  # 更新成功的回调函数
    def update(self, *args, **kwargs):
        """
        公开的更新方法
        1. 调用子类实现的 _update()
        2. 执行回调函数
        """
        result = self._update(*args, **kwargs)
        handle(self._on_update, self)
        return result

    @abstractmethod
    def _update(self, *args, **kwargs):
        """子类实现的数据更新逻辑（比如读取手柄输入）"""
        pass

    @property
    def on_update(self):
        """获取更新回调"""
        return self._on_update

    @on_update.setter
    def on_update(self, func):
        """设置更新回调"""
        validate(func, allow_args=True, allow_return=True)
        self._on_update = func

