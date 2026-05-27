"""

pybulletgui.py - PyBullet GUI 滑块输入设备

在 PyBullet 的 GUI 界面中创建 6 个滑块（-1 到 1），
每个滑块控制一个自由度。适合在没有物理手柄时调试用。

"""

import pybullet as p

from devices.device import InputDevice


class PyBulletGUI(InputDevice):
    """PyBullet GUI 滑块控制器"""

    def __init__(self, *args, **kwargs):
        InputDevice.__init__(self, *args, **kwargs)

    @property
    def codename(self):
        return 'pbg'  # 设备代号

    def _connect(self):
        """连接：在 PyBullet GUI 中创建 6 个滑块"""
        assert p.isConnected(), 'PyBullet 未连接！'
        self._axis_ids = []
        keys = list(self._action.keys())  # ['x','y','z','rx','ry','rz']
        for i in range(6):
            self._axis_ids.append(
                p.addUserDebugParameter(
                    paramName=keys[i],
                    rangeMin=-1,
                    rangeMax=1,
                    startValue=0))
        self._is_connected = True

    def _disconnect(self):
        pass

    def _update(self):
        """读取滑块的值并应用缩放"""
        action = []
        for i in range(6):
            value = p.readUserDebugParameter(self._axis_ids[i])
            value *= self.pos_scaling if i < 3 else self.orn_scaling
            action.append(value)
        self.pose = action
