"""
输入设备注册
============

注册所有支持的输入设备。采用懒注册，避免在模块导入时实例化设备。
"""

from human_demo.devices.base import InputDevice
from human_demo.devices.gui import PyBulletGUI
from human_demo.devices.xbox import XBoxController
from human_demo.devices.spacemouse import SpaceMouse

# 设备注册表（类引用，不实例化）
# 手动注册类名映射，避免在导入时实例化设备
REGISTRY: dict = {
    # codename -> class
    "pbg": PyBulletGUI,
    "xbc": XBoxController,
    "spm": SpaceMouse,
    # classname -> class
    "pybulletgui": PyBulletGUI,
    "xboxcontroller": XBoxController,
    "spacemouse": SpaceMouse,
}

