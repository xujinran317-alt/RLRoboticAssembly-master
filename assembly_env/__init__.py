"""
assembly_env - 机器人装配环境（gymnasium 接口）
=============================================
提供仿真和真实机器人两种环境，兼容 Stable-Baselines3。
"""

# 延迟导入：只有真正使用时才会加载 pybullet
from assembly_env.base_task import AssemblyBaseTask
from assembly_env.sim_task import AssemblySimEnv

# AssemblySimEnv 需要 pybullet，不在 __init__ 时自动导入
# 用法：from assembly_env.sim_task import AssemblySimEnv

__all__ = ["AssemblyBaseTask", "AssemblySimEnv"]
