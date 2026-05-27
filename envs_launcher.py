# ============================================
# 环境启动器 - 创建仿真或真实机器人装配环境
# ============================================
from envs.robot_sim_robotless import RobotSimRobotless
from envs.robot_sim_panda import RobotSimPanda
from envs.robot_real_example import RobotRealExample

# 开关：'sim' = 仿真模式（PyBullet），'real' = 真实机器人模式
t = 'sim'  # sim or real?

if t == 'sim':
    from envs.task_sim import TaskSim
    # 环境工厂函数：RLlib 调用这个函数来创建环境实例
    def env_creator(env_config):
        # 创建一个仿真装配环境
        environment = TaskSim(
            env_robot=RobotSimRobotless,      # 选哪个机器人模型（Robotless=搭接关节 / Panda=轴孔装配）
            self_collision_enabled=True,       # PyBullet 中是否开启自碰撞检测
            renders=True,                      # 是否显示可视化窗口；训练时设 False 加速，回放时设 True
            ft_noise=False,                    # 是否给力/力矩传感器加噪声（域随机化，提高 sim-to-real 迁移能力）
            pose_noise=False,                  # 是否给位置观测加噪声
            action_noise=False,                # 是否给动作加噪声
            physical_noise=False,              # 是否给物理参数（如摩擦系数）加噪声
            time_step=1/250,                   # 控制周期（秒），250Hz
            max_steps=200,                      # 每个 episode 最大步数
            step_limit=True,                   # 是否用 max_steps 限制 episode 长度
            action_dim=6,                      # 动作空间维度：3=仅平移，6=平移+旋转
            max_vel=0.01,                      # 最大线速度（m/s），每个轴方向
            max_rad=0.01,                      # 最大角速度（rad/s），每个轴方向
            ft_obs_only=False,                 # 观测是否只用力/力矩，不用位置信息
            limit_ft=False,                    # 是否限制力/力矩（超限时自动减速）
            max_ft=[1000, 1000, 2500, 100, 100, 100],  # 最大允许的力(N)和力矩(Nm)：Fx,Fy,Fz,Tx,Ty,Tz
            max_position_range=[2]*3,          # 观测空间中位置的取值范围（米）
            dist_threshold=0.005)              # 距离小于此值（5mm）就算装配成功
        return environment


if t == 'real':
    from envs.task_real import TaskReal
    def env_creator(env_config):
        environment = TaskReal(
            env_robot=RobotRealExample,        # 选真实的机器人控制接口
                               time_step=1/250,
                               max_steps=4000,
                               step_limit=True,
                               action_dim=6,
                               max_vel=0.01,
                               max_rad=0.01,
                               ft_obs_only=False,
                               limit_ft=False,
            max_ft=[667.233, 667.233, 2001.69, 67.7908, 67.7908, 67.7908],  # ATI-Delta 力传感器量程限制
                               max_position_range=[2] * 3,
                               dist_threshold=0.005)

        return environment


