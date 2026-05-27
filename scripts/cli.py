"""
robotic-assembly CLI 主入口
============================

统一的命令行接口，整合所有功能：
  - run:    运行环境 / 采集人工演示数据
  - demo:   自动装配演示（无需训练，比例控制）
  - grasp:  抓取提起演示（接近 -> 夹紧 -> 提升）
  - train:  使用 SB3 训练 RL 智能体
  - play:   加载训练好的模型进行回放/部署

用法:
    assembly run --help
    assembly demo --help
    assembly grasp --help
    assembly train --help
    assembly play --help
"""

from typing import Optional

import click


@click.group()
@click.version_option(version="2.0.0", prog_name="robotic-assembly")
def main():
    """机器人装配任务强化学习训练框架

    基于 PyTorch + Stable-Baselines3 + gymnasium + PyBullet
    """
    pass


@main.command()
@click.option(
    "--input-type", "-i",
    default="pbg",
    show_default=True,
    help="输入设备类型 (pbg=滑块, xbc=手柄, spm=SpaceMouse)",
)
@click.option(
    "--pos-scaling",
    default=10.0,
    show_default=True,
    help="位置缩放系数",
)
@click.option(
    "--orn-scaling",
    default=25.0,
    show_default=True,
    help="姿态缩放系数",
)
@click.option(
    "--action-space",
    default=6,
    type=click.IntRange(3, 6),
    show_default=True,
    help="动作空间自由度: 3（仅平移）或 6（平移+旋转）",
)
@click.option(
    "--save-demo",
    is_flag=True,
    default=False,
    help="是否保存演示数据",
)
@click.option(
    "--demo-path",
    default="human_demo_data/default",
    show_default=True,
    help="演示数据保存路径",
)
def run(
    input_type: str,
    pos_scaling: float,
    orn_scaling: float,
    action_space: int,
    save_demo: bool,
    demo_path: str,
):
    """
    运行仿真环境并采集人工演示数据

    默认启动 PyBullet 仿真窗口，可通过手柄 / SpaceMouse / 滑块控制机器人。
    """
    from scripts.run_env import run_env
    run_env(
        input_type=input_type,
        input_scaling=(pos_scaling, orn_scaling),
        action_space=action_space,
        save_demo=save_demo,
        demo_data_path=demo_path,
    )


@main.command()
def demo():
    """
    自动装配演示（比例控制，无需训练）

    工具从明显偏移的位置（X+6cm, Y+4cm）自动平滑运动到目标位置并对齐。
    按 Ctrl+C 退出。
    """
    from scripts.demo_auto_assembly import run_demo
    run_demo()


@main.command()
def grasp():
    """
    抓取提起演示（接近 -> 夹紧 -> 提升）

    展示末端执行器从偏移位置接近工件、模拟夹爪闭合并提起工件的
    完整流程。按 Ctrl+C 退出。
    """
    from scripts.demo_grasp_and_lift import run_demo
    run_demo()


@main.command()
@click.option(
    "--config", "-f",
    type=click.Path(exists=True),
    default=None,
    help="YAML 配置文件路径",
)
@click.option(
    "--algo", "-a",
    type=click.Choice(["SAC", "TD3", "PPO", "DDPG"]),
    default="SAC",
    show_default=True,
    help="RL 算法（推荐 SAC，连续控制最稳）",
)
@click.option(
    "--total-timesteps",
    default=1_000_000,
    show_default=True,
    help="总训练步数",
)
@click.option(
    "--name",
    default="assembly_experiment",
    show_default=True,
    help="实验名称",
)
@click.option(
    "--log-dir",
    default="./sb3_logs",
    show_default=True,
    help="日志保存目录",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="训练设备 (auto/cpu/cuda)",
)
@click.option(
    "--render",
    is_flag=True,
    default=False,
    help="训练时是否显示环境窗口（会大幅降低速度）",
)
def train(
    config: str,
    algo: str,
    total_timesteps: int,
    name: str,
    log_dir: str,
    device: str,
    render: bool,
):
    """
    使用 Stable-Baselines3 训练强化学习智能体

    支持 SAC、TD3、PPO、DDPG 等主流连续控制算法。
    推荐使用 SAC（Soft Actor-Critic），收敛快、稳定性好。
    """
    from scripts.train_agent import train_agent
    train_agent(
        config_file=config,
        algo=algo.lower(),
        total_timesteps=total_timesteps,
        experiment_name=name,
        log_dir=log_dir,
        device=device,
        render=render,
    )


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--steps",
    default=10000,
    show_default=True,
    help="回放步数",
)
@click.option(
    "--episodes",
    default=5,
    show_default=True,
    help="回放 episode 数",
)
@click.option(
    "--render/--no-render",
    default=True,
    help="是否显示环境窗口",
)
@click.option(
    "--algo",
    default="SAC",
    show_default=True,
    help="算法类型（需与训练时一致）",
)
@click.option(
    "--save-rollout",
    default=None,
    help="保存 rollout 数据到文件",
)
def play(
    model_path: str,
    steps: int,
    episodes: int,
    render: bool,
    algo: str,
    save_rollout: Optional[str],
):
    """
    加载训练好的模型进行回放/部署

    MODEL_PATH: 训练好的模型路径（.zip 文件）
    """
    from scripts.play_agent import play_agent
    play_agent(
        model_path=model_path,
        algo=algo,
        num_steps=steps,
        num_episodes=episodes,
        render=render,
        out_file=save_rollout,
    )


if __name__ == "__main__":
    main()
