# 面向 ASV 仿真的硬件在环实验平台 - 训练端

## 功能接口

| 入口 | 说明 |
| --- | --- |
| `src/train.py` | 模仿学习与强化学习微调 |
| `src/export_moving_target_bag.py` | bag → episode 导出 |
| `scripts/collect_*.ps1` | 数据采集 |

- 训练输入：CameraFrame、TaskEmbedding、EntityState、专家位移标签
- 训练输出：`vision.pt` / `policy.pt`

## 实现细节

- 模仿学习：用采集 episode 中的 专家二维位移 和 EnitityState 监督训练感知网络与策略网络
- 强化学习：在模仿学习权重之上，冻结视觉网络，用 PPO 在仿真闭环中微调策略
- 感知与决策网络与 Jetson 在线节点对齐，权重拷贝到 Jetson `models/` 后由 ROS 2 推理

## 测试环境

- Windows 11
- Python 3.13.5
- PyTorch 2.8.0 + CUDA 12.9
- NVIDIA GeForce RTX 5060 8GB
