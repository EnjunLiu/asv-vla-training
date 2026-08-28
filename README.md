# 面向 ASV 仿真的硬件在环实验平台 - 训练端

Windows 上的数据采集与策略训练。Jetson 在线推理源码不在本仓。

| 目录 | 说明 |
| --- | --- |
| `src/` | 训练与 bag 导出代码 |
| `scripts/` | 采集 |
| `experiments/` | 本地权重与实验结果，不入库 |

## 功能接口

| 入口 | 说明 |
| --- | --- |
| `src/train.py` | `vision` / `policy` / `all` 训练 |
| `src/export_moving_target_bag.py` | bag → episode 导出 |
| `scripts/collect_*.ps1` | 采集 |

- 训练输入：episode 图像、任务嵌入、结构化实体、teacher 位移标签
- 训练输出：`vision.pt` / `policy.pt`

## 实现细节

- Teacher 标签只用于训练；在线策略输出船体坐标系二维期望位移
- 感知与决策网络与 Jetson 在线节点对齐，权重拷贝到 Jetson `models/` 后由 ROS 2 推理
- 单步位移上限 0.55 m
- 数据集、权重与实验日志不入库

## 测试环境

- Windows 11
- Python 3.10+
- CUDA
