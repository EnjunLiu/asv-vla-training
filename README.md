# ASV VLA Training（Windows only）

PC 端采集、训练与闭环编排。**Jetson 在线推理源码不在本仓。**

| 端 | 路径 | 职责 |
| --- | --- | --- |
| 本仓 | `D:\asv-vla-training` | 数据集、`vision.pt` / `policy.pt` 训练、闭环编排 |
| Jetson | `jetson@192.168.137.100:/home/jetson/jetson_asv_ws` | ROS2 推理 / 闭环 |
| UE5 | `D:\asv-unreal-simulation` | 场景仿真 |

事实入口：`WORKSPACE_CONTEXT.md`（三端同步）。

## 当前验收（2026-08-25）

三场景 mean abs standoff error < 1 m，产物在 `experiments/chase_standoff_tight_1m/`。

| 场景 | mean abs |
| --- | --- |
| RED 4m | 0.76 m |
| BLUE 3m | 0.62 m |
| RED 3m | 0.87 m |

活动权重：`experiments/chase_standoff_tight_1m/{vision,policy}.pt`。

## 布局

| 路径 | 角色 |
| --- | --- |
| `src/train.py` | 训练入口（`vision` / `policy` / `all`） |
| `src/data.py` | episode 加载、teacher 标签 |
| `src/perception.py` | VisionModel（训练用） |
| `src/decision.py` | ActionPolicy + `build_entity_states`（训练用） |
| `src/export_moving_target_bag.py` | bag → episode 导出 |
| `scripts/run_closed_loop.py` | Windows 编排 UE + Jetson 闭环 |
| `scripts/plot_trace.py` / `plot_track_world_2x3.py` | 跟踪图 |
| `scripts/jetson_restart.sh` | 远程重启 Jetson 栈（scp 执行，不是 Jetson 源码） |
| `scripts/collect_*.ps1` | 采集 |
| `data/episodes/` | 本地 episode（不入库） |
| `experiments/chase_standoff_tight_1m/` | 当前验收实验目录 |

## 训练

```powershell
cd D:\asv-vla-training
python src/train.py all --out experiments/chase_standoff_tight_1m --device cuda
```

需要 Windows CUDA。Teacher 标签仅用于训练。单步位移上限 **0.55 m**。

## 部署权重到 Jetson

```powershell
scp experiments/chase_standoff_tight_1m/vision.pt jetson@192.168.137.100:/home/jetson/jetson_asv_ws/models/vision.pt
scp experiments/chase_standoff_tight_1m/policy.pt jetson@192.168.137.100:/home/jetson/jetson_asv_ws/models/policy.pt
```

在线节点源码只在 Jetson 仓库：`~/jetson_asv_ws/src/vla/`。

## 闭环验收

```powershell
python scripts/run_closed_loop_acceptance.py --out experiments/chase_standoff_tight_1m --runtime 180
```
