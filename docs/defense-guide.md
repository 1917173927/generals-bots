# 现场答辩准备指南

本文按评分细则整理项目展示材料，目标是在有限时间内说明：Demo 能跑、模型原理讲得清、技术细节能回答、实验结论有数据支撑、时间控制合理。

## 1. Demo 功能完整运行，无明显 Bug

建议准备三层 Demo，按现场机器条件逐层展示。

### 必跑基础 Demo

```bash
uv run python examples/simple_example.py
```

展示点：

- `GeneralsEnv` 可以正常 reset/step。
- 双人对战流程完整：取 observation、agent 出 action、环境推进、终局或截断。
- 动作格式固定为 `[pass, row, col, direction, split]`，便于后续接规则 bot 和 PPO 策略。

### 可视化 Demo

```bash
uv run python examples/visualization_example.py
```

展示点：

- pygame 棋盘能展示 generals、cities、mountains、双方领地和军队数量。
- 可以直观看到 `RandomAgent` 与 `ExpanderAgent` 的差异：随机策略探索无序，扩张策略会优先占领周边可攻占格子。
- 若现场没有图形界面，可改用仓库内已生成素材：
  - `generals/assets/gifs/parallel_training_process.gif`
  - `generals/assets/gifs/parallel_training_square_tiled.gif`
  - `generals/assets/images/parallel_training_square.png`

### PPO 人机对战 Demo

使用本机已训练好的 `.eqx` checkpoint：

```text
/Users/b./Downloads/generals-ppo-8x8-expander-gpu-v5.eqx
```

```bash
uv run python examples/play_against_model.py /Users/b./Downloads/generals-ppo-8x8-expander-gpu-v5.eqx \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --human-player 0 \
  --fps 30 \
  --preview-top-k 3
```

展示点：

- 左键选择源格，再点相邻目标格；`S` 切换半兵移动，`P` 跳过。
- 右侧面板展示 PPO Top-K 候选动作、概率和 value，可作为“模型可解释性”的现场说明。
- 这个模型是 sampled PPO 在 8x8 generated 地图上对 Expander 达到 90%+ 胜率的版本，现场建议用 `--policy-mode sample` 展示。

## 2. 所用模型原理

本项目可以按“环境建模 + 基线策略 + PPO 策略”讲。

### 环境建模

Generals.io 被建模为双人零和、部分可观测、离散动作的序贯决策问题。

- 状态 `GameState`：完整隐藏状态，包括 armies、ownership、generals、cities、mountains、time、winner、pool_idx。
- 观测 `Observation`：每名玩家只能看到战争迷雾下的信息，包括己方领地、可见敌方、山、城、未知区域和双方统计量。
- 动作：固定 5 维整数 `[pass, row, col, direction, split]`，方向为上下左右，split 表示移动一半军队。
- 奖励：终局捕获对方 general 是核心目标；训练中额外使用 army ratio、land ratio、city capture 等 shaping 信号缓解稀疏奖励。

### PPO 策略网络

PPO checkpoint 使用 `PolicyValueNetwork`，是一个卷积 Actor-Critic 网络。

- 输入：9 个空间通道，包括 armies、generals、cities、mountains、neutral、owned、opponent、fog、structures in fog。
- 主干：4 层 `3x3` 卷积提取棋盘局部空间特征。
- Policy head：`1x1` 卷积输出 9 个动作平面：4 个全兵方向、4 个半兵方向、1 个 pass。
- Value head：`1x1` 卷积加全连接层输出当前局面的 value。
- 合法动作 mask：非法移动 logits 加 `-1e9`，保证采样和 greedy 选择不会主动选非法动作。

### 训练路线

当前较强路线是：

1. 行为克隆 warm start：先模仿 randomized Expander，学会基础扩张。
2. PPO-vs-Expander fine-tune：用 PPO 在 generated 地图上继续优化。
3. 多 seed、双座位评估：分别评估 player 0 和 player 1，避免出生点或先后手偏差。
4. 后续可接 frozen checkpoint self-play：用历史 checkpoint 作冻结对手，避免同步自博弈不稳定。

## 3. 技术细节高频问答

### 为什么用 JAX？

核心游戏逻辑用 JAX array 和不可变 `NamedTuple`，可以 `jit` 编译、`vmap` 并行和 `lax.scan` 扫描多步 rollout。强化学习需要大量模拟对局，JAX 的向量化能显著提升采样吞吐。

### 为什么要 reset pool？

地图生成比单步推进更贵，且在 JIT 内频繁随机生成复杂地图会增加开销。项目用预生成的 `GameState` pool 做 cheap auto-reset：游戏终局或达到 truncation 后，从 pool 中取下一局初始状态。

### 如何处理战争迷雾？

完整状态保存在 `GameState`，但 agent 只能拿到 `get_observation(state, player_idx)` 的结果。可见性来自己方占领格周围 `3x3` 范围；不可见区域用 fog/structures-in-fog 通道表示。

### 如何保证动作合法？

规则型 agent 使用 `compute_valid_move_mask` 找合法 `(row, col, direction)`。PPO 网络在 logits 上叠加 mask，非法动作概率被压到接近 0；pass 动作单独保留。

### 为什么有行为克隆？

直接 PPO 面对稀疏终局奖励时探索成本高。先模仿 Expander 可以让网络学会“扩张、占城、聚兵”这类基础行为，再用 PPO 从这个起点优化，训练更稳定。

### 为什么评估要测两个 player seat？

地图随机生成和行动顺序可能带来座位偏差。`evaluate_policy.py` 支持 `--policy-player 0/1`，同一 checkpoint 两边都测，结论更可信。

## 4. 实验对比与数据支撑

### 性能 benchmark

当前 `bench.py` 已按最新 `GeneralsEnv.reset(key) -> (pool, state)` 和 `env.step(state, actions, pool)` 接口修复，并支持小参数快速验证。

快速验证命令：

```bash
uv run python bench.py --grid-size 8 --pool-size 128 --num-envs 16 --scan-steps 20 --reps 1 --single-steps 20
```

本机一次验证结果：

```text
Pool generation: 128 states in 1.97s, 0.1 MB total
Python loop (obs + agent + step): 606 steps/sec
Python loop (step only, pass): 26,781 steps/sec
env.step (pool auto-reset): 1,179,906 steps/sec
game_step (no reset, ceiling): 1,426,718 steps/sec
```

结论：Python 单步循环适合调试；正式训练和评估应使用 `vmap + lax.scan` 批量推进。

### 策略质量数据

训练文档记录的当前最佳 sampled PPO checkpoint 是：

```text
/Users/b./Downloads/generals-ppo-8x8-expander-gpu-v5.eqx
```

评估设置：

- 地图：8x8 generated
- mountains：0.12-0.22
- cities：4-8
- min generals distance：5
- max steps：500
- opponent：randomized Expander
- 每组 2048 局
- sampled policy，分别测 player 0 和 player 1

最终结果：

```text
seed 8501, policy_player=0: wins/losses/draws = 1854/150/44, win rate = 90.53%
seed 8501, policy_player=1: wins/losses/draws = 1846/168/34, win rate = 90.14%
seed 8503, policy_player=0: wins/losses/draws = 1859/155/34, win rate = 90.77%
seed 8503, policy_player=1: wins/losses/draws = 1856/160/32, win rate = 90.62%
```

对 Random 的 sanity check：

```text
seed 8504, policy_player=0: wins/losses/draws = 2039/2/7, win rate = 99.56%
```

可讲结论：

- 只打赢 Random 不够，Random 太弱。
- Expander 是更强基线，因此 90%+ vs Expander 更能说明策略有效。
- 报告 total win rate，不只报 decisive win rate，因为 draw 也是未赢。

## 5. 展示时间控制

建议按 8 分钟准备，留 2 分钟回答问题。

```text
0:00-0:40  项目目标：JAX Generals.io 环境 + bot 实验框架
0:40-2:00  Demo：simple/visualization/PPO 人机对战三选一或组合展示
2:00-3:30  环境建模：GameState、Observation、Action、reward
3:30-5:00  PPO 模型：9 通道输入、卷积 actor-critic、mask、value
5:00-6:20  训练流程：BC warm start、PPO fine-tune、reset pool、vmap
6:20-7:20  实验结果：90%+ vs Expander、多 seed、双座位、benchmark
7:20-8:00  总结不足和改进：greedy 尚未达 90%、更大地图、自博弈 league
```

现场答辩表达模板：

```text
这个项目不是只做了一个游戏界面，而是把 Generals.io 抽象成了可批量并行的强化学习环境。
核心贡献有三点：第一，JAX 化的环境支持 vmap/scan 高吞吐 rollout；第二，提供规则型 agent、PPO 训练、行为克隆和评估工具；第三，用独立 seed 和双座位评估证明 sampled PPO 在 8x8 generated 地图上对 Expander 超过 90% 总胜率。
```

## 6. 现场前检查清单

- `uv sync --extra dev` 已完成。
- `uv run python -m compileall bench.py generals examples tests` 通过。
- `uv run python examples/simple_example.py` 能运行。
- 若需要 GUI，提前确认 pygame 窗口能打开。
- 若展示 PPO，人机对战 checkpoint 路径存在，且 `--grid-size` 与模型训练尺寸一致。
- 准备好 benchmark 快速命令，避免现场跑默认 24x24 大配置等待太久。
- 准备好说明“`.eqx` checkpoint 是实验产物，不提交进 Git，只保存在 `/tmp` 或实验目录”。
