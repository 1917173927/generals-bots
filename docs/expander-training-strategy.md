# Expander 对抗训练流程与策略

本文记录 8x8 generated 地图上训练 PPO 策略超过 Expander 的可复用流程。它不是一次性 devlog，而是后续继续训练、复现实验和判断模型强度时的操作指南。

当前已验证的 checkpoint：

```text
/tmp/generals-ppo-8x8-expander-gpu-v5.eqx
```

该 checkpoint 在 sampled policy 模式下，对 randomized Expander 的独立 2048 局评估超过 90% 总胜率。`.eqx` 是实验产物，应保存在 `/tmp` 或实验目录，不提交进 Git。

## 目标与判定标准

训练目标要用独立评估确认，不能只看训练过程中的 rollout 胜率。

推荐验收标准：

- 地图：8x8 generated
- mountain density：0.12-0.22
- cities：4-8
- minimum general distance：5
- max steps：500
- opponent：`expander`
- policy mode：`sample`
- 每个评估至少 2048 局
- 同一 checkpoint 至少测两个独立 seed
- 同一 seed 分别测 `--policy-player 0` 和 `--policy-player 1`
- 目标胜率按总局数计算，draw 不是 win

不要只报告 decisive win rate。decisive win rate 可以辅助分析，但如果 draw 很多，总胜率仍然不足。

## 环境准备

安装开发依赖和 CUDA extra：

```bash
uv sync --extra dev --extra cuda13
```

确认 JAX 使用 GPU：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

期望输出包含：

```text
gpu
[CudaDevice(id=0)]
```

如果 GPU 不可用，可以用 CPU 做 smoke test，但不适合长时间策略训练。

## 训练路线

最终有效路线分为三段：

1. 行为克隆 warm start：让策略先学会 Expander 风格的基础扩张动作。
2. PPO probe：从 BC checkpoint 对 Expander 进行短 PPO，确认强化学习方向能提升胜率。
3. GPU fine-tune：从较强 PPO checkpoint 继续训练，使用大 reset pool、低学习率和多 epoch/minibatch 更新。

这条路线的关键判断是：BC 只提供起点，真正超过 Expander 来自后续 PPO-vs-Expander fine-tune。

## 阶段一：行为克隆 warm start

从 soft Expander teacher 训练一个可用起点：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --map-generator generated \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 1000 \
  --lr 0.0007 \
  --model-path /tmp/generals-bc-8x8-soft.eqx \
  --seed 46
```

经验结果：

- 浅层 BC 会很弱，对 Expander 可能只有个位数胜率。
- 1000-2000 iterations 后，通常能得到足够好的 PPO 起点。
- BC 模型可能对 Random 很强，但这不代表它强于 Expander。

如果训练中断，可以续训：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/behavior_clone.py 128 \
  --grid-size 8 \
  --pool-size 4096 \
  --num-steps 32 \
  --num-iterations 1000 \
  --lr 0.0007 \
  --init-model-path /tmp/generals-bc-8x8-soft.eqx \
  --model-path /tmp/generals-bc-8x8-soft-v2.eqx
```

## 阶段二：PPO probe

先用较小训练量确认 PPO-vs-Expander 会提升，而不是破坏 BC 策略：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 256 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 4096 \
  --num-steps 64 \
  --num-iterations 300 \
  --num-epochs 4 \
  --minibatch-size 2048 \
  --lr 0.00005 \
  --truncation 500 \
  --opponent expander \
  --init-model-path /tmp/generals-bc-8x8-soft.eqx \
  --model-path /tmp/generals-ppo-8x8-expander-probe.eqx \
  --seed 9101
```

训练日志中的 episode 胜率只用于观察趋势。保存 checkpoint 后必须独立评估。

## 阶段三：GPU fine-tune

接近 90% 时，继续使用更大的 reset pool 和更低学习率，减少固定地图池过拟合：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/train.py 512 \
  --grid-size 8 \
  --map-generator generated \
  --mountain-density-min 0.12 \
  --mountain-density-max 0.22 \
  --num-cities-min 4 \
  --num-cities-max 8 \
  --min-generals-distance 5 \
  --pool-size 16384 \
  --num-steps 64 \
  --num-iterations 700 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 0.000005 \
  --truncation 500 \
  --opponent expander \
  --init-model-path /tmp/generals-ppo-8x8-expander-gpu-v4.eqx \
  --model-path /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --seed 9104
```

有效的 fine-tune 特征：

- rollout 内胜率大多在 88-93% 区间波动。
- draw rate 下降，平均终局时间缩短。
- 独立评估中的 loss 数继续下降。

如果 rollout 胜率长期停在 85-89%，继续堆同一 PPO 配方收益会降低，应考虑调整奖励、对手课程或引入更强 teacher。

## 评估命令

评估 player 0：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --opponent expander \
  --policy-mode sample \
  --policy-player 0 \
  --seed 8501
```

镜像评估 player 1：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/_experimental/ppo/evaluate_policy.py /tmp/generals-ppo-8x8-expander-gpu-v5.eqx \
  --num-games 2048 \
  --grid-size 8 \
  --map-generator generated \
  --max-steps 500 \
  --opponent expander \
  --policy-mode sample \
  --policy-player 1 \
  --seed 8501
```

最终 v5 结果：

```text
seed 8501, policy_player=0:
  wins/losses/draws = 1854/150/44
  win rate = 90.53%

seed 8501, policy_player=1:
  wins/losses/draws = 1846/168/34
  win rate = 90.14%

seed 8503, policy_player=0:
  wins/losses/draws = 1859/155/34
  win rate = 90.77%

seed 8503, policy_player=1:
  wins/losses/draws = 1856/160/32
  win rate = 90.62%
```

同 checkpoint 对 Random 的 sanity check：

```text
seed 8504, policy_player=0:
  wins/losses/draws = 2039/2/7
  win rate = 99.56%
```

## 策略层面的经验

Expander 的优势是快速占地、局部贪心扩张和稳定吞噬中立格。直接行为克隆会把这些优点学到，但也会继承它的单一路线。

PPO fine-tune 后有效提升主要来自：

- 学会在局部扩张和进攻之间更早切换。
- 减少无意义 draw，让强势局更快转化成终局。
- 在对手贪扩张时更频繁地抓住突破窗口。
- 通过 sampled policy 保留一定行动多样性，避免 greedy 在少数局面中重复选错固定动作。

当前结论应精确表述为：8x8 generated 地图、500 step、sampled policy 对 randomized Expander 超过 90% 总胜率。不要把这个结果直接外推到更大地图、真实 generals.io 人类对战或 greedy policy。

## 常见问题

### 为什么 sample 超过 90%，greedy 没有？

网络动作空间中很多局面存在多个近似可行动作。sample 模式保留探索性和随机化，能避免在少数战术局面中固定走到坏分支。当前 v5 的 greedy 总胜率仍低于 90%，所以报告时必须注明 `--policy-mode sample`。

### 为什么要镜像 `--policy-player`？

地图生成和双方出生位置可能带来微弱偏差。只测 player 0 可能高估或低估模型强度。`evaluate_policy.py --policy-player 1` 会让模型坐到另一侧，并按模型视角统计 wins/losses。

### 为什么训练内胜率不能作为验收？

训练内 episode 来自当前 reset pool 和当前 on-policy 采样，且样本量通常较小。最终验收必须用独立 seed、独立地图批次和足够局数。

### 继续提升到 greedy 90% 怎么做？

可以尝试：

- 降低 entropy 对 sample 分布的依赖，增加 greedy 友好的蒸馏阶段。
- 用 v5 自采样数据做 DAgger 式 teacher 修正。
- 增加 anti-Expander 专门 reward，例如更强的终局速度奖励和 loss 风险惩罚。
- 引入 checkpoint league，避免只针对一个 Expander 分布过拟合。
