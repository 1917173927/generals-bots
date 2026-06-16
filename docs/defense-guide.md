# Generals Bots 期末验收汇报稿

用途：本文件用于 PPTMaster 生成期末验收 PPT。建议按 `---` 分隔为一页幻灯片，每页保留 3-5 个核心要点，演讲时补充“讲述提示”。

汇报主题：基于 JAX 的 Generals.io 双人对战模拟器与强化学习 Bot 实验框架。

---

## 1. 封面

### Generals Bots：基于 JAX 的策略博弈 AI 训练与可视化平台

- 项目类型：强化学习环境、规则 Bot、PPO 训练与评估、可视化 Demo
- 验收目标：展示系统可运行、模型可解释、实验有数据、工程有完整产物
- 最终成果：完成从环境建模、智能体实现、训练评估到人机对战展示的完整闭环

讲述提示：先强调这不是单纯游戏界面，而是一个可批量并行、可训练、可评估的 AI 实验平台。

---

## 2. 现场答辩评分细则对齐

### 评分细则

- Demo 功能完整运行，无明显 Bug：25 分
- 能清晰解释所用模型的原理：20 分
- 能回答关于技术细节的提问：20 分
- 实验对比有数据支撑，结论清晰：20 分
- 展示时间控制合理，表达清晰：15 分

### 本 PPT 对应安排

- Demo：第 14 页展示运行命令与现场流程
- 模型原理：第 6、8-10 页解释环境、智能体和 PPO 网络
- 技术细节：第 5-7 页解释架构、状态观测动作、JAX 高性能设计
- 实验数据：第 11-12 页展示 90%+ 胜率与 benchmark
- 时间控制：第 20 页给出演示节奏，第 21 页给出现场检查

讲述提示：开场说明汇报会严格围绕评分项展开，让老师知道每个得分点都有对应证据。

---

## 3. 项目背景与目标

### 为什么选择 Generals.io？

- Generals.io 是双人零和、部分可观测、离散动作的策略博弈
- 游戏包含战争迷雾、资源增长、路径扩张、攻防决策等复杂因素
- 非常适合验证规则 Bot、强化学习和大规模并行仿真能力

### 项目目标

- 构建可复现的 Generals.io 双人对战环境
- 支持 JAX `jit`、`vmap`、`lax.scan` 的高吞吐训练流程
- 实现规则型 Bot 与 PPO 神经网络策略
- 提供 GUI、人机对战、机器对战、评估脚本和 benchmark

---

## 4. 最终成果总览

### 已完成的核心产物

- `generals/core/`：JAX 游戏规则、地图生成、观测、动作、奖励和环境封装
- `generals/agents/`：Random、Expander、PPOPolicyAgent 等智能体
- `examples/_experimental/ppo/`：PPO 训练、行为克隆、策略评估、搜索增强等工具
- `generals/gui/`：pygame 可视化、人机交互、AI Top-K 动作预览
- `bench.py`：环境吞吐 benchmark
- `docs/`：中文手册、训练策略、开发记录和验收材料

### 最终可展示能力

- 一键运行基础环境 Demo
- 一键启动玩家对战训练好的 PPO 模型
- 一键观看两个 PPO 模型自动对战
- 用数据展示模型效果和环境性能

---

## 5. 系统架构

### 四层架构

1. 游戏核心层：`GameState`、规则推进、胜负判断、地图生成
2. 环境接口层：`GeneralsEnv`，提供 reset/step、reset pool、批量训练接口
3. 智能体层：规则 Bot、PPO 网络、checkpoint 加载和策略解释
4. 展示工具层：pygame GUI、PPO 人机对战、机器对战、benchmark 和文档

### 数据流

```text
地图生成 -> GameState -> Observation -> Agent/Policy -> Action -> game.step -> 新状态
```

讲述提示：这一页重点说明项目结构清晰，不是把所有逻辑写在一个脚本里。

---

## 6. 游戏环境建模

### 状态、观测、动作

- 完整状态 `GameState`：armies、ownership、generals、cities、mountains、time、winner、pool_idx
- 玩家观测 `Observation`：只能看到战争迷雾下可见区域和统计信息
- 动作格式：`[pass, row, col, direction, split]`
- 方向编码：`0=上`，`1=下`，`2=左`，`3=右`
- `split=1` 表示移动一半军队，`split=0` 表示移动除 1 个驻军外的全部军队

### 规则实现

- 支持城市、山地、将军、军队增长、领地归属和捕获胜利
- 支持合法动作 mask，避免 agent 主动选择非法移动
- 支持 generated 地图，可随机生成山地、城市和双方将军位置

---

## 7. 高性能 JAX 环境

### 为什么使用 JAX？

- 训练强化学习需要大量对局采样，普通 Python 循环吞吐有限
- JAX 可以把规则推进编译为高效数组计算
- `vmap` 支持大量环境并行
- `lax.scan` 支持批量 rollout，减少 Python 调度开销

### reset pool 设计

- 地图生成比单步推进更贵
- 项目预生成 reset pool，终局或截断时从 pool 中快速取新地图
- 训练时减少随机地图生成开销，提高批量 rollout 稳定性

---

## 8. 智能体设计

### 规则型 Bot

- `RandomAgent`：随机合法动作，作为最低基线
- `ExpanderAgent`：启发式扩张策略，优先占领可攻占格子和城市
- 其他 heuristic：用于训练数据、多样化对手和策略评估

### PPO 智能体

- `PPOPolicyAgent`：加载 `.eqx` checkpoint 进行推理
- 支持 `greedy` 和 `sample` 两种执行方式
- 支持 Top-K 候选动作解释：来源、目标、方向、概率、value

讲述提示：Random 用来证明流程能跑，Expander 用来作为更有意义的强基线。

---

## 9. PPO 模型原理

### PolicyValueNetwork

- 输入：9 通道棋盘张量
- 主干：4 层 `3x3` 卷积提取空间特征
- Policy head：输出 9 个动作平面
- Value head：估计当前局面的价值
- 合法动作 mask：非法 logits 加 `-1e9`，采样时几乎不会选到非法动作

### 9 个输入通道

- armies
- generals
- cities
- mountains
- neutral cells
- owned cells
- opponent cells
- fog cells
- structures in fog

---

## 10. 训练路线

### 从能玩到会赢

1. 行为克隆 warm start：先模仿 randomized Expander，学习基础扩张行为
2. PPO-vs-Expander fine-tune：在 generated 地图上继续强化学习
3. 多 epoch/minibatch 更新：提高样本利用率
4. 多 seed、双座位评估：减少随机性和出生点偏差
5. 后续探索：frozen checkpoint self-play、rollout search、conservative distillation

### 为什么不直接从零 PPO？

- 终局奖励稀疏，随机探索很难快速学到有效扩张
- 行为克隆能提供合理初始策略
- PPO 在 warm start 基础上继续优化，训练更稳定

---

## 11. 最终模型成果

### 当前展示模型

```text
generals-ppo-8x8-expander-gpu-v5.eqx
```

- 地图：8x8 generated
- 对手：randomized Expander
- 执行方式：sample policy
- 评估方式：独立 seed、player 0 / player 1 双座位测试
- 结论：sampled PPO 在 8x8 generated 地图上对 Expander 达到 90%+ 总胜率

### 评估结果

```text
seed 8501, policy_player=0: 1854/150/44, win rate = 90.53%
seed 8501, policy_player=1: 1846/168/34, win rate = 90.14%
seed 8503, policy_player=0: 1859/155/34, win rate = 90.77%
seed 8503, policy_player=1: 1856/160/32, win rate = 90.62%
```

### 对 Random 的 sanity check

```text
seed 8504, policy_player=0: 2039/2/7, win rate = 99.56%
```

讲述提示：这里要强调“不是只打赢 Random”，而是打赢更强的 Expander 基线。

---

## 12. Benchmark 性能结果

### 快速验证命令

```bash
uv run python bench.py --grid-size 8 --pool-size 128 --num-envs 16 --scan-steps 20 --reps 1 --single-steps 20
```

### 本机验证结果

```text
Pool generation: 128 states in 1.97s
Python loop (obs + agent + step): 606 steps/sec
Python loop (step only, pass): 26,781 steps/sec
env.step (pool auto-reset): 1,179,906 steps/sec
game_step (no reset, ceiling): 1,426,718 steps/sec
```

### 结论

- Python 循环适合调试
- 批量训练应使用 JAX 编译后的 `vmap + scan`
- reset pool 和 JAX 化规则显著提升 rollout 吞吐

---

## 13. 可视化与交互成果

### pygame GUI

- 展示双方领地、军队数量、将军、城市、山地和战争迷雾
- 支持左键选择源格，再点相邻目标格移动
- 支持 `S` 半兵移动、`P` 跳过、`R` 重开、`Q` 退出
- 支持玩家视角和机器对战观察

### AI 可解释性展示

- 右侧面板展示 PPO Top-K 候选动作
- 每个候选动作包含概率、方向、是否 split 和 value
- 棋盘上叠加箭头和候选标记，直观看到模型下一步想法

讲述提示：现场可以边操作边解释模型不是黑箱，至少能看到它当前最倾向的动作。

---

## 14. Demo 展示方案

### 推荐展示顺序

1. 基础环境运行：证明环境 reset/step 完整
2. 可视化对战：证明规则和地图展示清楚
3. PPO 人机对战：证明训练模型可加载、可交互、可解释
4. PPO 机器对战：证明两个模型可以自动对局

### 玩家对战 PPO

```bash
cd /Users/b./Code/generals-bots
./play-v5.command
```

### PPO 自动对战

```bash
cd /Users/b./Code/generals-bots
./watch-v5.command
```

现场提示：如果 GUI 机器环境不可用，就展示仓库内已有 GIF 或截图素材。

---

## 15. 工程质量与测试

### 工程化特征

- 使用 `uv` 管理依赖和 Python 版本
- 核心代码按 `core / agents / gui / remote / examples / tests / docs` 分层
- 训练、评估、可视化和 benchmark 命令可复现
- `.eqx` checkpoint 与源码分离，避免把大型实验产物提交到 Git

### 测试覆盖

- 游戏规则测试
- 地图生成测试
- reward 测试
- PPO agent 和 checkpoint 加载测试
- GUI 输入与渲染辅助函数测试
- benchmark 和训练脚本 smoke test

---

## 16. 项目亮点

### 技术亮点

- 将 Generals.io 抽象为 JAX 可编译强化学习环境
- 支持战争迷雾下的部分可观测策略学习
- 使用 reset pool 降低批量训练中的地图生成开销
- 从规则策略、行为克隆到 PPO fine-tune 形成完整训练路线
- 提供可解释的 PPO Top-K 动作预览，便于答辩和调试

### 成果亮点

- 有可运行 Demo
- 有训练好的 PPO checkpoint
- 有 90%+ vs Expander 的量化结果
- 有百万级 step/sec 的 JAX benchmark
- 有完整中文文档和展示材料

---

## 17. 局限与改进方向

### 当前局限

- 当前最佳结论主要限定在 8x8 generated 地图
- sample policy 对 Expander 达到 90%+，但 greedy 策略仍有提升空间
- 大地图和更强对手下仍需要更多训练与评估
- 真正稳定的 checkpoint league 自博弈还可以继续完善

### 后续计划

- 扩展到 16x16 或更接近真实 Generals.io 的地图规模
- 引入历史 checkpoint league，提升自博弈稳定性
- 结合 rollout search 或更强 value head 提升决策质量
- 接入远程 generals.io 客户端，进一步验证真实对战能力

---

## 18. 期末验收总结

### 一句话总结

本项目完成了一个从游戏规则、并行环境、智能体训练、模型评估到可视化展示的 Generals.io 强化学习实验框架。

### 三个核心贡献

1. 工程贡献：实现 JAX 化 Generals.io 环境，支持高吞吐并行 rollout
2. 算法贡献：实现规则 Bot、行为克隆、PPO 训练和 checkpoint 推理
3. 展示贡献：实现 pygame 人机对战、AI 候选动作预览和完整验收材料

### 最终验收结论

- 系统能运行
- 模型能加载
- 策略有数据支撑
- Demo 可现场展示
- 项目文档和代码结构完整

---

## 19. 现场答辩话术

```text
这个项目的核心不是只做一个游戏，而是把 Generals.io 建成了一个可用于强化学习研究的实验平台。
我完成了 JAX 化的游戏环境、规则型 Bot、PPO 训练与评估工具，并训练出可以在 8x8 generated 地图上以 sample 策略对 Expander 达到 90%+ 总胜率的模型。
现场 Demo 可以展示基础环境运行、pygame 可视化、人机对战和 AI Top-K 动作解释，说明项目从工程实现到实验验证形成了完整闭环。
```

---

## 20. 展示时间控制

### 8 分钟展示 + 2 分钟问答

```text
0:00-0:40  项目目标：JAX Generals.io 环境 + bot 实验框架
0:40-2:00  Demo：PPO 人机对战或 PPO 机器对战
2:00-3:20  系统架构：core / env / agents / gui / docs
3:20-4:40  模型原理：9 通道输入、卷积 Actor-Critic、动作 mask
4:40-5:50  训练路线：BC warm start、PPO-vs-Expander、双座位评估
5:50-7:10  实验结果：90%+ vs Expander、benchmark 吞吐
7:10-8:00  总结：成果、局限、后续方向
```

### 时间控制原则

- Demo 控制在 80 秒内，避免现场操作占用过多时间
- 每页只讲一个核心结论，不逐字读 PPT
- 数据页只强调关键数字：90%+ 胜率、百万级 step/sec
- 问答时优先回答评分相关问题：模型、技术细节、实验结论

---

## 21. 现场检查清单

- `uv sync --extra dev` 已完成
- `generals-ppo-8x8-expander-gpu-v5.eqx` 位于仓库根目录，或已设置 `MODEL_PATH`
- `./play-v5.command` 能启动人机对战
- `./watch-v5.command` 能启动 PPO 机器对战
- pygame 窗口能正常打开
- benchmark 快速命令已提前跑通过
- 准备好 GIF / 图片作为 GUI 失败时的备选展示

---

## 22. 评分项覆盖矩阵

### 100 分评分项逐项覆盖

| 评分项 | 分值 | PPT 证据页 | 现场展示证据 |
| --- | ---: | --- | --- |
| Demo 功能完整运行，无明显 Bug | 25 | 第 13-15 页 | `./play-v5.command`、`./watch-v5.command`、GUI 人机对战 |
| 清晰解释模型原理 | 20 | 第 6、8、9、10 页 | GameState、Observation、Action、PPO 卷积 Actor-Critic |
| 回答技术细节提问 | 20 | 第 5-7、15 页 | JAX、reset pool、合法动作 mask、工程分层、测试覆盖 |
| 实验对比有数据支撑，结论清晰 | 20 | 第 11-12 页 | 90%+ vs Expander、Random sanity check、benchmark steps/sec |
| 展示时间控制合理，表达清晰 | 15 | 第 19-21 页 | 8 分钟展示节奏、2 分钟问答、现场检查清单 |

### 答辩策略

- 先跑 Demo，再讲原理，最后用数据收束结论
- 每个技术点都落到代码模块或命令
- 每个实验结论都给出评估设置和数字
- 控制口径：当前成果限定在 8x8 generated 地图，不夸大到所有地图规模

---

## 23. PPTMaster 生成建议

### 页面风格

- 使用深色标题 + 浅色内容背景
- 架构页使用四层结构图
- 数据页使用表格或柱状图突出 90%+ 胜率
- Demo 页放启动命令和界面截图
- 总结页突出“三个贡献”和“最终验收结论”

### 建议配图

- `generals/assets/images/preview.png`
- `generals/assets/images/parallel_training_square.png`
- `generals/assets/gifs/parallel_training_process.gif`
- 人机对战窗口截图
- AI Top-K 候选动作预览截图
