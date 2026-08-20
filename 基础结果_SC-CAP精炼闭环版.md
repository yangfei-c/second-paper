# 基础结果与 SC-CAP 当前实现

> 更新日期：2026-08-20  
> 当前状态：已完成冻结感知模型接入、SC-CAP 四策略规划、4 首音乐逐曲反馈闭环、双轨迹记录与 Web 交互。尚未完成正式离线对比实验、参数冻结和用户实验，因此本文只报告已实现功能与已有基础结果，不把设计目标写成实验结论。

## 1. 当前项目组成

| 模块 | 作用 | 当前状态 |
|---|---|---|
| 外部 MSMMR | 提供冻结的 Text-VA checkpoint 和 MTG Music Catalog | 已接入，不在本项目中重新训练 |
| `SC_CAP/sc_cap/` | 唯一 canonical SC-CAP core、策略约束、离线接口和评价 | 已实现，原型与部署共用 |
| `SC_CAP/backend/` | 第一篇单曲链及第二篇 SC-CAP 的 FastAPI 接口 | 已实现 |
| `SC_CAP/frontend/` | 第一篇/第二篇统一 Vue Web 界面 | 已实现 |

原先独立的 `MusicMoodRegulationWeb/` 已合并进 `SC_CAP/`。工程只保留一个项目根目录、一个 SC-CAP core 和一份规划参数配置。

第二篇论文的实际处理链为：

```text
用户文本
  -> 冻结 Text-VA 预测初始状态
  -> 用户选择 Comfort / Calm / Energize / Maintain
  -> 策略可行域 + 累计前缀约束
  -> VA 近邻初筛 + 内容连续性排序
  -> 推荐并播放一首音乐
  -> 用户报告听后 felt VA
  -> 以 felt VA 作为当前状态重新规划
  -> 完成 4 首后计算双轨迹指标并保存会话
```

## 2. 冻结基础模型与目录

### 2.1 感知模型

- 音乐模型以 MERT 分段特征为输入，生成二维 Music-VA、183 维标签概率和 256 维音乐表示。
- 文本模型以 XLM-R base 为骨干，通过文本生成二维 Text-VA；当前 Web 系统加载 checkpoint 后将模型完全冻结，只执行推理。
- Text-VA、Music-VA 和用户听后 felt VA 表示不同对象：

$$
VA^{text}\neq VA^{music}\neq VA^{user}.
$$

文本与音乐的 256 维表示没有共享空间训练目标，因此当前系统不使用二者的余弦相似度做跨模态匹配。

### 2.2 Music Catalog

本地冻结目录已复核为 55,525 首唯一歌曲，其中 49,553 首具有可播放 URL。当前部署规划器实际使用：

| 字段 | 形状 | 用途 |
|---|---:|---|
| `song_id` | $(55{,}525,)$ | 唯一标识与去重 |
| `pred_va` | $(55{,}525,2)$ | 策略约束、前缀约束与 VA 排序 |
| `tag_embedding` | $(55{,}525,256)$ | 内容连续性与历史重复度 |
| `tag_prob` | $(55{,}525,183)$ | 已加载并保留，当前规划评分未使用 |

目录预测范围为：

$$
V\in[-0.8474,0.7571],\qquad
A\in[-0.9388,0.9264].
$$

部署加载时会检查数组形状、有限值和 $[-1,1]$ 范围，并只保留具有音频地址的歌曲作为 Web 候选。

## 3. 已有基础结果

以下结果来自本地 MSMMR 运行目录中的 `test_metrics.json`，本次更新没有重新训练模型。

### 3.1 MTG-Jamendo 标签

| 任务 | mAP | ROC-AUC |
|---|---:|---:|
| full183 overall | 0.1399 | 0.8147 |
| genre87 | 0.1887 | 0.8692 |
| instrument40 | 0.2009 | 0.7707 |
| mood56 | 0.1435 | 0.7688 |

full183 的分组结果为 genre 0.1821、instrument 0.1150、mood 0.0920，TagScore 为 0.1297。单组任务使用不同过滤子集和损失函数，不能据此直接声称拆分训练优于 full183。标签只用于内容层辅助排序，不承担 Calm、Comfort 等策略的情绪真值。

### 3.2 音乐 VA

| 训练方式 | 测试域 | CCC-V / CCC-A | 平均 CCC |
|---|---|---:|---:|
| DEAM | DEAM | 0.7406 / 0.7737 | 0.7571 |
| PMEmo | PMEmo | 0.7409 / 0.8580 | 0.7995 |
| Joint | DEAM | 0.7028 / 0.7644 | 0.7336 |
| Joint | PMEmo | 0.7400 / 0.8756 | 0.8078 |
| Joint | pooled | 0.7458 / 0.8351 | 0.7905 |

Joint checkpoint 的选模分数是两个域等权平均的 0.7707；0.7905 是样本合并后的分数，二者不能混用。各域 RMSE 约为 0.18–0.23，因此当前 $\delta$ 和 $\epsilon$ 仍是开发参数，不能解释为精确的心理阈值。

### 3.3 文本 VA

| 范围 | CCC-V / CCC-A | 平均 CCC |
|---|---:|---:|
| 全部测试集 | 0.8033 / 0.7416 | 0.7724 |
| English | 0.7521 / 0.8230 | 0.7875 |
| Chinese | 0.8136 / 0.4918 | 0.6527 |
| 语言等权选模 | — | 0.7201 |

当前主要弱点是中文 Arousal。首曲虽然只由 Text-VA 驱动，但用户初始 VA 会被单独记录，用于评估文本估计误差。

## 4. SC-CAP 规划器如何实现

### 4.1 状态

记文本预测初态为：

$$
q_0=(V_0,A_0).
$$

第 $t$ 步规划时：

- 规划状态 $c_1=q_0$；
- $t\ge2$ 时，$c_t=u_{t-1}$，即上一首播放后的用户 felt VA；
- `initial_text_va`：整个会话始终固定为 $q_0$；
- $H_{t-1}$：已经推荐的歌曲；
- $p_i=(V_i,A_i)$：候选歌曲的 Music-VA。

因此反馈会修正“当前状态”，但策略安全线和累计前缀仍锚定文本初态。

### 4.2 四种策略可行域

默认开发参数为：

$$
T=4,\ K=200,\ \delta_A=0.08,\ \delta_V=0.06,
$$

$$
\epsilon_V=0.20,\ \epsilon_A=0.22,\ \epsilon_P=0.26,\ 
r_{\mathrm{cong}}=0.30,\ \alpha=0.65.
$$

| 策略 | 当前候选必须满足的条件 |
|---|---|
| Energize | $A_i-A_{cur}\ge\delta_A$，且 $V_i\ge V_0-\epsilon_V$ |
| Calm | $A_{cur}-A_i\ge\delta_A$，且 $V_i\ge V_0-\epsilon_V$ |
| Maintain | 不增加单曲方向约束，由前缀均值约束整体稳定 |
| Comfort 第 1 首 | $\lVert p_i-q_0\rVert_2\le r_{\mathrm{cong}}$ |
| Comfort 第 2–4 首 | $V_i-V_{cur}\ge\delta_V$，且 $|A_i-A_{cur}|\le\epsilon_A$ |

Comfort 因此实现为“第一首承接当前情绪，后续逐步恢复 Valence”，而不是直接跳到高 Valence。
当 $c_t$ 靠近 VA 边界时，单步要求使用裁剪后的可达目标；若冻结目录中仍无满足项，则显式记为 infeasible。

### 4.3 累计前缀约束

策略首先定义经过边界裁剪的参考 Music-VA 轨迹 $r_j$：

$$
r_j=
\begin{cases}
\operatorname{clip}(q_0+(0,j\delta_A)), & \text{Energize},\\
\operatorname{clip}(q_0-(0,j\delta_A)), & \text{Calm},\\
q_0, & \text{Maintain},\\
\operatorname{clip}(q_0+((j-1)\delta_V,0)), & \text{Comfort}.
\end{cases}
$$

Comfort 因而满足 $r_1=q_0$。候选加入历史后，分别计算音乐前缀均值和参考前缀均值：

$$
\bar p_t=\frac{1}{t}\sum_{k=1}^{t}p_{m_k}.
$$

$$
\bar r_t=\frac{1}{t}\sum_{j=1}^{t}r_j.
$$

在未触及 VA 边界时，Energize/Calm 主轴参考为 $A_0\pm\frac{t+1}{2}\delta_A$，Comfort 的 Valence 参考为 $V_0+\frac{t-1}{2}\delta_V$。实现中直接对裁剪后的 $r_1,\ldots,r_t$ 求均值，以正确处理边界。

| 策略 | 前缀条件 |
|---|---|
| Energize | $\bar A_t\ge \bar r_{t,A}-\epsilon_P$，$\bar V_t\ge V_0-\epsilon_V$ |
| Calm | $\bar A_t\le \bar r_{t,A}+\epsilon_P$，$\bar V_t\ge V_0-\epsilon_V$ |
| Maintain | $\lVert\bar p_t-q_0\rVert_2\le\epsilon_P$ |
| Comfort | $\bar V_t\ge \bar r_{t,V}-\epsilon_P$，$|\bar A_t-A_0|\le\epsilon_A$ |

可行域控制当前一步，前缀约束控制整段累计方向。两者同时通过后，歌曲才进入排序阶段。

### 4.4 候选排序

部署版的选择顺序为：

1. 在全部 55,525 首目录中应用策略约束、前缀约束、已播去重和可播放过滤。
2. 计算候选到策略期望点的 VA 距离，保留最近的 $K$ 首。
3. 第一首直接选择最接近期望 VA 的候选。
4. 后续歌曲使用归一化 `tag_embedding` 计算内容分数。

策略期望点为：

$$
d_t=
\begin{cases}
q_{cur}+(0,\delta_A), & \text{Energize},\\
q_{cur}-(0,\delta_A), & \text{Calm},\\
q_0, & \text{Maintain 或 Comfort 第 1 首},\\
q_{cur}+(\delta_V,0), & \text{Comfort 恢复阶段}.
\end{cases}
$$

所有期望点均裁剪到 $[-1,1]^2$。

后续歌曲的软排序为：

$$
J(i)=
\alpha\left(1-\tilde s(z_i,z_{t-1})\right)
+(1-\alpha)\max_{j\le t-2}\tilde s(z_i,z_j),
$$

其中 $\tilde s=(\cos+1)/2\in[0,1]$。第一项只减少相邻内容突变；第二项只惩罚非相邻历史中的最大冗余，上一首不再被重复计入 diversity 项。没有非相邻历史时第二项为 0。

prototype 与 Web 均调用同一个 canonical core。无可行歌曲时抛出带候选计数的 `InfeasiblePlanError`，记录 `selection_status=infeasible`；系统不会静默放宽约束或返回伪 hard-feasible 结果。

## 5. 逐曲反馈闭环

### 5.1 首曲保持文本驱动

开始会话时，系统同时获得 Text-VA 与用户播放前报告的真实初态，但首曲接口只把 Text-VA 传给规划器：

$$
m_1=\operatorname{SC\!-\!CAP}(q_0,s,\varnothing).
$$

用户初始 VA 只进入真实轨迹，不参与首曲选择。接口返回字段 `first_recommendation_source=text_va_only`，避免把用户手动输入伪装成文本模型能力。

### 5.2 felt VA 反馈与重新规划

每首播放结束后，前端要求用户填写：

- 当前真实 Valence 和 Arousal；
- 1–5 分策略特定评分；
- 1–5 分音乐偏好；
- 播放时长与重复播放次数。

第 $t$ 首后的 felt VA 记为 $u_t$。下一首使用：

$$
m_{t+1}=\operatorname{SC\!-\!CAP}(u_t,s,H_t,q_0).
$$

前端只有在音频结束后才显示反馈表单；后端保存歌曲、规划约束、前缀状态、内容相似度、felt VA、评分、播放信息和时间戳。四首结束后再采集整体策略契合度、满意度、愉悦度、衔接平滑度和再次使用意愿，并将完整会话写入 `outputs/user_sessions.jsonl`。

### 5.3 双轨迹可视化

系统分别保存并展示：

$$
P=(q_0,p_1,\ldots,p_T),
\qquad
U=(u_0,u_1,\ldots,u_T).
$$

前端 SVG 图用不同颜色呈现文本起点、Music-VA 轨迹、用户初态和逐曲 felt-VA 轨迹，没有把 Music-VA 当作用户真实情绪。

## 6. 已实现评价

### 6.1 用户策略成功

| 策略 | 当前实现 |
|---|---|
| Energize | $A_T^{user}>A_0^{user}$，且 $V_T^{user}\ge V_0^{user}-\epsilon_V$ |
| Calm | $A_T^{user}<A_0^{user}$，且 $V_T^{user}\ge V_0^{user}-\epsilon_V$ |
| Maintain | $\max_t\lVert u_t-u_0\rVert_2\le\epsilon_P$ |
| Comfort | 不输出二元成功结论，以 strategy-specific rating 为主要记录 |

Comfort 没有被简化成“最终 Valence 上升即成功”，这与界面的“被理解、安慰或承接”问题保持一致。

### 6.2 双轨迹指标

已实现两个核心指标：

$$
\operatorname{DirectionAgreement}
=
\frac{1}{T}\sum_{t=1}^{T}
\mathbf 1\left[
\operatorname{sign}(\Delta p_t^{axis})
=
\operatorname{sign}(\Delta u_t^{axis})
\right],
$$

$$
\operatorname{RTD}
=
\sqrt{
\frac{1}{T+1}
\sum_{t=0}^{T}
\left\|
(p_t-p_0)-(u_t-u_0)
\right\|_2^2
}.
$$

此外记录用户轨迹相对真实初态的最大漂移和通用漂移告警。该告警当前统一采用欧氏距离阈值，不等同于四种策略各自的规划前缀条件。

## 7. Web 系统实现

后端使用 FastAPI，前端使用 Vue 3 + Vite，均位于 `SC_CAP/`。界面保留两条互不混合的实验链：

- 第一篇：冻结的旧 Text-VA 与 VA+Tag 单曲排序，可换歌并记录单曲偏好与调节效果；
- 第二篇：冻结的 MSMMR Text-VA、Music Catalog、SC-CAP 四步规划、逐曲 felt-VA 反馈和最终问卷。

第二篇主要接口为：

| 接口 | 功能 |
|---|---|
| `POST /api/second/sessions` | 文本推理、创建会话并生成首曲 |
| `POST /api/second/sessions/{id}/feedback` | 保存 felt VA，重新规划下一首 |
| `POST /api/second/sessions/{id}/finish` | 计算指标并保存完整会话 |

## 8. 验证结果与尚未完成部分

### 8.1 已验证

| 检查 | 结果 |
|---|---|
| Music Catalog 完整性 | 55,525 首均唯一，VA 范围合法 |
| 可播放目录交集 | 49,553 首 |
| SC-CAP 单元测试 | 12 项通过，包含 canonical/Web 一致性与 infeasible 记录检查 |
| SC-CAP synthetic demo | 可生成 4 步计划 |
| Vue/Vite 前端构建 | 通过 |
| Anaconda `music` 环境后端健康检查 | HTTP 200 |
| 真实 Text-VA + Catalog 首曲推荐 | HTTP 200，确认首曲来源为 `text_va_only` |

### 8.2 当前边界

- 正式离线 baseline 对比、消融、跨 seed 统计和参数敏感性尚未运行。
- 当前仅保留名称与实现一致的 `endpoint_knn`、`linear_waypoint` 和 `open_loop_sc_cap` 计划入口；closed-loop 由会话反馈链调用同一 core。正式 baseline/evaluation 批量接线不属于本阶段，尚未运行。
- 当前 `outputs/` 中没有正式用户会话，不能声称 SC-CAP 已改善真实情绪调节。
- $\delta_A$、$\delta_V$、$\epsilon_V$、$\epsilon_A$、$\epsilon_P$、$K$ 和 $\alpha$ 仍是开发默认值，必须只用 validation/pilot 冻结。
- prototype 与 Web 已统一为一个 canonical core、同一算法配置和相同的 hard-infeasible 语义。
- Prefix 已统一为“参考轨迹前缀均值”，不再使用 $\rho_t\delta$ 或 $t\delta$ 与音乐均值直接比较。
- 两个入口均引用现有的 `seed42_20260806-110723` Text-VA checkpoint；模型文件未修改。
- Anaconda `music` 环境可以运行 Web 后端，但当前未安装 `pytest`；单元测试是在系统 Python 环境中通过的。
- 当前 Direction Agreement 对 Maintain 和 Comfort 都使用 Valence 轴。Comfort 可作为恢复方向参考，Maintain 不应作为主要方向指标。

## 9. 当前可支持的研究表述

项目已经实现：

1. 将 Comfort、Calm、Energize 和 Maintain 表示为不同的策略可行域；
2. 用累计前缀均值约束多步音乐序列，而不是只检查单首歌曲；
3. 首曲由 Text-VA 独立驱动，后续根据用户逐曲 felt VA 重新规划；
4. 分离 Music-VA 与 User-VA，并记录规划轨迹、真实轨迹和策略特定评分；
5. 在同一 Web 系统中完成播放、反馈、重规划、可视化和会话保存。

当前不能表述为：

- SC-CAP 优于 waypoint、KNN 或其他 baseline；
- 四种策略已经产生显著的用户情绪改善；
- 当前阈值具有心理学通用意义；
- 内容连续性、VA 桥接、多步推荐或用户反馈本身是新的首创方法。

现阶段准确的总结是：

> SC-CAP 已形成唯一的四策略、参考前缀约束和逐曲 felt-VA 重规划核心；下一阶段仍需冻结评价定义并准备正式实验基础设施，之后才能形成效果性结论。
