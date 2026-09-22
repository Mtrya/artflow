# SII VLM judge probe（Stage 6 reward/judge 前置验证）

2026-09-22。用学院自部署 VLM（`sii:Qwen3.8-27B`，SII 内网）对 hero-256p run
现有 swanlab 采样网格图做小样本 judge 质量验证：复现性、thinking 开关、判别力、
每图打分时间成本。**结论：可用**，但必须 `enable_thinking: False`，且只能当
组内排序信号用，不能当绝对质量分。

## 结论（TL;DR）

| 问题 | 结论 |
|---|---|
| 能否承担 Stage 6 reward/judge | **能**。10 张样本 × 5 次重复、56 次主实验调用 + 50 次噪声地板调用，0 次传输失败；打分与本人肉眼判断同序 |
| 复现性是否合格 | **基本合格**。thinking off：10 张里 9 张两次完全一致（另一张差 1 个 rubric 分）；5 次重复下 4/10 全同、8/10 极差 ≤2 分；单图 score sd 平均 0.021（≈0.8 rubric 分），最坏一张单次跳 0.2 |
| thinking 是否值得开 | **不值得**。16 倍延迟、25 倍 token、1/20 截断丢样本，且复现性明显变差（1/9 全同、最大差 6 分），还会把 abstract 图的 anatomy 抬到 10 |
| 每张图打分成本 | thinking off：约 **0.27 s/图**（16 并发实测 3.7 calls/s）；768 图/RL 迭代 ≈ 3.5 min。thinking on(2048)：7.5–11.3 s/图，768 图 ≈ 1.6–2.4 h |

## 探针设置

- **模型/端点**：`sii:Qwen3.8-27B`（`CaptionClient` 的 `sii` provider，
  `SII_VLM_API_KEY` 从 GPFS `$W/secrets/sii_vlm.json` 读入环境，未进任何产物）。
- **judge**：仓库原样 `src/posttrain/rewards.py` 的 `VLMJudge` + `JUDGE_RUBRIC`
  （`judge-v1`，4 轴 anatomy/adherence/naturalness/aesthetics 各 0–10，
  `score = mean(axes)/10 ∈ [0,1]`）；`temperature=0.0`，`max_tokens=256`（judge 默认），
  thinking 开关走 `extra={"chat_template_kwargs": {"enable_thinking": ...}}`。
- **图**：hero-256p 最新一步（step 47500，网格间隔 2500；最新 ckpt 48000 生成于
  09-21 00:49，之后约 30 h 无新 ckpt / 无 48000 网格，该 run 可能已停）的 12 张网格图里
  取 10 张，每张切出左上角 cell
  （`make_grid` 2 列 + 2px padding；cell 顺序 zh_short/en_short/zh_long/en_long，
  对应 `panel_step_047500.json` 的 prompt 文本）。切块用仓库的 `encode_image`
  （JPEG q88，≤1024px 不缩放）走真实编码路径。
- **并发/重试**：`CaptionClient(concurrency=16)` + 内置 6 次指数退避（含 Retry-After），
  实测 106 次计费调用 0 次传输失败、1 次非传输错误（截断）。
- **缓存隔离**：每个 (condition, rep) 用独立 `cache_dir`，并断言所有主实验记录
  `cached=false` —— 两次重复真的是两次请求，不是缓存回放。另做等价性校验：
  10/10 用真 `VLMJudge.score` 重放同 cache 得到与探针子类一致的分数（顺带验证磁盘缓存生效）。
- **探针脚本**：`jobs/judge_probe_sii.py`（本地；`jobs/*` 不进 git），
  在 notebook `/root/judge_probe_0922/` 上跑，只 scp 了 `src/dataset/caption_client.py`、
  `src/posttrain/rewards.py` 和两个 `__init__.py` 的最小快照。

## 样本与 prompt

10 张 cell 全部来自 step 47500（最新可用采样）；prompt 为 panel JSON 原文。
本地留档：`data/judge_probe_sii/cells/`（切块图）、`summary_step47500.json`（含全文 prompt）。

| target | cell 尺寸 | prompt 长度 | prompt 摘要 |
|---|---|---|---|
| face_man_zh_short | 224×288 | 42 字 | 印象派油画胸像，中年男子四分之三侧面 |
| face_woman_zh_short | 224×288 | 39 字 | 女性胸像 |
| hands_book_zh_short | 224×288 | 34 字 | 双手持书 |
| hands_tea_zh_short | 256×256 | 45 字 | 双手端茶 |
| figures_baker_zh_short | 224×288 | 48 字 | 面包师全身像，横托长面包 |
| layout_boat_zh_short | 288×224 | 44 字 | 浅绛山水，空木船/石桥/群山 |
| style_flowers_zh_short | 256×256 | 40 字 | 花卉风格图 |
| architecture_church_zh_short | 288×224 | 44 字 | 水彩乡村石砌小教堂 |
| face_woman_zh_long | 224×288 | 710 字 | 长约束版（同一 scene） |
| layout_boat_en_short | 288×224 | 203 字 | 英文短版 |

**本人对 step 47500 网格的肉眼读图**（判断 judge 是否脱离常识的参照）：
face / hands / layout / architecture 的 cell 基本是近抽象的笔触纹理，脸和手都不可辨认；
`style_flowers` 勉强像印象派花丛；`figures_baker` 能看出人形与手中物。judge 的打分
排序与这个读图一致（0.05–0.875，均值 0.35），所以低分不是 judge 悲观，是这步模型确实还很糊。

## 主实验分数（off = thinking off/256，on = thinking on/2048）

每格 `anatomy/adherence/naturalness/aesthetics`（0–10）。

| target | off r1 | off r2 | on r1 | on r2 |
|---|---|---|---|---|
| architecture_church_zh_short | 5/1/6/4 | 5/1/6/4 | 10/1/2/3 | 10/1/5/4 |
| face_man_zh_short | 2/1/6/4 | 2/1/6/4 | 1/1/6/4 | 1/1/3/2 |
| face_woman_zh_long | 0/1/6/4 | 0/1/6/4 | 1/1/4/2 | 0/0/2/2 |
| face_woman_zh_short | 0/0/1/1 | 0/0/1/1 | 0/0/2/2 | 0/0/1/2 |
| figures_baker_zh_short | 4/5/7/6 | 4/5/7/6 | 3/5/7/6 | 3/5/7/6 |
| hands_book_zh_short | 0/0/6/4 | 0/0/6/4 | 1/1/2/3 | 1/1/6/4 |
| hands_tea_zh_short | 0/0/6/4 | 0/0/6/4 | 1/1/3/3 | 1/1/2/2 |
| layout_boat_en_short | 0/1/2/2 | 0/1/2/2 | 10/1/7/2 | 10/1/5/3 |
| layout_boat_zh_short | 5/0/7/4 | 5/0/6/4 | (截断, None) | 10/1/6/2 |
| style_flowers_zh_short | 9/9/8/9 | 9/9/8/9 | 9/6/8/6 | 10/9/8/8 |

## 复现性（同图同 prompt，独立 cache，同参数）

| 条件 | 次数 | 全同 | ≤1 分 | mean\|Δscore\| | max\|Δscore\| | mean latency | token/调用 |
|---|---|---|---|---|---|---|---|
| off (2 rep, 10 图) | 20 | 9/10 | 10/10 | 0.0025 | 0.025 | 2.07 s | 33.5 |
| off (5 rep, 10 图，全部新调用) | 50 | 4/10 | 8/10 极差≤2 分 | — | 0.20 | 1.64 s | ~35 |
| 对照集 (2 rep, 8 图) | 16 | 6/8 | 8/8 | 0.0063 | 0.025 | 2.00 s | 35 |
| on/2048 (2 rep, 10 图) | 20 | 1/9 | 3/9 | 0.078 | 0.15 | 33.3 s | 826 |

5 次重复（50 次全新调用）的分布细节：4 张图 5 次完全一致；`style_flowers_zh_short`
极差 1 分；`face_man`（anatomy 0↔2）、`figures_baker`（adherence 3↔5）、`hands_book`
（naturalness 4↔6）极差 2 分；`layout_boat_en`（naturalness 2↔6）4 分；
`face_woman_zh_short` 5 分（naturalness 1→6、aesthetics 1→4，单次偶发，score 0.05→0.25）。

**读法**：单图 score 的重复噪声 sd ≈ 0.021（最坏 0.08），跨图真实信号 sd ≈ 0.21
（10 张均值 0.35，范围 0.09–0.88）。噪声约占信号 10%，组内 G=16 的排序基本不受影响；
但两个候选分差 <0.05 时不能只看一次调用。因为 API 免费，**每图重复 k 次取均值**
（k=2–3）是最省事的降噪手段（k=2 时成本 0.55 s/图）。

## thinking 开关

- **延迟/token**：off 2.07 s / 33.5 completion tokens；on/2048 33.3 s（中位 30.9，
  p90 58.8，max 112.8）/ 826 tokens。**16× 延迟、25× token**。端点延迟由服务端排队主导
  （raw probe 同图同 prompt：1024 tokens 用 53.0 s，4096 tokens 只用 11.0 s），所以
  33 s 是量级而非准确值。
- **截断**：on/2048 下 1/20（5%）`finish_reason=length`、`content` 为空 →
  `VLMJudge.score` 返回 `None`（该样本 reward 直接缺失）。raw probe 显示 512 token
  必然截断、1024 起才出 JSON。
- **复现性**：on 模式 1/9 全同、3/9 ≤1 分、max 差 6 分（layout_boat_zh r1 截断之外的
  分布见上表）。raw probe 同一张图三次 on 调用（1024/2048/4096）分别给出
  `2/2/4/3`、`1/1/6/4`、`1/1/4/3` —— 同 session 内 temperature=0 也不稳定。
- **语义漂移**：on 模式把两张纯纹理的横构图（architecture_church、layout_boat_en）
  的 anatomy 从 0–5 抬到 10，thinking 里的"没有人体 → 无 anatomy 错误 → 10"把该轴
  从"结构正确性"变成"不可证伪"，与 off 模式的分数不可比。
- 计数注意：该端点即使输出 thinking 也把 `usage.completion_tokens_details.reasoning_tokens`
  报成 0（reasoning 按 completion token 计费），思维链在 message 的非标准 `reasoning` 字段里。

**建议：Stage 6 固定 `enable_thinking: False`（与 notes/posttrain_reward_budget.md 的接入示例一致）。**
若将来非开不可：max_tokens ≥4096、按 completion_tokens 记账、解析 `reasoning` 字段、
并显式处理 5–10% 的截断丢样本。

## 判别力对照（thinking off）

| 对照 | 构造 | 分数 |
|---|---|---|
| 真实照片 ×3 | pexels 真图 + 其 caption 当 prompt | 0.85（9/7/9/9）、0.875→0.90（9/8/9/9→9/9/9/9）、0.875（9/9/8/9） |
| 生成样本 ×10 | step 47500 网格 cell | 0.05–0.875，均值 0.35 |
| 纯灰图 | 128 灰 + face_man prompt | 0.0（0/0/0/0） |
| 均匀噪声 | `effect_noise` + face_man prompt | 0.075（0/0/2/1） |
| prompt 错配 ×3 | 图取自 A、prompt 用 B | figures_baker→face_woman：adherence 5→1，其余三轴 −2/−3/−3；另两组本已贴近 adherence 地板（1→0、0→0） |

结论：judge 能在 0–0.9 全量程区分"真照片 / 生成纹理 / 灰图"，且对 prompt 有反应
（虽然错配对照被地板效应削弱）。**但 `style_flowers` 这类风格/纹理 prompt 的生成样本
拿到 0.875，与真照片同分** —— 四轴等权平均会让 naturalness/aesthetics 把 adherence
拖上去。Stage 6 用法上要留意这点（见建议）。

## 成本（每张图打分时间）

| 条件 | 单调用延迟 | 10 图一轮 wall（并发 16） | 摊到每图 | 768 图/迭代 | 1152 图/迭代 |
|---|---|---|---|---|---|
| thinking off (256) | 均值 2.07 s / 中位 2.12 / p90 2.62 / max 2.94 | 2.5–3.0 s | **≈0.27 s**（3.7 calls/s） | ≈3.5 min | ≈5.3 min |
| thinking on (2048) | 均值 33.3 s / p90 58.8 / max 112.8 | 75–113 s | 7.5–11.3 s | 1.6–2.4 h | 2.4–3.6 h |

调用费 ≈0（自部署），成本就是共享端点的占用时间：**并发保持 ≤16**（本次 16 并发 0 次
限流/5xx，secrets 里记录的已验证并发是 8），并保留客户端退避。按此速率，
notes/posttrain_reward_budget.md 里 5.5k 的 probe 池约 25 min，G=16 主 run 每次迭代
约 3.5 min —— 相对训练迭代开销可以接受。k 次重复按 k 线性乘。

## CaptionClient / VLMJudge 观察（未改代码，仅记录）

1. **无功能性 bug**：provider 选择、`extra` 进请求体且进缓存键、零价 `ModelPricing`
   跳过 `/models` 询价、退避重试，均在真实端点上按文档行为工作（106 次计费调用，
   0 传输失败，1 非传输失败）。
2. `VLMJudge.score` 只返回 float 或 `None`，丢弃 `Response`（latency/usage/raw text）——
   这次探针为了记录成本才写了 `ProbeJudge` 子类（同一 prompt/参数/解析器，10/10 与
   真 `VLMJudge.score` 分数一致）。Stage 6 若要做成本监控，需要同样的小包装。
3. **`None` 的处理是 Stage 6 的待决项**：截断/解析失败/HTTP 错误都会返回 `None`，
   而下游对 `None` 没有定义行为（`RewardEnsemble.combine` 按权重相乘会 `TypeError`，
   缺键会 `KeyError`；`group_normalize` 求和也会 `TypeError`）。需要在 reward 侧显式
   规定策略（丢弃该样本 or 给 0.5 中性分）。
4. 截断类失败（`error="empty content"`，非 transient）**会写进磁盘缓存**，重跑时回放
   失败而不是重试；需要删缓存文件才会再请求。属于已知设计，但运维上要记得。
5. `Response.latency_s` 在重试时被重置，只记最后一次尝试的耗时；一次先失败后成功的请求
   会低估延迟（本次未触发，仅代码阅读发现）。

## 复现方式

```bash
# 1) 本地：把最小快照传到 notebook（jobs/ 与 GPFS 产物都不进 git）
tar czf /tmp/stage.tar.gz -C /tmp/judge_probe_stage repo   # src 子集 + jobs/judge_probe_sii.py
inspire notebook connection refresh inko-patrol --workspace CPU资源空间
inspire notebook scp inko-patrol --workspace CPU资源空间 /tmp/stage.tar.gz /root/judge_probe_0922_stage.tar.gz

# 2) notebook（$W=/inspire/.../ky26021/artflow；/root/judge_probe_0922/run.sh 负责注入
#    SII_VLM_API_KEY 与 PYTHONPATH，用 $W/venv-harvest/bin/python，该 venv 有 PIL+httpx）
./run.sh raw-probe  --out-dir /root/judge_probe_0922/out --thinking-tokens 1024,2048,4096
./run.sh controls   --out-dir /root/judge_probe_0922/out --reps 2 \
    --real-jsonl $W/data/caption_enrich/pexels/captions.jsonl \
    --real-thumbs $W/data/caption_enrich/pexels/thumbs --n-real 3
./run.sh run        --out-dir /root/judge_probe_0922/out --conditions off,on2048 --reps 2
./run.sh run        --out-dir /root/judge_probe_0922/out_reps5 --conditions off --reps 5 --no-equivalence-check
./run.sh report /root/judge_probe_0922/out/results_off.json /root/judge_probe_0922/out/results_on2048.json \
    /root/judge_probe_0922/out/results_controls.json --matched-results /root/judge_probe_0922/out/results_off.json
```

本地留档（`data/` 不进 git）：`data/judge_probe_sii/` 下 `results_off.json`、
`results_off_rep5.json`、`results_on2048.json`、`results_controls.json`、
`aggregate.json`、`summary_step47500.json`（含 10 张图的全文 prompt 与 rubric 原文）、
三份运行日志和 `cells/`（12 张切块图：10 张样本 + 灰图/噪声图）。

探针脚本本身有两份一次性本地验证脚本（也是本地留档，跑在假端点/合成网格上，
不依赖网络）：`test_judge_probe_geometry.py`（切块几何 + prompt 映射，验证 2×2
padding=2 的行列映射与 variant 顺序）、`test_judge_probe_plumbing.py`（本地假
OpenAI 兼容端点：每 rep 独立缓存、`extra` 进请求体、usage/latency 记录、
与真 `VLMJudge.score` 等价、controls 8 个 target 计数）。两者最后一次运行均通过。
注意 `results_controls.json` 的记录里没有 `prompt` 字段（controls 先于该字段加入
脚本时跑），prompt 见 `controls.log`。

## 建议与未做

**给 Stage 6 的建议**

1. reward judge 用 `enable_thinking: False`、`max_tokens=256`、`temperature=0.0`，
   并发 ≤16，沿用客户端退避与磁盘缓存。
2. 只把分数当**组内排序**信号（group normalize 本来就是这么用的），不要跨 run 比绝对值；
   分差接近 0.05 时每图重复 2–3 次取均值。
3. 显式定义 `None`（截断/解析失败）的 reward 策略，否则下游
   `RewardEnsemble.combine` / `group_normalize` 遇到 `None` 会抛异常。
4. 保留 held-out scorer：judge 的已知失效模式是**给风格化纹理高分**（style_flowers 0.875
   与真照片同分），这正是 reward hacking 的入口。若想更贴 adherence，可考虑对四轴加权
   而不是简单平均（当前 `VLMJudge.score` 是等权平均）。
5. 本次只验证了 256p/47.5k 的"糊样本"低端 + 3 张真照片高端。Stage 6 落在 640p/896p 收敛
   checkpoint 上时，样本质量分布不同，建议用同一脚本在那些 checkpoint 的网格图上再做一次
   （探针可直接指向对应 runs/<run>/samples）。

**未做**

- 没有在真实 reward 路径上打分（真实路径可能是 latent→内存解码的 JPEG，而非从 PNG 文件
  再编码；本次用仓库 `encode_image` 走文件路径，编码结果应一致但未对拍）。
- 没有构造真实 429/5xx 验证退避（只有单测覆盖），本次端点 0 传输失败。
- 语言/长度覆盖有限：8 张中文短 prompt + 1 张中文长 + 1 张英文短，未系统比较语言影响。
- 样本量小（10 张图），复现性数字的置信区间偏宽；若要作为 reward 的正式验收门槛，
  建议扩到 50–100 张再估一次噪声 sd。
