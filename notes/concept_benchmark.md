# Concept benchmark：概念覆盖与 SFT 决策

2026-09-27 定稿。评估模型对常见概念的掌握程度，指导中途针对性数据生成，
并为 Stage 6 optional SFT 提供依据。构图、解剖、质感等优化目标由
[Stage 6 DMD+RL](redesign_plan.md) 与 [reward preflight](posttrain_preflight.md)
处理；这里按概念逐项判读。

运行节奏：hero **450k / 570k / 600k** 各全量一次；回补数据在下一运行点
检验效果。主要处置是中途针对性数据生成，Stage 6 SFT 作为兜底。

## 概念词表

- 外部受控词表提供全集：WordNet/ImageNet 名词、Getty AAT 艺术/建筑/博物馆
  子集、策展国画技法词表（工笔/没骨/写意等）和场景/环境词表。
- 世界先验频率用于尾部清洗：罕见概念降级或剔除。
- 用词表回扫训练 caption，获得训练频率，辅助区分数据缺口和容量/优化问题。
  Caption 词频与画面主体频率存在偏差，判读时保留这一限制。

## 三轴与 seed 组合

| 轴 | 内容 |
|---|---|
| 名词/实体 | 材质、姿态/动作、服装/配饰作为实体属性 |
| 技法/风格 | 包含颜色约束，例如水墨以黑白为主 |
| 场景/环境 | 空间与背景概念 |

Seed 是 **1–5 个概念的跨轴组合**；轴内避免重复组合，跨领域搭配允许，
例如国画技法与现代题材。领域自洽由轴结构约束。

规模为 **1000+ 概念、500 prompts、每 prompt 8 图**（4 zh + 4 en），
约 4000 图/轮，固定生成随机种子跨 checkpoint 复用。每个概念至少出现在
两个搭档不同的 seed 中，因此 seed 平均约含四个概念。

`deepseek-flash`（ZenMux）将 seed 展开成中英 prompt 和**逐概念 rubric**。
每个 seed 中的概念分别判档，再聚合到概念 verdict。

## 判读与处置

| 观察 | 处置 |
|---|---|
| 世界常见、训练稀薄、生成失败 | 数据缺口，优先中途针对性数据生成 |
| 训练高频、生成失败 | 疑似容量/优化问题，记录并人工复核主体频率 |
| 世界罕见 | 记录 |

Verdict 聚合该概念跨 seed、语言、prompt 和随机种子的表现。阈值在首轮用
真实分布校准；用户人工过 grid，拦截判官制造的假缺口。

## 词表源落地（2026-09-27 晚）

代码分工：`src/dataset/concept_vocab.py` 放可测的库逻辑（源加载、清洗、
频率洗尾、轴归属、seed 采样），`scripts/data/build_concept_vocab.py` 放
下载与编排流水线。原始 dump 不进 git。

已拉取并验证的源：

| 源 | 规模 | 质量检查 |
|---|---|---|
| WordNet（NLTK） | 82k 名词 synset | 实体轴 60.3k lemma → 形态过滤 37.6k → zipf≥2.5 共 21.0k；场景轴 4.8k→933；人物/职业 18.9k→4.8k。小写过滤可去绝大多数专名，仍需 NSFW blocklist |
| ImageNet-1k | 1,000 标签 | `data/vocab/imagenet1k.json`，直接可用 |
| Getty AAT explicit dump | 56,775 英文优选词（带 facet） | 134MB zip 走代理下载，已抽取为 `data/vocab/aat_en_pref.jsonl`；Objects 28.4k / Materials 4.5k / Activities 3.9k / Styles 5.7k；1,720 个 `<...>` 导航节点可滤除 |
| 策展国画清单 | 80 技法 + 56 实体/场景 | `data/vocab/curated/zh_*.jsonl`，中英对照 v0 |
| 世界先验频率 | wordfreq（en+zh，含 jieba） | 已验证；zipf 2.5 阈值下 Objects 保留 62%、Materials 60%、Activities 66%、Styles 仅 35% |

**v1 词表产物**（`data/vocab/concepts_v1.jsonl`，50k 概念，构建命令见
`data/vocab/concepts_v1.report.md`）：entity 45.7k / technique 3.2k /
scene 1.1k。构建中确认的三条结构性决定：

1. AAT 轴归属按**完整层级路径**（parentString）而非 facet：Processes and
   Techniques、Color 两个 hierarchy 直接进技法轴（process 1,947 +
   color 465），Settlements and Landscapes 与 Built Complexes and
   Districts 进场景轴（972），Abstract 的 Activities 其余部分丢弃。
2. WordNet noun.location 抽象污染无法靠物理实体祖先修复（time zone 与
   desert 同树），场景轴只保留自然地貌/聚居地祖先子树（120 条），
   建成环境场景交给 AAT。
3. NTriples 的非 ASCII 是字面 `\uXXXX` 转义，抽取时必须反转义。

已知残留噪声（留给 seed 采样与 LLM 展开稀释）：AAT Styles 仍有少量
民族名漏网（inuit/cara 级）；Activities→process 有实验室方法
（electron probe microanalysis）；AAT 为复数形式、与 WordNet 单数
未归一。

**训练频率回扫 v1**（2026-09-27 晚，`data/vocab/trainfreq_v1.jsonl`，
扫描器在 qb `$W/concept_vocab/scan_trainfreq.py`）：对 640p 全部 17 个
数据集的 1,304,540 行 caption 做词索引匹配（英文按词序列、末词容忍复数，
中文子串），行级去重计数。结果：48% 概念有命中（entity 49% / scene 45% /
technique 39%）；策展清单 135 条中 126 条有命中，零命中的 9 条
（荷叶皴/乱柴皴/逆锋/锥画沙/花盆底鞋/梅兰竹菊/江南水巷/徽州村落等）
正是预期的领域缺口候选。两条已知偏差，判读时必须保留：

- 多词概念的 wordfreq 先验是逐词平均，虚高（"range animal" zipf≥4 但
  作为短语几乎不出现）；零命中清单里的这类条目是词表噪声，不是数据缺口。
- 集合型概念（梅兰竹菊）在 caption 里通常拆开写，短语匹配漏计。

## 待定项

1. 判官与协议在词表、prompt 就绪后讨论。候选 SII Qwen3.8-27B thinking off
   的质量排序证据见 [preflight](posttrain_preflight.md)，概念/技法判别尚待
   实际使用确认。用户决定不做专项 judge 探针。初拟 1–3 档；artifact 监控
   由 hero_monitor 承担。
2. 训练频率分桶与世界先验阈值，用建词表后的真实分布确定。
3. 数据回补数量、进入阶段、权重，以及留给 Stage 6 SFT 的边界。
4. 与 hero_monitor 共用生成/判分基建的工程方案；prompt 池独立。

## 执行顺序

1. 整理词表、三轴标签与合并去重。
2. 建立世界先验频率与训练 caption 词频。
3. 生成领域均匀、每概念至少两个不同搭档的 500 个 seed。
4. 展开中英 prompt 与逐概念 rubric。
5. 450k 首次生成 4000 图、判分并输出 verdict 表。
