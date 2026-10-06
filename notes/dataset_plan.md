# Dataset plan

训练覆盖国画、西洋画、人物和世界知识四域，按
256p → 640p → 896p 渐进训练。逐源权重与阶段设置以
[`configs/pretrain.toml`](../configs/pretrain.toml) 为准；训练契约见
[pretrain_recipe.md](pretrain_recipe.md)，平台路径与存储操作见本机 `INSPIRE.md`。

## 数据域与行契约

| 域 | 内容与来源 |
|---|---|
| D1 国画 | NPM-TW 与博物馆中国绘画；保留题款/书法信息及作品裁剪框 |
| D2 西洋画 | WikiArt、NGA、Rijksmuseum、Met 等；覆盖油画、印象派及其他绘画 |
| D3 人物 | human-recaption、people_supp、Pexels；覆盖面部、全身、动作和解剖 |
| D4 世界 | vintage、relaion、PD12M、z-image、megalith、iNaturalist、Pexels；覆盖物体、建筑、自然与场景 |

训练先按源权重抽图像行，再在该行的 caption 列表中选择文本；多条 caption
共享同一图像行。预计算保存 Qwen-Image VAE latents、清洗后的 caption 文本
及抽样所需长度元数据。Qwen3-0.6B tokenization/encoding 在线执行，支持随训练
推进的长度课程。长度统计使用训练所用 tokenizer 和同一文本契约。

每个分辨率使用经过尺寸、裁剪、解码和 caption 检查的可用池。记录实际输出行数、
各类丢弃原因及 eligible-row 数；原始 metadata 行数、图片数、发布行数和训练行数
分别对账。源权重归一化后才是抽样概率，训练曝光以实际 draw 计数。

保留来源、作者/作品信息和许可证元数据；受限来源独立成 mix 条目，便于构建
许可兼容的变体。来源许可须按具体记录和发布用途核对。大规模采集、caption 和
预计算均在计算平台执行。

## 来源与清洗

D1 的版本依据是发布集 `kaupane/chinese-painting-collection` 的 metadata，
包括最终裁剪框、融入题款的中英文 caption 及原始图像身份。训练 manifest
按 image_id 合并发布 metadata 与长 caption，记录共享盘图像路径及解析后的
裁剪框。每行最多保留中、英、长三条 caption；缺 caption 或图像文件的行单独计数。

来源清单保留 full/mounted/detail，剔除 rolled/junk；博物馆国画筛选使用
Chinese culture 元数据。规则清洗处理色卡、装裱边及无效画面；bbox 在预计算
时裁剪。OCR 对不可辨字和印文保留显式标记。

训练文本清洗处理 “The image shows” 等模板前缀。human-recaption 抓取条件
包括 RGB、水印与来源分数。z-image 按普通来源入池，排除损坏的
1/9/16.parquet。训练池不做跨域 pHash 合并。

D3 Pexels 覆盖全身、运动/舞蹈、地域服饰与劳作场景；发布身份为
`kaupane/pexels-people-captions`。D4 Pexels 补充肢体解剖、建筑（含中国古建）、
自然、城市、cosplay、五官细节与审美场景。五官类包括动物特写。
`d4-extra` 包含该 Pexels 批次与 megalith；`d4-extra2`、`d4-extra3` 是配置中
独立加权的补充池。各阶段可用数量以最终 Arrow 工件为准。

## Caption 生产

API 请求缓存保留原始响应、模型/提示词版本和接受结果。长文本是图像行内的
补充 caption；被拒文本不进入 accepted 列表，图像可继续使用其已有 caption。
具体请求配置随生产工件记录，不能从一次测量的费用或速度推断后续批次。

通用长 caption 选择器的默认目标为 256–511 / 512–895 / 896–1280 tokens，
份额 55% / 30% / 15%。特定来源可使用显式份额。Pexels caption 配方的四档
包括 64–255，目标为 44.9% / 27.5% / 16.6% / 11.0%，中英各半，
prose/structured 按 hash 分配，seed 42。

维护工具见 [scripts/README.md](../scripts/README.md)。Pexels 抓取支持
`term | pages` 逐词页预算，target 按落盘图片数计；共享 metadata 的采集任务
串行写入。Caption 生产使用选择、生成、冻结、附加四步，冻结时按训练文本契约
检查质量与长度。采集/caption/预计算各自保存输入输出对账。

## 分辨率独立的混合权重

640p/896p 按最终训练工件的 eligible-row 数乘以来源倍数，再统一归一化：

- `d1`、`d2-wikiart`、`d2-museum`、`d3-pexels`：1.8。
- `d4-extra`、`d4-extra2`、`d4-extra3`：1.5。
- 其余来源：1.0。

拆分目录各用自己的行数，caption 条数不当作图像行数。已完成的 256p 阶段
保留 checkpoint 中的配方权重，其行数注释用于盘点现有工件。

| 阶段 | 配置来源数 | Arrow 训练行数 | 加权行质量 | d1 权重 |
|---|---:|---:|---:|---:|
| 256p | 15 | 1,699,951 | — | 8.093704% |
| 640p | 19 | 1,366,193 | 1,645,691.4 | 9.922857% |
| 896p | 16 | 649,772 | 838,639.8 | 19.362067% |

独立审计覆盖配置中的 50 个数据集：逐 shard 计数，校验 sidecar 的行数、
prompt contract 与 source signature；所有行均有 caption。640p/896p 权重
与行数乘倍数后的归一化结果在六位小数精度内一致。

640p 的构建规则删除所有 caption 均不足 16 raw tokens 的行，896p 阈值为 50。
这些阈值与训练 caption curriculum 使用的 retained prompt length 不同；
存活行内的 caption 变体仍可被选中。本次审计验证工件计数、sidecar 与配比，
没有对全部原始 caption 重新分词以复验过滤过程。

`dataset_info.json` 的 split 摘要可能描述上游数据，不能代替 Arrow 实际计数。
逐来源行数、自然占比与倍数见配置，审计边界见
[pretrain_recipe.md](pretrain_recipe.md)。480k 是阶段内重启点；迁移工件按过滤后的池重建
sampler cycle，清空队列并从 curriculum position 0.8 开始新抽样。模型、EMA、
optimizer、scheduler 和训练 RNG 在全部 16 个 rank 上精确恢复。训练遥测与
实际生成结果用于评估该配比的运行表现。

## 训练评估与后训练数据

预训练使用各分辨率的固定可用池，通过固定双语面板和 loss probe 监测。
600k 完成后做最终 capability benchmark（形式待定），决定是否需要针对性 SFT。
新增数据须有采集/caption/预计算对账、来源元数据和完整配置；接入训练遵守
[recipe 的显式迁移契约](pretrain_recipe.md)。
