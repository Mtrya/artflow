# Dataset plan

训练覆盖国画、西洋画、人物和世界知识四域，按
256p → 640p → 896p 渐进训练。逐源权重与阶段设置以
[`configs/pretrain.toml`](../configs/pretrain.toml) 为准；当前配方见
[pretrain_recipe.md](pretrain_recipe.md)，平台路径与存储操作见本机 `INSPIRE.md`。

## 数据域与训练契约

| 域 | 内容与来源 |
|---|---|
| D1 国画 | NPM-TW 与博物馆中国绘画；保留题款/书法信息及作品裁剪框 |
| D2 西洋画 | WikiArt、NGA、Rijksmuseum、Met 等；覆盖油画、印象派及其他绘画 |
| D3 人物 | human-recaption、people_supp、Pexels；补足全身、动作和解剖覆盖 |
| D4 世界 | vintage、relaion、PD12M、z-image、megalith、iNaturalist、Pexels；补充物体、建筑、自然与场景 |

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

## D1 与清洗契约

D1 使用发布集 `kaupane/chinese-painting-collection` 的 metadata 作为版本依据：
最终裁剪框、融入题款的中英文 caption，以及原始图像身份。
D1 manifest 准备已完成：发布 metadata 与长 caption 按 image_id 合并，
记录共享盘图像路径并解析 JSON 裁剪框。每行最多保留中、英、长三条 caption；
缺 caption 或图像文件的行单独计数。这是现有数据的来源记录，
不再保留一次性 manifest 构建流程。

来源清单保留 full/mounted/detail，剔除 rolled/junk；博物馆国画筛选使用
Chinese culture 元数据。规则清洗处理色卡、装裱边及无效画面；bbox 在预计算
时裁剪。OCR 对不可辨字和印文保留显式标记。

训练文本清洗处理 “The image shows” 等模板前缀。human-recaption 落地清单
已在抓取时筛过 RGB、水印与来源分数。z-image 采用用户确认的普通入池方式，
排除损坏的 1/9/16.parquet。跨域 phash 去重已于 9 月 3 日取消；不要将旧
组装清单中的这项待办重新执行。

## Caption 增补契约

API 请求缓存保留原始响应、模型/提示词版本和接受结果。长文本是图像行内的
补充 caption；被拒文本不进入 accepted 列表，图像可继续使用其已有 caption。

通用长 caption 选择器的默认目标为 256–511 / 512–895 / 896–1280 tokens，
份额 55% / 30% / 15%。特定来源可使用显式份额；Pexels 的四档包括 64–255，
目标为 44.9% / 27.5% / 16.6% / 11.0%，中英各半，prose/structured 按 hash
分配，seed 42。9 月 27 日的 Pexels 批次使用 `deepseek:deepseek-flash`，
完整 routing/models 配置可从 Git revision `6665738` 复现。

## 256p 初始池对账基线

2026-09-04 全量预计算完成时，共 **1,579,327 训练行 + 2,371 eval 行**。
该表用于识别基础池，后续 caption 更新与来源增补以实际阶段 manifest 为准。

| 源 | 训练输出行数 |
|---|---:|
| D1 | 90,223 |
| D2 WikiArt | 214,041 |
| D2 museum | 10,065 |
| D3 human | 118,393 |
| D3 people | 114,345 |
| D4 vintage | 194,625 |
| D4 z-image | 45,228 |
| D4 megalith | 5,504 |
| D4 iNaturalist | 2,823 |
| D4 PD12M | 162,307 |
| D4 relaion | 621,773 |
| **训练合计** | **1,579,327** |

当时含 eval 的输入/输出总数为 1,584,994 / 1,581,698，丢弃 3,296
（约 0.21%）。D3 human 的 1,697 条尺寸不足及 D1 的 440 条无效图像
是对账中的主要来源；记录丢弃类别有助于区分输入质量和处理故障。

## Pexels 增补

### D3 全身人物

人物照片偏面部/半身，增补全身、运动/舞蹈、地域服饰与劳作场景。
[`fetch_pexels.py`](../scripts/data/fetch_pexels.py) 支持 `term | pages`
逐词页预算，target 按落盘图片数计。79 个查询、144 次页请求实收 **6,008 张**；
原始图片数 40,472 → 46,480，metadata 行数 48,223 → 55,858。

Caption 批次发出 6,006 请求，接受 6,002 条长 caption，费用 $3.17；
并发 48 时约 19 requests/s。delta manifest 为 6,008 行。
训练池合并后为 **43,425 行**；发布集 `kaupane/pexels-people-captions`
记录为 43,414 行，两者口径分别保留。

这些数字记录增补时的数据来源，不代表当前分辨率的可用行数。
后续阶段按各自最终可用池重新计算权重；已完成的 256p 配方保持记录原样。

### D4 欠曝内容

七类目标按用户需求覆盖肢体解剖、建筑（含中国古建）、自然、城市、cosplay、
五官细节与审美场景。采集共 498 次页请求、33,781 条候选记录、
**29,574 张图片（约 11 GB）**：

| Family | 图片数 |
|---|---:|
| 肢体/身体部位 | 8,001 |
| 建筑/古建/桥塔 | 6,519 |
| 自然/动植物/天空 | 4,985 |
| 城市/霓虹 | 4,015 |
| Cosplay | 2,914 |
| 五官特写 | 2,135 |
| 审美场景 | 1,005 |

肢体和五官查询分别使用内容关键词过滤；共享 metadata 的采集任务串行写入。
Caption 批次 29,574 请求、零请求错误，接受 29,569 条（5 条重复句式被拒），
费用 **$15.66**；平均延迟 2.67s，并发 48 时 17–19 requests/s。
四个长度档接受数 13,282 / 8,077 / 4,906 / 3,304，中英 14,928 / 14,641，
median 297 tokens，maximum 1,267。

最终 manifest 为 29,574 行，image_id 唯一；抽查 400 行路径均存在。
五官类含 223 张动物-only 特写（该类 10.4%、全批 0.75%），用户决定保留，
作为动物五官词汇覆盖。

该批与 megalith 合并为 **d4-extra**。640p/896p 中 megalith 可用数分别为
5,361 / 1,152，合并池为 **34,935 / 30,726**。这些是当时的预计算
对账值；后续过滤后的最终行数用于重新计算混合权重与 bucket 规划。

## 分辨率独立的混合权重

每个阶段按最终训练工件的 eligible-row 数乘以来源倍数，再统一归一化：
`d1`、`d2-wikiart`、`d2-museum`、`d3-pexels` 为 1.8；
`d4-extra`、`d4-extra2`、`d4-extra3` 为 1.5；其余来源为 1.0。
拆分目录各用自己的行数，caption 条数不当作图像行数。

640p/896p 配比使用各自过滤后的实际 Arrow 行数。独立审计覆盖配置中的
50 个数据集：逐 shard 计数，校验 sidecar 的行数、prompt contract 与
source signature；所有行均有 caption。640p 为 19 个来源、1,366,193 行，
896p 为 16 个来源、649,772 行。两阶段权重与行数乘倍数后的归一化结果
在六位小数精度内一致；d1 占比分别为 9.922857% 与 19.362067%。

640p 删除所有 caption 均不足 16 raw tokens 的行，896p 阈值为 50；
这与训练时 caption curriculum 使用的 retained prompt length 不同。
已完成的 256p 配方权重保持不变，配置中的 256p 行数注释是当前工件盘点。
`dataset_info.json` 的 split 摘要可能保留上游行数，不能代替 Arrow 实际计数。
逐来源行数、自然占比与倍数见 `configs/pretrain.toml`，归一化汇总与审计
边界见 [pretrain_recipe.md](pretrain_recipe.md)。采样池变化仍需在重启时
显式处理 sampler state，并重新验证 bucket 规划对应的完整工作负载。

## 后续数据工作

概念测评在 200k 与 480k 各跑过一轮，concept benchmark 已退役；
450k / 570k 不再做中途概念测评与数据补充。600k 完成后做一次最终
benchmark（形式待定），决定 SFT go/no-go。每次新增数据都保留
采集/caption/预计算对账和完整配置变更记录；使用
[recipe 的严格阶段迁移流程](pretrain_recipe.md) 接入训练。
