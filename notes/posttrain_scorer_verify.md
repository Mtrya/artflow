# Stage 6 本地打分器实重验证（aesthetic + HPSv2）

2026-09-23。用真实权重在 inko-patrol（CPU）上跑仓库 `ClipAestheticScorer` /
`HPSV2Scorer`，样本 = step 47500 网格 cell ×10（糊生成图）+ pexels 真图 ×3。
**结论：两个打分器均可用**，输出区间正确、real/gen 可分；HPS 区分度干净，
美学头对"笔触纹理"类糊图区分度弱（已知特性，靠 ensemble + held-out 兜底）。

## 环境修复（已入库）

- `ClipAestheticScorer` 原实现假设单层 Linear head；实际权重是
  christophschuhmann/improved-aesthetic-predictor 的 MLP
  （768→1024→128→64→16→1，dropout 在 eval 下无效），且 state dict 带
  `layers.` 前缀（LightningModule 封装）。已改为对应 MLP 结构并剥前缀加载。
- notebook 侧：`venv-precompute-gpu` 装 open_clip_torch 3.3（--no-deps）
  + torchvision 0.29.0+cu130（--no-deps，nexus 源与 torch 2.14 ABI 兼容）+
  timm/ftfy/regex/sentencepiece/protobuf。

## 权重（GPFS，复用 $W/hf_cache 做 HF_HOME）

| 用途 | 位置 |
|---|---|
| 美学头（3.7MB） | `$W/reward_weights/aesthetic_head_l14.pth`（HF `camenduru/improved-aesthetic-predictor` 的 `sac+logos+ava1-l14-linearMSE.pth`，注意文件名是两个 `+`） |
| HPS v2.1（1.97GB） | `$W/reward_weights/HPS_v2.1_compressed.pt`（HF `xswu/HPSv2`，xet-bridge 限速，需续传重试） |
| CLIP ViT-L/14 (laion2b_s32b_b82k) | `$W/hf_cache/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K` |
| CLIP ViT-H/14 (laion2b_s32b_b79k) | `$W/hf_cache/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K` |

下载脚本（notebook 侧，一次性）：`/root/dl_reward_weights.sh`（hf-mirror +
HF_HUB_DISABLE_XET=1 + 8 次重试；laion 老 repo 非 xet 速度快，xswu 走
xet-bridge 慢但可续传）。

## 验证结果

| 组 | aesthetic（0-1） | HPS 余弦 |
|---|---|---|
| gen_47500（n=10，糊生成） | 0.510–0.602，均值 0.551 | 0.043–0.184，均值 0.147 |
| real pexels（n=3） | 0.529–0.596，均值 0.571 | 0.227–0.258，均值 0.241 |

- 区间验收：美学分在 [0,1]、HPS 余弦在 ~0.2–0.35 预期带（real 端）。✓
- 区分度：HPS real/gen 分离干净（0.24 vs 0.15，且无重叠：real 最低 0.227
  > gen 最高 0.184）。美学头 real 仅高 0.02——笔触纹理在 LAION 美学分布里
  本来就得中等分，与 judge 探针"style_flowers 拿高分"的观察一致。
- 注意：pexels caption 是中文，HPSv2 为英文训练，余弦系统性偏低；
  组内排序不受影响，跨语言比绝对值要谨慎。

## 复现

staging 在 notebook `/root/scorer_verify_stage/`（repo 最小快照 +
`scorer_verify.py` + 10 张 cell + manifest）；本地原件 `/tmp/scorer_verify_stage/`
（易失）。运行：

```bash
HF_HUB_OFFLINE=1 HF_HOME=$W/hf_cache \
  $W/venv-precompute-gpu/bin/python scorer_verify.py \
  --weights-dir $W/reward_weights --manifest manifest_cells.json \
  --pexels-jsonl $W/data/caption_enrich/pexels/captions.jsonl \
  --pexels-thumbs $W/data/caption_enrich/pexels/thumbs --n-real 3 \
  --out results.json
```

## 对 Stage 6 的含义

1. ensemble 里 HPS 承担主要的 real/gen 分离信号；美学头权重不宜过高。
2. 美学头的弱区分度 + judge 对纹理的宽容（judge_probe_sii.md）指向同一个
   结论：**held-out scorer 必须保留**，且最好选与训练 reward 不同族的信号。
3. CPU 前向速度未测（本次只验证正确性）；reward budget 若依赖本地 scorer
   吞吐，需在 reward 路径联通后补测。
