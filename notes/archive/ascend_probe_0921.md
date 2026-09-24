# Ascend 910B 可行性探针（2026-09-21）

> Archived record. Current pretraining decisions and status are in
> [the Ascend plan](../ascend_pretraining_0924.md); old launch commands and
> configuration switches require the source revision used for that run.

## 结论：可用，且值得开一个昇腾分支做 infra pass

## 资源

- `昇腾卡公共空间` / `910B资源`：ASCEND 910B (64GB)，**804 空闲 / 1984 总量**，单节点 16 卡。
- job 配额只有 16 卡档（`16,64,1024` / `16,128,1024`，priority any）；notebook 有 1/2/4/8/16 卡档。
- **notebook exec 在昇腾区不可用**（JupyterTerminal 不响应，两个镜像均复现）——只能走 job。

## 软件栈（已验证）

- 镜像 `cann:9.0.1-910b-ubuntu22.04-py3.12`（CANN 9.0.1，py3.12，与 host driver 25.0.rc1 匹配）。
- 内部源 `http://nexus.sii.shaipower.online/repository/pypi/simple/` 在 sj 区可达且快（torch 900MB 约 2 分钟）。
- `pip install torch==2.9.1` + `pip install --no-deps torch-npu==2.9.1` → **torch 2.9.1+cu128 + torch_npu 2.9.1，与我们锁定的 2.9.1 运行时完全一致**。
- 不要用 `ascend-torch2.8.0-cann8.3.rc2-a2_910b-...amd64` 镜像：CANN 8.3 与 host driver 不匹配（import 符号错 / aclInit 507008）。

## 实测算力

- BF16 matmul 4096³：**316 TFLOPS**（910B 标称 ~320，近峰值；4090 有效 BF16 ≈ 165）。
- SDPA bf16 ✓、backward ✓、HCCL available ✓、16 卡可见 ✓。
- 模型层 smoke：hero 配置 DiT（1 double + 24 single, 1152d/16h）fwd+bwd 通；Muon step 通。
  256p bs16 无 ckpt/compile 753ms/it 38GB（未优化数字，仅供正确性参考）。

## 网络

- 公网可达（pypi/github/huggingface/hf-mirror/gitee/modelscope 全通），但**大文件吞吐差**（~1.5MB/s，疑似出口 QoS）。
- 内部 nexus pip 源快（~8MB/s）。
- hf-mirror 下 17MB 代码包秒级完成 → **HF 可作为 qb→sj 代码桥**（已建 `kaupane/artflow-code-bridge`，public）。

## 存储隔离（最大障碍）

- 昇腾 job **只挂 `/inspire/sj-ssd3/global_public`（只读）**；qb-ilm GPFS 不挂。
- 1.3TB precomputed 数据集在 qb-ilm，过不去。选项：
  1. 在昇腾侧从 HF 原始数据**重新 precompute**（910B 空闲免费，text encoder + VAE encode 是推理负载，NPU 适合；权重 Qwen3-0.6B 走 ModelScope，e2e VAE 走 HF 桥）。
  2. 只把 hero 需要的子集搬过去（上传 HF/ModelScope 再拉，1.5MB/s 下 1.3TB 不现实；子集仍数百 GB）。
  3. ModelScope 做 bulk 数据桥（速度未测，阿里 CDN 理论快）。
- 代码：走 HF 桥（已验证）或推 git 分支。

## 代码适配面（预估小）

- 训练代码基本 device-agnostic（accelerate + `accelerator.device`）；`torch.cuda.*` 调用均有 `is_available()` 守卫（NPU 上自动跳过，失去 mem 遥测，可后补 torch.npu 等价物）。
- 待验证：accelerate 分布式在 NPU 上选 hccl backend、torch.compile（torchair/inductor-npu）、bucket 采样器、native_flash_varlen（CUDA-only，需 fallback）。

## hero recipe 昇腾版修订（用户 09-21 拍板，infra pass 验证后实施）

- 只改：micro batch size、gradient accumulation steps、步数设定；其他不动。
- 步数：600k（4090 版为 480k）。
- warmup：20k 步（4090 版 5k）。
- EMA：调参让 EMA 不要落后 live 太多（600k 步下 0.9999 记忆跨度 ~10k 步相对更短，可加大 decay 或按 640p 检查点实测 ema_rel_distance 定）。
- bucket plan：按 64GB HBM 重算（4090 24GB 版不适用）。

## 未做（等拍板）

- 端到端 train.py smoke（需搬 Qwen3-0.6B + e2e VAE + 一个小的 precomputed 子集，或在昇腾侧现做 precompute）。
- 16 卡 HCCL 多卡 smoke。
- 吞吐 infra pass（compile、ckpt、bucket plan 重算——显存 64GB 与 4090 24GB 完全不同，bucket plan 必须重算）。

## e2e smoke 根因（09-21 晚，已定位并修复）

- 现象：e2e5~e2e9 全部挂在训练前 baseline eval-loss probe（train.py:986），faulthandler 栈停在 MSRoPE.forward / artflow.py:267，native 栈显示阻塞在设备队列（tensor_to_list 同步点），TBE task_distribute 子进程报错。
- 根因：`_autocast_ctx`（src/evaluation/eval_loss.py:64）**只在 device.type=="cuda" 时给 autocast**。NPU 上返回 nullcontext → fp32 模型权重直接吃 bf16 输入（probe 里 z0/z1 显式转 bf16，txt 来自 bf16 text encoder）→ 混合 dtype 的 TBE kernel 编译死锁。训练主循环不受影响：accelerate mixed_precision=bf16 用 `state.device.type` 自动包 autocast（"npu" 可用）。
- 为什么最小复现全部通过：repro 的输入 dtype 与模型 dtype 一致，没踩到混合 dtype 路径。
- 修复：`_autocast_ctx` 扩到 `("cuda", "npu", "xpu")`（commit 待提交；本地 tests/test_evaluation.py 通过）。
- 教训：昇腾上 dtype 不匹配不报错而是 TBE 编译挂死——以后 NPU 挂起先查 dtype 路径。

## 根因排查第二轮（09-21 深夜）

- `_autocast_ctx` 修复后 e2e10 仍挂在同位置 → dtype 不匹配不是（唯一）根因。
- repro3：fp32 权重 + bf16 输入 + `torch.autocast("npu")` + complex rope / real_rope 单卡 forward **均 2.2s 通过**（每模块 sync hook 定位，无挂点）。probe 条件的裸 forward 全部排除。
- 新假设：挂起点在 probe 之前 enqueue 的 kernel（text encoder encode，output_hidden_states=True，分块、最长 2048）——Python 栈指向 probe 内第一个同步点（MSRoPE `img_freqs.to(device)` H2D），实际设备队列可能早已被 text encoder 卡死。
- repro4（运行中）：完整复刻——真 EvalLossProbe（mini-d1 + 真 Qwen3 encode）+ EMA 模型 evaluate()，faulthandler 300s。
- 若 repro4 通过：剩余差异只有 train.py 外围状态（Accelerator/prepare/Muon 构造/采样器）。

## 真相（09-21 23:15，e2e11 插桩结论）

- **probe 从未挂死，只是慢**：插桩显示每个 batch（bs=64, txt_w=1948, latent 42x24, 4 个 t 步）约 34s 稳定推进；1717 samples = 27 batches ≈ 15 分钟。此前 e2e5~e2e10 的 480s/720s abort 都在 probe 跑完前触发，faulthandler 栈只是慢速前进中的快照。
- 慢的原因：probe 走 fast_attn=False 的 eager 路径 + NPU 无 compile；单 forward ~8.5s。
- 教训：「挂起」诊断必须先排除「稳态慢」——加进度日志比读栈更直接。
- 对 hero 的含义：NPU 上 baseline/周期 probe 每次 ~15min（256p），需要 compile（torchair）或减少 probe 频率/规模；TBE 编译在固定 probe shape 下可摊销。
- smoke.toml 已把 eval.loss_samples 缩到 48（~2.5min），e2e12b 验证完整训练循环。
- 本地 mini-d1 缺 sidecar 的原因：save_to_disk 不复制 length_metadata.npz；真机是训练 sampler 构建时现算并写入数据集目录，probe 随后读到 → banded（1717）。

## 训练 OOM 根因与修复（09-22 00:20）

- e2e13：probe 2.3min 跑完（loss_samples=48，eval/loss=1.78834），首个训练步 NPU OOM（58.6/61GB）。
- 根因：`sdpa_with_bias`（src/models/dit_blocks.py）对非 CUDA 设备强制 `sdpa_kernel([MATH])` → math backend 物化 B·H·S² 注意力分数并为 backward 保存，长 caption bucket（seq~2.3K）25 层 ~68GB。
- repro6 实测：NPU 上**默认 SDPA + float bias 走 fused kernel**（bs8 S2304 bf16，fwd 0.03s/bwd 0.05s，与 fp32 math 参考 maxdiff 9e-4）；**bool mask 会让 npu_fusion_attention 报 161001**，必须保持 float bias。
- 修复：sdpa_with_bias 对 npu 直接调 F.sdpa 不进 sdpa_kernel 上下文；CPU 保持 MATH；CUDA 逻辑不变。test_dit_blocks 11 passed。
- 注：probe 的 34s/batch 慢也部分来自同一 MATH 强制（probe fast_attn=False 也走 sdpa_with_bias），修复后 probe 应显著提速。

## OOM 深层定位（09-22 01:10，repro7/8/9/10）

- repro7：最小 bucket（bs=110, tl=19）fwd 即 OOM；repro8 逐模块钩子：**每个 single-stream block 留存 ~3GB**，24 层 ≈ 72GB > 60GB。
- repro9：`torch.autocast("npu")` **功能正常**（Linear/matmul→bf16，layer_norm→fp32，SDPA fp32 输入→bf16 输出）。此前"autocast 失效"假设排除。
- repro10 逐 op 追踪：dtype 全部正确（rope/sdpa 均 bf16，fused kernel 生效）；3GB 是 eager 模式的完整 autograd 存档——qkv/RMSNorm/rope 链 ~1.05GB、**NPU fused SDPA 单次调用 +0.95GB**（疑似保存 P 矩阵/物化 bias，约 3× CUDA 后端）、mlp ~0.85GB、norms fp32 输出 ~0.6GB。
- 4090 (24GB) 能跑 bs=110 是因为 hero 全程 torch.compile（inductor 优化 autograd 存档）；代码库**没有 activation checkpointing**。
- 昇腾出路（infra pass 再定）：①torchair compile（若等效 CUDA 则 block 省回 ~0.8GB）；②按 64GB 重算 bucket plan（eager 下短 caption bucket bs 需砍 ~35-50%）；③（更重的改动）引入 activation checkpointing。
- smoke 临时措施：bucket plan ×0.45（min 4）→ e2e15 验证完整训练循环。

## 单卡 e2e 跑通（09-22 01:55，e2e16）

- bucket plan ×0.45 后 30/30 步完成：~1.95s/it（2 grad accum，bs~50/桶），throughput 26.9 samples/s，"Training finished."。
- probe（loss_samples=48）2.3min，eval/loss=1.78834（随机初始化，与 repro4 一致）。
- 已知缺口：NPU 显存遥测为 0（torch.cuda.* 守卫跳过，torch.npu 等价物后补）；eager 模式 bucket bs 只有 4090 的 ~45%，infra pass 需用 torchair compile 或重算 plan 补回。
- e2e15 还暴露桥下载偶发截断（Qwen3 config.json 缺 model_type）→ canonical 脚本已加 dl() 重试 + config 校验。

## 16 卡 HCCL（09-22 02:20-03:20）

- hccl-smoke（v1）：挂在 caption telemetry 的 float64 all_reduce（HCCL 不支持 double）→ `src/pretrain/caption_telemetry.py` reduce 改 **int64**，进 0922a 代码包。
- hccl-smoke2（0922a）：probe 2.6min 通过（eval/loss=1.78834，与单卡一致），训练跑到 **20/30 步 ~2.05s/it**（与单卡相同，16 卡 DDP 扩展性 OK），随后 **rank2 SDPA fwd OOM**：43.8GB allocated / **59.5GB reserved** → 典型碎片化。其余 rank 无错。
- hccl-smoke3：加 `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` 后 probe 阶段 >15min 无输出（smoke2 同阶段 2.6min）——疑似该 allocator 设置在 probe 大 batch 下病态变慢，等 1500s 窗口的 SIGABRT 栈确认。备选：`expandable_segments:True` 或不设（smoke2 前 20 步本无问题，OOM 出现在 20 步的 SDPA fwd 1.26GB 分配）。

## 存储与数据迁移探查（09-22 03:00）

- 昇腾 job 容器内：qb 区 W 路径**不可见**；挂载 `sj-ipfs01 19P /inspire/sj-ssd3/global_public`（**只读**，全是平台公共数据集）。
- **可写**：job 工作目录 `/inspire/sj-ssd3/project/cq-scientific-cooperation-zone/ky26021`（与 qb 区同项目路径结构，sj 区独立存储）。1.3TB precomputed 的落地区有了。
- CPU 区（inko-patrol）只见 qb-ilm / qb-ilm2 / ssd，**看不到 sj-ssd3** → 无法经 CPU 区直接 rsync 跨区。
- `inspire job create` 无自定义存储挂载选项；`inspire dataset` 只挂官方数据集；`inspire model register` 只能注册同区路径。
- hf-mirror 从昇腾区测速：单连接 2MB/s，**8 连接并行 13MB/s**（近线性扩展）。1.3TB @13MB/s ≈ 28h；开 16-32 连接或多 job 并行可压到 ~12h 量级。
- 迁移候选（待与用户定）：A) qb 区把 precomputed 分片推上 HF（私有 repo，205GB d1 已成功推过），昇腾侧多连接拉；B) 昇腾侧重做 precompute（原始数据集本就在 HF kaupane/*，权重走桥，NPU 做 VAE/text encode 是推理负载正合适）；C) 问平台/学院是否有跨区拷贝通道。

## torchair（09-22 03:00）

- pypi 无 torchair（pypi.org 404），内部 nexus 无，CANN 9.0.1 镜像内 find 无 wheel。torchair 只从 hiascend 官方渠道分发。此线放弃，除非用户能从别的渠道拿 wheel。
- eager 基线（torchair-probe 附带测）：hero 模型 bs=32 tl=64 全模型 fwd+bwd **0.514s/step，peak 29.1GiB**。
- 省显存现实路径变为：activation checkpointing（新代码）或按 64GB eager 重算 bucket plan。

## SDPA 二次方显存根因与出路（09-22 03:30-04:00，bench-sweep + sdpa-mem repro1/2）

- bench-sweep（eager 全模型 fwd+bwd）：img=256 tl=64 bs=54 peak 46.7GiB OK；**tl=512 bs=16 即 42.7GiB、bs=24 OOM；tl=2048/640p/896p bs=16 全 OOM**。显存随 S² 爆炸。
- repro1（B8 H16 S2304 D72 单算子）：no_mask F.sdpa fwd 仅 +0.06GiB（真 flash）；**float bias（B,1,1,S 或 B,1,S,S）fwd +2.81GiB** = per-head S² bf16 的 bias 物化 + P 矩阵，每层存到 backward → 24 层叠乘即 60GB+。bool mask 经 F.sdpa 派发 → aclnnFlashAttentionScore 报错 161001。
- repro2：直接调 `torch_npu.npu_fusion_attention` + **bool mask (B,1,S,S)（或 11SS/uint8/BNSD）均接受，fwd 仅 +0.04GiB = 真 flash**。全 1 mask maxdiff 0.386 → 疑似约定 1=drop（repro3 验证）。
- smoke4（expandable_segments:True）：仍在第 20 步 rank2 OOM，58GB 真实占用（这次不是碎片）——长 caption bucket 的 S² 存档 legit 超容。**唯一能根治的就是换 npu_fusion 路径。**
- 含义：修复后长 caption bucket 的 bs 上限将不再被 S² 锁死，bucket plan 可按线性显存重算，bs 接近 4090 档；不需要 activation checkpointing。

## hccl smoke 现状（09-22 04:00）

- smoke2：DDP/HCCL 本身已验证（20 步 × 2.05s/it，16 卡与单卡同步）。smoke3（max_split）/smoke4（expandable_segments）都在同一 S² OOM 上死——该 OOM 与 allocator 设置无关，等 sdpa 修复后一并复验。

## SDPA 修复落地（09-22 04:00-04:15，repro3-6 + 0922b）

- repro3 参照系设计失误（zeros mask 对 drop25 ref），数值一度自相矛盾；repro4 小规模精确参照两种约定都对不上 → 怀疑 scale。
- repro5：**`npu_fusion_attention` 默认 scale ≠ 1/sqrt(D)**（≈1.0），显式传 `scale=D**-0.5` 后与精确 math **maxdiff=0.000000**。此前所有"约定之谜"都是 scale 默认值导致的。
- repro6（带显式 scale）：**mask 约定 True=drop**；S=2304 bf16 out maxdiff 0.002、梯度 ≤0.004；**saved-for-backward 仅 +0.06GiB**（对比 float-bias 路径 +2.81GiB）。mask 必须 (B,1,S,S)，aclnn 拒 (B,1,1,S)。
- 修复（`src/models/dit_blocks.py` sdpa_with_bias NPU 分支）：`torch.isneginf(bias)` 得 drop 位 → expand (B,1,Sq,Sk).contiguous() → `torch_npu.npu_fusion_attention(q,k,v,H,"BNSD",atten_mask=mask,keep_prob=1.0,scale=D**-0.5)[0]`。lazy import torch_npu（本地 CPU 环境无此包）。本地 test_dit_blocks 11 passed。
- 已打包 **artflow-code-0922b.tar.gz** 上桥；e2e17（单卡）+ hccl-smoke5（16卡）+ bench-sweep2（修复后 bs 复测）已提交。
- 待办：smoke 全绿后，用 bench-sweep2 数据重算 bucket plan（线性显存模型，bs 上限应显著回升）；torchair 线已弃；activation checkpointing 大概率不再需要。

## 数据迁移测速补充（09-22 04:30）

- qb 区（inko-patrol）→ HF 上传：单流 **9.8MB/s**（200MB 测试文件，scratch repo kaupane/speed-test-scratch，事后删）。
- HF raw 覆盖盘点：**只有 d1 国画（chinese-painting-collection，含图 205GB）raw 在 HF**；pexels-people-captions 只有 caption+url；wikiart/artbench/human/vintage/relaion 均 captions-only；d2-museum、d3、d4 原始图像都不在 HF → "昇腾全量重 precompute"不可行（除非重新跑 fetchers 重抓，等于重做 Stage 1-3）。
- 结论：现实路径只有 **qb→HF（私有 repo，多分片+多进程并行上传）→ sj 多连接拉取**。单流 9.8MB/s × 并行 4-6 进程 ≈ 40-60MB/s → 1.3TB 上传 6-9h；sj 侧 13MB/s×(16-32 连接或多 job) ≈ 50-100MB/s → 下载 4-7h。合计约 1 天，可无人值守。
- 待用户定的点：①是否就走这条 HF 桥（还是有学院内部跨区通道更快）；②precomputed 是否原样分片（parquet + sidecar，训练可直接流式读 HF？不——sj 侧落地到 sj-ssd3 项目目录再训练）。
- 昇腾 job 并发上限观察：同一用户同时最多 2 个 16 卡 job 运行，其余排队。

## 修复验证通过（09-22 04:30，e2e17 + hccl-smoke5，代码 0922b）

- e2e17（单卡）：30/30 "Training finished"，稳态 **1.52s/it**（修复前 1.95，省掉 S² 物化还更快）。probe eval/loss=1.78834 **与修复前逐位一致**。
- hccl-smoke5（16卡 torchrun DDP）：**30/30 全程通过**，稳态 1.57s/it——越过了此前必死的第 20 步长 caption bucket。HCCL/DDP/EMA/probe 全链路绿。
- 昇腾 bring-up 核心链路完成。剩下：bench-sweep2 → bucket plan 重算 → 数据迁移（用户定）→ hero recipe 昇腾版。

## bucket plan 昇腾版（09-22 05:10，sweep3 + assemble）

- sweep3（0922b 修复后，139 个 OK 点）：显存严格线性 **peak ≈ 3.3GiB + 2.1MB/token**（三个分辨率类 R²=0.999）；eager autograd 存档 ~2.1MB/token 是主项（torch.compile 在 CUDA 上省的就是它，NPU 无 compile 只能吃这个成本）。吞吐 ~17-21ktok/s/card（长 caption 桶更高），约等于 4090+compile 的 ~24ktok/s 的 7-8 成。
- 计划组装：`scripts/ascend/assemble_ascend_plan.py`。**boundary 原样复用 0914**（数据/模型没变），bs 按线性模型重算。sweep 静态只有 3.3GiB（SGD 无状态），真实训练 Muon+Adam+EMA 静态 ~13GB → 预算取 **44GiB**（保守版）：256p bs 8..72、640p 5..12、896p 3..6，各 stage 预测步时 p5-p95 分别在 0.90-0.96 / 0.97-1.07 / 0.98-1.19s（桶间对齐好）。
- 有效批量（16 卡）：256p accum1 ≈930（≥640 ✓）、640p accum4 ≈688（≥512 ✓）、896p accum5 ≈430（≥400 ✓）。
- 600k 步（75:20:5 = 450k/120k/30k）单节点墙钟估计：450k×1.05 + 120k×4.1 + 30k×5.3 ≈ **312h ≈ 13 天**（~5000 NPU-h）。加速杠杆是多节点（2 节点 ~6.5 天、4 节点 ~3.3 天），需先验证跨节点 HCCL。
- 保守 44GiB 版已上桥（hero-256p-k20-ascend.json）；ascend-hccl-realplan 用**真实 plan**（非 ×0.45）跑 16 卡验证中。若实测峰值有富余，可按 54GiB 预算重出一版（bs +25%）。
- 备选未做：activation checkpointing（bs 可 3-4 倍但 +33% 重算，短桶净收益 ~20%，新代码风险，暂不做）。

## 2026-09-22 05:30-06:00 realplan 验证 + 迁移启动

- **ascend-hccl-realplan ✅**:真实 44GiB plan(boundary 复用 0914 + sweep3 线性模型 bs)16 卡 30/30,无 OOM,eval/loss=1.78834 与 smoke 逐位一致。保守 plan 全链路验证完毕。
- **0922c 已上桥**:0922b + train.py NPU 显存遥测补丁(mem_gb/peak 走 _device_* helper)。
- **ascend-hccl-headroom54g**(进行中):54GiB 预算版 plan(bs +25%:256p 10..90 / 640p 6..14 / 896p 4..7)+ 0922c,借遥测读真实 peak(含 Muon+Adam+EMA 静态),校准最终预算。
- **数据迁移启动**(用户已预授权"把文件迁过去"):
  - 数据源:`/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/artflow/precomputed_dataset` = 1.3TB(256p 104G / 640p 502G / 896p 574G,arrow shard,每 dir ~15 个大文件)。
  - 只迁 hero mix 内的 dir(跳过 d3-synth、d4-inat@640p、d3-synth-v2@896p、d4-zimage@896p 等 0 权重项),**加 light-eval@(评测量要用)**。
  - 路径:inko-patrol(qb)→ HF 私有 repo `kaupane/artflow-precomputed-{256p,640p,896p}` → 昇腾侧 hf-mirror 拉取。脚本 `scripts/ascend/migrate_upload.py`(6 并发 upload_folder,每 dir 3 次重试),256p 已在 inko-patrol 后台跑(nohup,日志 /root/ascend_migrate/upload_256p.log)。
  - 昇腾侧下载不烧 16 卡 job:昇腾 workspace 支持 **2 卡 notebook**(quota 2,16,128),已建 `ascend-dl-staging` 做下载落盘 + 后续单卡 probe。
  - 顺序:256p 先(解锁 hero 第一阶段 75% 步数),640p/896p 跟上。
- kaupane/speed-test-scratch 已删。

## 2026-09-22 05:55 预算校准（54 OOM → 50 验证中）

- **ascend-hccl-headroom54g ❌ OOM**:54GiB sweep 预算在 step 0 全 rank OOM(active 59.5GiB / 60.96GiB)。由此两点标定真实静态（Muon+Adam+EMA,非 sweep 的 SGD 3.3GiB):real_peak ≈ (budget−3.3) + static_real,OOM 点 59.7 → **static_real ≈ 9GiB**。
- 4090 惯例 0.88×VRAM → 目标 peak ~53.6GiB → sweep 预算 ≈ 53.6−9+3.3 ≈ 48GiB。取 **50GiB** 出第三版(bs 约为 44 版的 +14%),ascend-hccl-headroom50g 验证中；0922c 遥测在成功跑完 30 步后可读真实全局 peak，最终预算按实测再定。
- 迁移上传实测：6 并发 upload_folder 聚合 **~55MB/s**(单流 9.8 → 近线性扩),104GB/256p 约 30min;640p(502G)≈2.5h、896p(574G)≈3h,全程远好于昨天单流估算的 1 天。
- **教训**:headroom54g OOM 后 torchrun 未退出、job 挂住占卡（部分 rank 死于 OOM，其余卡在 collective)。hero 启动脚本必须加外层 timeout/watchdog,OOM 即退出让 job 系统重启。
- 256p 上传完成（104GB,14 dir,零失败，~32min)。640p 已启动。
- **ascend-dl-staging notebook 失败**:cann 镜像在 notebook 模式下 JupyterTerminal 始终不响应（restart 无效，events 显示 ready 但终端死）→ 已删。昇腾 workspace notebook 路线放弃，下载改走 16 卡 job（并发上限 2,可接受）。
- ascend-dl-256p 已提交:snapshot_download(max_workers=24,hf-mirror,私有 repo + token)→ sj-ssd3 …/ky26021/artflow/precomputed_dataset。

## 2026-09-22 06:30 预算定稿 + 下载 xet 坑

- **预算定稿 48GiB**:50g 实测 peak_mem_gb=58.9 / reserved 59.5(0922c 遥测生效);real static(Muon+Adam+EMA)≈12GiB,与最初估算 13 吻合。48g → 预测 peak ~56.9,留 4GiB 余量抗 13 天长跑的碎片爬升。48g plan(256p 9..79 / 640p 5..13 / 896p 4..6)已全量上桥 hero-{res}-k20-ascend-48g.json。hero_recipe 昇腾节已定稿（等用户 sign-off)。
- **xet 坑**:hf-mirror 只代理主站 resolve,不代理 xet CDN(us.aws.cdn.hf.co,sj 区直连超时）。upload 端 hf_hub 1.32 默认 xet 上传不受影响（服务端兼容经典 LFS 拉取）,**下载端必须 HF_HUB_DISABLE_XET=1**。ascend-dl-256p-v2 已带此修复重提。

## 2026-09-22 06:40 hero 启动脚本 + 0922d

- **xet 复盘**:hf-mirror 对 resolve 一律 308 到 huggingface.co,再 302 到 xet-bridge(us.aws.cdn.hf.co);sj 区**能**连该 CDN(bridge 的 qwen3 就是这么拉的,~2MB/s/连接),但 24 并发会拥塞崩溃(11 错/7 文件)。v1 的"禁 xet"方向错了——redirect 目标与客户端无关,是文件存储后端决定的。昨天测速 8 并发=13MB/s 是甜点。ascend-dl-256p-v3 改 max_workers=8,ETA ~2.2h。
- **hero_stage_ascend.sh**(scripts/ascend/):16 卡 HCCL 一 stage 一 job;持久化 staging(pylibs pip --target + .done marker、repo-ascend、models 都在 sj-ssd3);OOM watchdog(日志出现 "NPU out of memory" 即 kill 整个进程组,治 headroom54g 的挂死);flock 单写者;resume 逻辑同 4090(--resume_full / 跨 stage --reset_sampler,stage_control 用 src.pretrain 新路径);swanlab 自动探测(连不通则 disabled);DRY_RUN=1 可验证 staging+config 渲染不点火。override:T=600000(默认)、accum 1/4/5、lr_warmup_steps=20000、bucket_plan=ascend-0922-48g。
- **artflow-code-0922d.tar.gz** 上桥 = 0922c + configs/ + bucket_plans/hero/ascend-0922-48g + assets/eval + 本启动脚本。
- 896p 上传已挂链(640p 完自动起)。
- **ascend-hero-dryrun ✅**:pylibs 持久环境(pip --target + marker,~4min 建成,后续 restart 免装)、repo-ascend、models 全部落 sj-ssd3;config 渲染/resume 分支正确;48g plan 经 override 生效( layering 后覆盖 0914 路径)。
- **netprobe**:sj 区 swanlab.cn 200、huggingface.co 200、xet transfer/cas 端口均可达 → hero 可以开 swanlab cloud(netrc 在 submit 时 base64 注入,launcher 已改)。

## 2026-09-22 07:35 下载吞吐现实 + 时序账

- xet 客户端大文件实测也不快（6GB > 20min 未完成，≤5MB/s),classic 与 xet 同被 xet-bridge CDN 限流，当前窗口 ~4-8MB/s/job（昨晚测速 13MB/s 是更空闲窗口）。24 并发会崩，8 并发稳定。
- **时序账（关键）**:hero 256p stage 墙钟 ~10 天，而 640p+896p 下载即使 6MB/s 也只要 ~2 天 → 下载不是瓶颈，256p 落地（预计今午）即可发射。下载链：v3(256p) → 640p → 896p，由巡检接力。
- **安全**:ascend_download.sh 早期版本 set -x 把 HF oauth token 打进 job 日志（v1/v2/v3/xet-test/xet-test2 均中招）。已改脚本（set -x 前 export,去重复行）。**需提醒用户轮换该 HF token**。

## 2026-09-22 14:20 CDN 限流窗口 + 补缺经验

- **症状**:snapshot_download 报 ALL_DONE 但 sj 侧缺 22 文件(relaion 2、vintage 6、zimage 10、light-eval 4;后两者整目录缺失)。kill 后 resume 的 metadata 会误判已完成——**ALL_DONE 不可信,必须做文件清单比对+行数对账**。
- **限流本质**:us.aws.cdn.hf.co(xet-bridge)对 sj 区按时间窗口限流。好窗口(01:50-06:30 拉 250GB;12:20-12:33 拉 2 文件)3.5MB/s/连接,坏窗口(11:19-12:19、12:33-14:20+)所有节点 0 字节。HTTP/1.1 与 HTTP/2 同样 stall;TTFB 正常(~1s 拿到 302),body 0 字节。**xet 协议不是独立路线**:chunk 重建也走同一 xet-bridge host,同被限。直连 huggingface.co 从 sj 超时(与 netprobe 早晨的 200 矛盾,可能也是窗口性的)。
- **无效假设**:单文件 cache 中毒(relaion 79 在坏节点 stall、好节点秒下);出口 IP 抽签(同窗口多节点全好或全坏,更支持时间窗)。
- **有效策略**:curl -C - 断点续传 + 6 并发 + 一轮零进展即退出让平台换节点重启(refill-v6,--auto-fault-tolerance ×50),等好窗口批量补。speed guard --speed-time 120 --speed-limit 10240 快速失败。
- **再踩 token 泄漏**:诊断脚本 set -x + curl -H "Authorization: Bearer $TOKEN" 把 oauth token 打进 cksz2/cksz3 job 日志(加上之前下载脚本那次共三次)。**已两次提醒用户轮换**。
- **绕过尝试（均死路）**:sj 区只挂 sj-ssd3,看不到 qb-ilm(ascend-mnt-probe);sj→inko-patrol(10.244.4.138)直连不通,curl 000(ascend-xzone-probe)。跨区传输只能走公网。备选未试:ModelScope 做中转（需账号，用户决策）。
- 896p 上传失败是 HF 账号级计费限制("setup automatic credit recharge")。用户决策(2026-09-22):等 sj 侧 256p+640p 下载对账完成后删 HF 上这两个 repo 腾配额再重传 896p。
- **netrc 注入坑**:inspire notebook exec 输出末尾带一行 `OK`,`| tail -1` 会把它当 base64 → launcher `base64: invalid input` 重启循环(ascend-hero-smoke 首轮中招)。正确取法:`... | grep -E '^[A-Za-z0-9+/=]{20,}$' | tail -1`。
- 更正:inspire notebook exec 输出里 base64 与 `OK` **粘连无换行**(`...Ugo=OK`),纯字符集过滤会把 OK 吞进去。稳妥取法:qb 侧 `echo B64START; base64 -w0 <f>; echo; echo B64END`,本地 awk 按标记抽取再 tr -d 空白(smoke3 已验证 len=120 解码正确)。

## 2026-09-22 18:00 hero smoke 暴露的两个 launcher bug + 碎片化 OOM

- **256p 对账通过**(17:07 SJ_ROWS 14 项全对)后发全链路 smoke(T=1000/stop 750、ckpt 200、grid 250、loss 100、warmup 100、swanlab cloud、KID at end)。
- **bug1 陈旧 OOM 签名杀新训练**:watchdog `tail -c 400000 $LOG | grep "NPU out of memory"` 扫描的是**持久化追加日志**,上一轮的真实 OOM 文本留在日志里,重启后新 watchdog 立刻误杀新训练,worker-1/2/3 连环死亡。修复:字节游标,只扫 launcher 启动后追加的内容(hero_stage_ascend.sh)。
- **bug2 netrc base64 粘连 OK**(见上条,smoke/smoke2 各烧几轮重启)。
- **真实 OOM(48g plan,~200 步内)**:rank7 rotary emb 分配 96MiB 失败,51.77GiB active / **59.62GiB reserved** —— 碎片 ~8GiB 吃光 48g 的 4GiB 余量。speedtest 只跑 100 步没遇到(峰值 reserved 57.9)。max_split 的"病态慢"结论来自 sdpa 修复前(smoke3 时代),证据薄弱。smoke4 试 `ALLOC_CONF=max_split_size_mb:256`(launcher 新增 env 透传,默认仍 unset),盯 s/it 与 reserved 峰值;若慢或仍 OOM,回退重生成 44g plan。
- **48g plan 判死刑(2026-09-22 18:30)**:smoke3(无 alloc 设置)~step250 OOM、smoke4(max_split_size_mb:256)~step55 OOM,两者都是 reserved 爬到 ~59.7GiB 天花板后 96-146MiB 常规分配失败,active 仅 ~51.4GiB。碎片化随遇到的桶形状数单调累积,max_split:256 无效。speedtest 100 步幸存纯属抽桶运气。**结论:48g 余量(4GiB)在这个 allocator 上对长程训练为零**。降用 ascend-0922(44g 级,bs 8..72,sweep 实测 384 spl/s 不低于 48g),0922g 代码包含 ascend-0922 + ascend-0922-48g 两套,launcher 新增 PLAN_DIR(默认 ascend-0922)。smoke5 = 44g + 无 ALLOC_CONF + 游标 watchdog,750 步全程验证;通过则 hero 同配置。

## 碎片 OOM 攻坚战（09-22 晚，smoke5/6/7）

- smoke5（44g plan）：前 489 步完全健康（loss 2.02→0.95，曲线与 4090 早期一致），~step 501 再次 NPU OOM（reserved ~59.7GiB 触顶 vs active ~51.4GiB）。
- smoke6（+cache_clear_interval=50，已确认配置经 [telemetry] override 生效、_merge 后者优先）：**未能防住**，同样在 ~step 500 段 OOM。结论：reserved 蠕变不是 pytorch caching allocator 的空闲块碎片，empty_cache 压不住（疑似 CANN/HCCL workspace 或 torch_npu empty_cache 不 trim）。
- smoke6 次生故障：watchdog kill -9 后 **/dev/shm/torch_* 段泄漏**，同节点后续 attempt 3 分钟内死于 ENOSPC(28)。launcher 已修：启动时 + watchdog kill 后 `rm -f /dev/shm/torch_*`，并 df -h /dev/shm 进日志。
- smoke7 = CACHE_CLEAR=50 + ALLOC_CONF=expandable_segments:True（sdpa S² 修复后首次复测 expandable）。若仍败，兜底为"每 2000 步计划性自重启清 allocator"。
- swanlab run 状态有滞后：kill -9 的 run 会显示"运行中"直到心跳超时，以平台 job 状态为准。

## ACL 500001 / SetPrecisionMode 根因（09-22 晚，memdiag3-5 三次复现定位）

- 现象：自建探针脚本首个 acl 算子（conv）即死 `AclSetCompileopt(ACL_PRECISION_MODE) 500001` + `ModuleNotFoundError: No module named 'tbe'` → GEInitialize/OpCompileProcessor 初始化失败。
- 根因：探针脚本 `export PYTHONPATH=$W/pylibs`（**覆盖**），把 CANN 镜像自带 PYTHONPATH 里的 toolkit python 目录（含 tbe 模块）冲掉了； eager 模式首个算子也要初始化 GE 编译处理器，找不到 tbe 即 500001。训练 launcher 是 `export PYTHONPATH=$W/pylibs${PYTHONPATH:+:$PYTHONPATH}`（**追加**），所以训练免疫。
- 教训：**sj 侧一切脚本 PYTHONPATH 一律追加，不得覆盖**。memdiag6/srcb3 已修。
- torchair srcb2 另发现：py3.12 无 stdlib distutils，torchair configure 需要 → 脚本开头 pip install setuptools。
- sj 区可达性：gitcode.com / raw.gitcode.com / modelscope.cn 均 REACH 302（可用于 torchair 源码与 896p ModelScope 中转）。
- smoke8（仅 expandable_segments，0→750）绿 → hero 显存配置定为 expandable_segments:True，不需要 cache_clear。
- NPU 利用率实测：存活训练 worker 86% GPU util（平台跨 pod 平均会被死 pod 拉低，勿误读）。

## 2026-09-22 21:30 memdiag6 定案 + torchair 构建第三轮 + 三探针并行

- **memdiag6（修复 PYTHONPATH 追加后首次跑出有效数据）**：A 变形状单卡 resv 51.3→59.8→53.4→58.7GiB 随桶形振荡、seg 564→646 累积但 inact_split 4.6-10.4GiB 波动不单调；B 固定形状 100 步**完全平坦**（resv 51.11 / seg 568 / inact_split 7.46 零漂移）；C 16 卡 HCCL 52.5→59.2GiB 后走平，alloc 比单卡多 ~2GiB（HCCL workspace）。**结论：无泄漏，reserved 蠕变是峰顶现象不是趋势**；早前 OOM 系 48g 余量（4GiB）不够峰顶波动，expandable_segments 单配置即正解，cache_clear 只会添乱。坑：`torch.npu.memory_snapshot()` 在此栈零参数（传 dev 会 TypeError），每 10 步 memory_stats 行已足够。
- **srcb3 失败根因**：torchair configure 的新鲜 python 进程 `import distutils` 失败。setuptools 的 distutils shim 依赖 `distutils-precedence.pth` 在解释器启动时执行，而 **.pth 只对真实 site 目录生效，PYTHONPATH 目录不处理 .pth**——所以 pylibs 里的 setuptools 帮不上。srcb3 的 `pip3 install setuptools` 可能装到了别的解释器或索引不通（输出被 tail -1 吞了）。srcb4 修复：`python3 -m pip`（同解释器）+ tuna/aliyun/pypi 三索引轮换 + 装完显式验证 `import distutils` 和 `import torch` 再进 configure。
- **0922i 代码包上桥**：含 --npu_fused_adamw / --compile_backend=torchair / launcher 新开关 TORCHAIR/FOREACH/FUSED_ADAM/EXTRA_ARGS，CODE_VER 默认 0922i。
- **并行在跑**：ascend-torchair-srcb4（构建+bench）、ascend-npu-caps（NpuFusedAdamW/NPUGraph/triton/foreach 盘点 + NPUGraph 整模型捕获可行性）、ascend-hostprobe（400 步真实训练 + --step_breakdown --cpu_wall_profile + foreach + fused_adam，量化 host vs device）、ascend-dl-640p（502GB 提前并行下载，不等 hero 占槽——availability 578 卡充足）。
- 昇腾 availability 21:00：2224 总 / 1646 用 / 578 可用 / 0 整机节点，ICLR 前用量持续上涨。

## 2026-09-22 21:45 caps 盘点 + srcb4/5 pip 坑

- **ascend-npu-caps 结果**：NpuFusedAdamW ✓、foreach ✓、triton 3.5.0 ✓（但 triton_ascend 缺，inductor 无路）、NPUGraph API 存在但**整模型捕获死于 Conv2D 是 aclop  legacy 算子**（`Cannot run aclop operators during NPU graph capture`，报错提示可试 `torch.npu.config.allow_internal_format=False`——patch embed conv 走了内部格式）。NPUGraph 路线工程量大（20 桶 × 5 分辨率的静态图 + DDP 交互），后备；torchair 优先。
- **srcb4 失败根因（确凿）**：pip 的 already-installed 检查（importlib.metadata）会扫 PYTHONPATH 目录，看到 $W/pylibs 里的 setuptools 就直接跳过安装；而 setuptools 的 `distutils-precedence.pth` 只在真实 site 目录被 site 模块执行，PYTHONPATH 目录不触发 → 新鲜解释器 `import distutils` 永远失败。三个索引全部"成功"但什么都没装。
- **srcb5 修复**：`--ignore-installed` 强制装进解释器自己的 site-packages（.pth 会生效）+ 文件级 shim 兜底（/tmp/pycompat/distutils.py 别名到 `setuptools._distutils`，覆盖 `import distutils.<submod>`）；另外 `set -o pipefail`——srcb3 里 `timeout 300 configure | tail -5 || fail` 的 fail 永远不触发（管道退出码是 tail 的），configure 超时被静默吞掉继续 cmake。

## 640p 对账基准（qb 侧 load_from_disk num_rows，2026-09-22 21:55 取）

d1=91151, d2-museum=8727, d2-wikiart=138346, d3-human=114100, d3-people-a=55009, d3-people-b=55066, d3-pexels=37417, d3-synth-v2=10893, d3-synth=20000, d4-inat=ERR:IndexError(qb 侧即如此，sj 侧同样报错即视为一致), d4-megalith=5361, d4-pd12m=157683, d4-relaion-p0..p4=78307/78055/78347/78329/78388, d4-vintage=158531, d4-zimage=45248, light-eval=1875。

## 2026-09-22 22:20 hostprobe 崩溃根因 + srcb5 进展 + 0922j

- **ascend-hostprobe 两次 attempt 均 3 分钟整 TRAIN_EXIT 1（确定性）**：`--step_breakdown` 的 `bd_mark` 用 `torch.cuda.Event`（stage-3.5 在 4090 上写的，首次上 NPU），首个 micro 即崩；swanlab 两条空曲线佐证死于 step 50 前。0922j 修复：按 accelerator.device.type 选 torch.cuda/torch.npu 事件 API（与 mem telemetry 同款模式）。教训：**telemetry 新开关上 NPU 前必须过一遍 device 假设**。
- **srcb5 重大进展**：distutils（shim 生效，SETUPTOOLS_AT 证实 srcb4 根因= pylibs setuptools 84 骗过 pip 检查）→ configure/cmake/make/wheel/pipinstall 全过 → 死于 `import torchair` 的 `import pkg_resources`（setuptools 84 已删 pkg_resources）。srcb6：构建结果在 $W 持久化（BUILD_SKIP 直接复用），往 pylibs-torchair 里补 `setuptools<81` 提供 pkg_resources，再 import + eager/torchair bench。
- hostprobe2 用 0922j（FOREACH=1 FUSED_ADAM=1 + 两个 breakdown）重发；CODE_VER 默认已 0922j。

## 2026-09-22 22:50 srcb6：torchair 可 import，bench 死于第三次 PYTHONPATH 覆盖

- pkg_resources 修复生效（setuptools<81 入 pylibs-torchair），**TORCHAIR_OK**——7.2.0 分支源码构建的 wheel 在此栈可用。
- 但 eager/torchair 全部 BENCH FAIL ACL 500001：脚本第 90 行 `export PYTHONPATH=$W/pylibs-torchair:$W/pylibs` 是覆盖式（顶部追加的 tbe 路径又被丢了）。**这是第三次同一根因**（memdiag3-5、srcb3 前、srcb6）。规则升级：sj 侧任何脚本写完必须 grep 一遍所有 PYTHONPATH 赋值确认 `${PYTHONPATH:+:$PYTHONPATH}` 结尾。srcb7 已修，纯 bench 重跑（构建 BUILD_SKIP 复用）。

## 2026-09-22 23:10 hostprobe2 挂死 75 分钟根因 + launcher 硬化

- 现象：pod 21:42 启动，pip.conf 两行 + notice 两行后**75 分钟零输出**——卡在第 3 个 pip install（deps 大包）。hostprobe1 同等工作 6 分钟完成 → 节点级问题（该节点到 nexus 内网镜像慢/不通）。launcher 的 pip 阶段无超时（watchdog 只管训练阶段），会无限挂死烧节点。
- 修复（launcher 本地改、注入即生效，不需新代码包）：① 4 个 pip install 全部 `timeout` 限时 + `PIPESTATUS[0]` 显式判（**不能加全局 pipefail**——latest_ckpt 靠 `ls|sort|tail` 掩盖空 glob，fresh start 会挂；也不能靠 `| tail ||`——管道退出码是 tail 的）；② dl() 加 `--speed-time 60 --speed-limit 10240 --max-time 900`（CDN 坏窗口 0 字节挂连接对代码包下载同样致命）。hostprobe3 已带硬化 launcher 重发。
- 教训汇总：sj 侧脚本三要素——PYTHONPATH 只追加、管道退出码显式判、一切网络操作带超时/限速熔断。

## 2026-09-22 23:40 srcb7 bench 结果 + bundled torchair 发现

- **eager 基线（单卡 hero 架构，无优化器）**：bs32/tl64 0.451s/step peak 24.4GiB；bs16/tl512 0.570s/step peak 28.7GiB。
- 独立构建 torchair 7.2.0：import OK、block wrap OK，**compile 失败 BackendCompilerFailed**（bench 打印截断 200 字符丢了真错误——探针一律打印完整 traceback 的教训）。
- **关键发现**：compile 警告路径暴露 `torch_npu/dynamo/torchair/`——**torch_npu 2.9.0 wheel 自带捆绑版 torchair**（`from torch_npu.dynamo import torchair`），这才是与此栈版本匹配的后端；独立 7.2.0 构建疑似与 torch_npu 2.9.0 dynamo 钩子不兼容。ascend-ta-bench2 专测 bundled 后端 + 完整 traceback。
- 平台日志流今晚有严重滞后（结果早已打印但 logs 查询几十分钟后才可见）；watcher 以 job 状态为准，勿以日志即时性判断卡死（srcb7 就是虚惊）。

## 2026-09-23 00:30 torchair 全链路定位 + 0922k + launcher 自报崩溃

- **ta-bench3 三重结论**：① `import torchair` 顶层导入在 import torch_npu 后**自动别名到捆绑版**（pylibs/torch_npu/dynamo/torchair）——srcb7 的 BackendCompilerFailed 实为独立 7.2.0 构建 shadow 捆绑版导致；② pkg_resources 用干净目录 $W/pylibs-pkgres（仅 setuptools<81）提供即可，不要动 pylibs 主环境；③ 捆绑后端真正卡在 **`aten.isneginf` 无 GE converter**（dit_blocks.py sdpa_with_bias 的 drop-mask 检测，NPU 分支专属）。
- **0922k**：`torch.isneginf(bias)` → `bias == float("-inf")`（浮点语义等价，aten.eq.Scalar 有 converter）；launcher 的 TORCHAIR=1 路径从 pylibs-torchair 改挂 pylibs-pkgres（不再 shadow 捆绑版）。ta-bench4 验证全链路。
- **launcher 新增失败自报**：TRAIN_EXIT 非 0 时把训练日志尾 8KB 打进 job stdout——崩溃循环不再需要一个专职 logtail job 才能看到 traceback（hostprobe3 三连崩的教训）。hostprobe4 带此 launcher 重发三杠杆探针。
- hostprobe2 "pip 挂死 75 分钟"部分误判：日志滞后严重，它实际已走完 pip+代码包（hostprobe3 启动 77 秒即 LAUNCH 反证 .0922j.done 与 .code_ver=0922j 都已就位）。pip 超时硬化仍然正确，但节点判死刑要看平台 events 而非日志即时性。

## 09-22 22:35 hostprobe4 真凶 + 0922l + bench5

- **ta-bench4（torchair 全链路）两个 GE 级 blocker**：① 图中出现 `DT_COMPLEX64`——RoPE 的 `torch.view_as_complex`（apply_rotary_emb），GE 无复数支持；② `TensorMove` 算子不在 CANN op store（cann:9.0.1-910b 镜像 opp 包疑似裁剪），疑似为内部分形格式转换生成。
- **hostprobe4 三杠杆崩溃真凶**（TRAIN_LOG_TAIL 生效，但 8KB 尾被 tbe 关机噪声淹没；从完整日志 grep 到 rank1/rank15 真实异常）：`ValueError: set_to_none is not supported in fused optimizers`——NpuFusedAdamW 不支持 `zero_grad(set_to_none=True)`，首个 optimizer step 即崩。与 torch.npu.Event（0922j 修复对象）无关。
- **0922l 修复**：train.py 优化器循环 `opt.zero_grad(set_to_none=not args.ddp_gradient_bucket_views and not args.npu_fused_adamw)`。已上桥，launcher CODE_VER 默认 0922l。教训：**第三方 optimizer 替换要核对 zero_grad 契约**（fused 实现普遍只支持 set_to_none=False）。
- **bench5（已提交 ascend-ta-bench5）**：双管齐下重测 torchair per-block compile——`set_real_rope(model, True)`（apply_rotary_emb_real 无复数等价旋转，现成于 dit_blocks.py:184）+ `torch.npu.config.allow_internal_format = False`（禁内部格式 → 不产生 TensorMove）。同进程重测 eager-realrope 基线对照。
- **hostprobe5（已提交）**：同 hostprobe4 三杠杆配置，0922l 代码，验证 fused zero_grad 修复并拿 step_breakdown 段耗时。
- watcher bash-veeb003a 盯 bench5+hostprobe5+640p 下载。

## 09-22 23:00 bench5 判读 + 0922m 去复数改造 + bench6

- **bench5 判读**：`torch.npu.config.allow_internal_format` 在 torch_npu 2.9.0 **不存在**（AttributeError，缓解 2 未生效）；图里仍有 DT_COMPLEX64——`apply_rotary_emb_real` 内部 `view_as_real(freqs_cis)` 仍吃复数频率表，复数张量照样作为图输入跨界 → TensorMove(复数输入) 再次 EZ3002。eager-realrope 基线 0.522s/0.652s（比复数 eager 0.451/0.570 慢 ~15%，real rope 在 eager 下 op 更多，编译后才会融合回来）。
- **0922m 去复数改造**（根治频率表跨界）：新增 `apply_rotary_emb_realfreq`（直接吃 [S,D/2,2] 实数频率表）；`ArtFlow.forward` 在模型级（编译区外）`prepare_freqs` 后 `view_as_real` 一次物化实数表；`SingleStreamAttention` 在 real_rope 策略下用 realfreq 变体。double-stream 块保持自有复数 rope（eager 不管）。CPU 端到端 smoke：real vs complex 输出 **bitwise 一致**；tests/test_dit_blocks 12 passed（新增三变体等价性测试）。
- **bench6（已提交 ascend-ta-bench6）**：只编译 24 个 single-stream 块（double-stream 留 eager），realrope + 实数频率表。若全实数图仍撞 TensorMove → torchair 在此 CANN 镜像判死刑，eager 发 hero。
- 0922m 已上桥；launcher 仍指 0922l（hero 发射时若 torchair 可用则升 0922m + TORCHAIR 路径也要 set_real_rope——注意 train.py 侧还需接 real_rope 开关，bench6 绿后补）。

## 09-22 23:50 巡检

- hostprobe5 前两次 attempt（22:33/22:37 起跑）均 ~7 分钟 TRAIN_EXIT 1，平台日志无 Python traceback（rank 14 exitcode 1，尾 8KB 又是 tbe 噪声）；**第三次 attempt（22:41 起跑）已存活 69 分钟无第三次崩溃**，0922l fused zero_grad 修复大概率生效，待 TRAIN_EXIT 0 确认 + step_breakdown。
- ascend-logtail-hp5（拉 sj 侧 hero-256p.log 找前两次崩溃真异常）与 ascend-ta-bench6 均排队中（dl-640p + hostprobe5 占两个节点，availability 收紧）。
- 640p 下载 157/1126（CDN 限速窗口，~30 文件/小时）。

## 09-23 00:10 停 dl-640p 腾节点

- 用户决策：**停掉 ascend-dl-640p**（157/1126 已持久化在 sj-ssd3，hero 的 256p 阶段不需要 640p 数据），hero 发射后再续传。640p 对账相应顺延。
- 停job撞名坑：`inspire job stop` 遇到同名历史 job 要加 `--pick 1`（选最新=在跑的那个）。
- 腾出的节点立刻被 bench6 拿到（job_running 00:10）；logtail-hp5 仍排队，等 bench6 跑完。
- hostprobe5 第三次 attempt 存活 80+ 分钟：要么修复生效在训练，要么静默挂死——平台日志滞后无法区分，等 logtail-hp5 看 sj 侧 $W/logs/hero-256p.log 的 step 时间戳。

## 09-23 00:35 torchair 终局判死刑 + ascend-hero-256p 发射

- **bench6/7 终局**：0922m 去复数**根治了 TensorMove/DT_COMPLEX64**（错误不再出现），但暴露第三个独立 GE blocker：FX 图一个 `[0]` 空张量输出被 GE 物化为 `[0,0,0,0]`（NetOutput dim 不匹配）。bench7 排除 all-keep metadata 假设（带 padding 的 mask 同样失败）；从 FX 输出形状（[bs,16,320,8]/[bs,16,320,1]）推断是 **NPU 融合 attention 在 dropout=0 时返回的空辅助输出被 AOT 存进 backward 图**，修需动 sdpa_with_bias 快路径，迭代周期 20-90 分钟/次且成功率不确定。
- **torchair 死刑判据**（用户决策规则：收益 >20% 才上）：hostprobe5 cpu-wall 分解显示 **syncopt（梯度同步+优化器）~850ms/step 占 1.3s/step 大头**，fwd+bwd host 时间仅 ~175ms——torchair 只能压缩后者的一小段 dispatch 开销，对通信等待无能为力，乐观收益也远不到 20%。**三连 GE blocker + 收益预期不足 → 放弃 torchair，eager 发 hero。**
- **hostprobe5 完整结论**（0922l 三杠杆，300/300 步跑完，事后发现平台 job 卡在 shutdown 是 torchrun tbe 退出死锁，非训练挂死）：eval/loss 1.998→1.004 平滑下降；steady 710 samples/s（被 50 步一次 eval-loss + profiling 拉低，不可与 smoke8 的 860 直接比）；peak mem 52.7/53.3 GiB；**FUSED_ADAM+FOREACH 验证安全有效**。前两次 attempt 闪崩疑似 repo-ascend 共享目录竞态（bench5 同时在 rm -rf 重建代码目录）——并发 job 共用一个代码目录的坑，hero 独占不受影响。
- **launcher v2 修复**：新增 endpoint 看门狗（日志出现 "reached stage endpoint" 后 180s 优雅期，超时强杀进程组并 exit 0）——治 torchrun shutdown 挂死，hero 三个阶段切换都靠它。
- **00:35 发射 ascend-hero-256p**：eager + expandable_segments + FOREACH=1 FUSED_ADAM=1 + 0922l，T=600000（256p END=450k），fault-tolerance 200 次。启动 watcher bash-42yvgrjb（崩溃循环检测 + 45 分钟干净后 logtail 确认步数）。
- 吞吐估计：~860 samples/s（smoke8 值）→ 600k 步 × ~928 samples/step ≈ 180 小时 ≈ 7.5 天三阶段全量。

## 09-23 00:55 hero 换无杠杆配置重发（ascend-hero2-256p）

- **用户指正**：smoke8（无杠杆）~890 samples/s vs hostprobe5（带杠杆）表观 710——剔除 eval 探针阻塞（每 50 步 ~57s）后 hp5 干净吞吐 ≈ 927.8 samples/step ÷ 1.28s ≈ **~725 samples/s**，仍低 ~18%。差异来源未分清（cpu-wall 的逐 micro npu.Event 同步开销 vs 杠杆本身负收益），但 syncopt ~850ms 以 allreduce 等待为主、优化器 dispatch 本就不是瓶颈——杠杆打错靶子，**收益存疑的配置不应上 hero**。
- **决策**：hero 回到 smoke8 验证配置（eager + expandable_segments，无 FOREACH/FUSED_ADAM）。torchair/FUSED_ADAM/FOREACH 的 A/B 全部留到 hero 跑稳后的 checkpoint 重启窗口再做。
- **job 管理混乱记录**：① 第一次发射（00:20，带杠杆）后被平台重启过一次（00:25 出现第二条同名运行中记录，疑似 fault-tolerance retry 独立成行）；② 00:47 的 stop 因同名多 job 报错未生效，但同命令里的 create 却成功了（产生第三个同名 job，且后来 list 里找不到——幽灵创建）；③ 最终 `--pick 1/2` 停掉 00:20/00:25 两条带杠杆记录，改名 **ascend-hero2-256p** 重发（RUN_NAME 仍 ascend-hero-256p）。教训：**同名 job 会繁殖，发射就用唯一名字**。
- hero2 启动 watcher bash-ee9cy82p。

## 09-23 01:05 hero2 闪崩真凶（/dev/shm 64MB）+ 0922n + ascend-hero3-256p

- **hero2 4 分钟闪崩根因**：logtail job（ascend-logtail-hero2，占整机拉 sj 侧 `$W/logs/ascend-hero-256p.log`）挖到 DataLoader worker bus error——pod 的 **/dev/shm 只有 64MB**，worker 用默认 `file_system` 共享策略写 `/dev/shm/torch_*` 被撑爆（"No space left on device (28)"）。这是掷硬币级潜在 bug：smoke8 / hostprobe5-attempt3 成功纯属运气，hp5 attempt 1/2、带杠杆 hero（00:20）、hero2（00:28）全死于同一原因——此前归因给 repo-ascend 竞态是误判（竞态可能仍是次要因素，但主凶是 shm）。
- **0922n 修复**：train.py 新增 `--dataloader_sharing_strategy`（choices file_system/file_descriptor，default None = 行为不变，4090 路径不受影响），main() 在 DataLoader spawn 前 `mp.set_sharing_strategy()`；launcher CODE_VER 默认升 0922n，torchrun 前 `ulimit -n 1048576`（file_descriptor 策略每 worker 每 batch 耗几个 fd），torchrun 命令加 `--dataloader_sharing_strategy file_descriptor`。验证：`bash -n` launcher + `ast.parse` train.py + `--help` 确认 argparse + pytest tests/test_dit_blocks 12 passed。
- **01:05 发射 ascend-hero3-256p**（唯一名字，RUN_NAME 仍 ascend-hero-256p）：eager + expandable_segments + 0922n + file_descriptor，T=600000，fault-tolerance 200。吞吐校验窗口 = 头 2000 步，预期 ~890 samples/s（smoke8 无杠杆水平）；若仍 ~725 说明瓶颈在 0922l 代码/环境而非杠杆。启动 watcher bash-18xbc45a。
- 教训沉淀：① **训练 stdout 不进平台日志，崩溃定位必须 logtail job 或 launcher TRAIN_LOG_TAIL**；② ascend pod /dev/shm=64MB 是平台固定值，file_system 策略不可用——launcher 里的 `rm -f /dev/shm/torch_*` 清理只能治标；③ 同名 job 繁殖问题再次出现（hero2 stop 后创建 hero3 无冲突）。

## 09-23 01:10 hero3 attempt 1 闪崩（原因未明）+ 诊断版 launcher

- **hero3 attempt 1**（00:41 起跑，0922n 确认部署 DL_CODE_OK）：训练 ~4 分钟后 rank 3 exit 1（00:49:32），elastic `error_file: <N/A>`，TRAIN_LOG_TAIL 8KB 全是 tbe 拆除噪声，**真实 traceback 未捕获**。shm ENOSPC 是掷硬币（smoke8/hp5-attempt3 无修复也跑通），所以这次崩溃既不能证明 file_descriptor 修复无效，attempt 2 存活也不能证明有效——需要真实 traceback 才能定论。
- **诊断版 launcher**（本地已改，未上桥——随下次重发部署）：① torchrun 加 `--log-dir $W/logs/elastic`，崩溃时把最fresh 的 error.json（含失败 rank 完整 py_callstack）打进平台日志；② 新增 TRAIN_FIRST_TRACEBACK：以 launcher 启动时的字节游标切出本次 attempt 的日志区间，打印首个 "Traceback (most recent call last)" 起 60 行（首个 traceback 时序上几乎必是肇事 rank，先于拆除噪声）；③ TRAIN_FATAL_LINES grep（bus error/ENOSPC/RuntimeError/OOM 等）；④ TRAIN_LOG_TAIL 缩到 4KB 兜底。**此后崩溃自报 traceback，不再需要 logtail job 占整机节点**。
- **自动接力 watcher**（/tmp/watch_hero3_relaunch.sh）：以 watcher 启动时的 TRAIN_EXIT 1 计数为基线，检测到新崩溃即停 hero3、以唯一名 **ascend-hero4-256p** 重发（诊断版 launcher）；45 分钟干净则退出。
- attempt 2（00:50:48 起跑）写作时已存活 ~20 分钟（过了 4 分钟崩点），暂稳。
- 另：inko-patrol（qb-ilm 存储）**看不到 sj-ssd3**——sj 侧日志只有 ascend job 能读，这是诊断必须内建进 launcher 的根本原因。
