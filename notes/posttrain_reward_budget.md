# Stage 6 reward budget（VLM judge 侧成本估算）

更新于 2026-09-21。**成本前提已变化**：学院自部署 VLM（Qwen3.8-27B，SII 内网，
endpoint/key 存于 GPFS `$W/secrets/sii_vlm.json`）实测可用（8 并发 7.7s 全成功，
支持 image_url，`enable_thinking:false` 直达输出）。**API 调用费用≈0**，不再需要
按次计费；约束变为吞吐与共享资源礼貌使用（并发 ≤16–32 + 退避）。ZenMux 仅作
 fallback（校外、judge 全挂、或需第二意见时）。原文的按次计费模型保留作 fallback 参考。

## 每次 RL 迭代的打分量

NFT/ReFL 结构：**每迭代打分图片数 = prompts/iter × G**。

| 配置 | prompts/iter | G | 图/迭代 |
|---|---:|---:|---:|
| 初始（省钱包） | 48 | 16 | 768 |
| NFT 默认 | 48 | 24 | 1,152 |

每图 1 次 VLM judge 调用（rubric 多维一次输出；若拆多调用则乘倍数），
加 ~10% 重试/解析失败冗余。

## 总成本公式

```
calls      = iterations × prompts_per_iter × G × (1 + retry_margin)
cost_usd   = calls × PRICE_PER_CALL
```

## 场景估算（PRICE_PER_CALL 留空待填）

| 场景 | 迭代数 | 配置 | 图总数 | 备注 |
|---|---:|---|---:|---|
| VLM judge probe（640p ckpt 后） | 1 轮 × 5k 池 | 全 5k 池每图 1 次 | ~5.5k | 含思考开关对比 ×2 |
| G 两档排序一致性 | 2 轮 × 500 prompts | 48 × {16,24} | ~39k | probe 顺手做 |
| 算法消融（ReFL vs NFT） | 2 × 2k iter | 48×16 | ~337k | RL loss 与算法无关的量（λ_rl 不另花） |
| G 消融并入上 | 已含 | | | 不额外迭代 |
| 主联合 run（规模未定，变量 J） | J | 48×16 | 768×J | 最大单项，J 待 hero 后定 |

**结论**：除主联合 run 外，消融+probe 总量约 0.4M 次调用。把当前单价代入
`0.4M × PRICE_PER_CALL` 即消融打标总预算；主 run 同理按 768×J 算。
若账单吃紧，第一调节旋钮是 G（成本线性），第二是 prompts/iter。
