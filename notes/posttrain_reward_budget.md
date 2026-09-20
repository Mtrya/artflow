# Stage 6 reward budget（VLM judge 侧成本估算）

更新于 2026-09-21。定价按 ZenMux gemini-2.5-flash-lite 档当前单价填（见下），
价格变动只需改 `PRICE_PER_CALL` 一行。

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
