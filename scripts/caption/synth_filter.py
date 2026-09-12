"""Screen a synthetic-image manifest for fatal generation flaws, image by image.

The synthetic corpora are generated, not harvested, so a row can be broken in
ways no caption can repair: fused fingers, two faces melted into one, a back
view wearing clothes drawn for the front, garbled pseudo-text.  Captioning such
a row spends API money describing something that should not be in the dataset
at all, so this pass asks a vision model for a per-image verdict first and only
the rows that pass go on to the caption pass.

The request is deliberately blind to the generation prompt that produced the
image: the verdict must follow the pixels, not the request they came from.

Verdicts are cached under the same request key as captions (image bytes, model,
prompt text, settings), so a rerun is free for rows already judged and a
changed prompt cannot reuse an old answer.

CLI:
    python -m scripts.caption.synth_filter \
        --selection pilot.jsonl --index thumbs/index.jsonl \
        --out verdicts.jsonl --cache-dir ~/.cache/artflow_filter \
        --model google/gemini-2.5-flash-lite --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.dataset.caption_client import CaptionClient, summarise
from scripts.caption.generate import load_done, load_index

PROMPT_VERSION = "synth-filter-v6"

# The verdict's vocabulary.  Counting flaws by these ids is what makes the
# batch report readable; the prompt describes each one in Chinese.  Repetition
# goes beyond the six labels that judge a single figure: nine identical people
# read as a plausible group portrait to them, while a cloned figure is exactly
# the kind of generation failure this pass exists to catch.  Intrusion and
# frame catch what a subject-centred check misses on real photographs —
# softboxes and light stands rendered into the scene, a photographer's hand or
# phone reaching in from the edge, pictures nested inside book pages, hanging
# scrolls or fan borders — defects an earlier label set let pass.
FLAW_LABELS = ("anatomy", "orientation", "fusion", "text", "garment", "quality",
               "repetition", "intrusion", "frame")

FILTER_PROMPT = """你在为文生图训练数据做质检。随消息附上的是一张由文生图模型生成的人像 / 传统服饰场景图（东亚题材）。\
请判断它是否存在"根本性缺陷"——严重到整张图应当丢弃、无法靠裁切或轻微修图补救的结构性错误。\
风格平淡、构图普通、细节简单、AI 图像常见的轻微光滑感，都不算缺陷。

先逐条回答下面七个问题，每条都要给出画面中的具体依据（说清在什么位置看到了什么），不要只写"无"：
A. 画面里有几个人？每个人的脸、双手、双臂、腿是否完整、数量正常、连接自然？手指有没有多、缺、粘连成团？（手被遮挡或出画就写明"被遮挡/出画"）
B. 每个人是正面、侧面还是背面朝向镜头？他的衣领、衣襟、盘扣方向与这个朝向一致吗？身体有没有扭到关节不可能的角度？
C. 人物之间、人物与背景之间有没有互相融合、穿插，或者半截无端消失的地方？
D. 画面里有文字或印章吗？如果有：是能辨认的正常书法、题字、印章，还是乱码伪文字、歪扭字符、糊块？
E. 画面有没有严重模糊或涂抹感、明显的拼接缝或粘贴块、重复平铺（tiling）的图案？
F. 画面里有没有两个以上几乎一模一样的人物——同一张脸、同一套衣服、同一个姿势反复出现？背景中模糊路人的相似不算，但同一个清晰形象被复制粘贴式地重复出现属于生成 artifact。
G. 画面里有没有不该出现在场景里的东西？逐项检查：\
（1）摄影棚器材入镜：柔光箱、灯架、反光板、摄影灯等出现在画面里（场景设定为摄影棚内部时除外）；\
（2）拍摄者穿帮：从画面边缘伸入的手、手机、相机或自拍杆，属于拍照的人而不是画面中的人物（画面人物自己拿着手机拍照不算）；\
（3）框中框与边框：整张画面被嵌在画框、翻开的书页或画册内页、挂轴装裱、扇面、手机屏幕等边框结构里，或者画面四周带有一圈整齐的装饰边框或白边——合成图应当就是场景本身，装裱边框和"画中画"都属于生成 artifact。

然后给结论。判定标准：上面任何一条出现明显错误，ok 即为 false；只描述画面中真实看得到的问题，不要因为题材普通、笔触简单或"看起来像 AI 生成"就判缺陷。

最后输出一个 JSON 对象，它必须是回复的最后一行：
{"ok": true, "flaws": [], "reason": "一句话中文说明"}
其中 flaws 只能取九个英文标识：anatomy（解剖错误）、orientation（朝向矛盾）、fusion（人物融合/残缺）、text（文字 artifact）、garment（服饰结构不可能）、quality（图像级问题）、repetition（同一人物被复制重复）、intrusion（摄影器材或拍摄者穿帮入镜）、frame（框中框或装裱边框，画面嵌在画框/书页/挂轴/扇面/屏幕边框内，或四周有整齐边框），可以多个；ok 为 true 时 flaws 必须是空数组；reason 用一句简短中文概括主要依据。"""


def parse_verdict(text: str) -> Tuple[Optional[bool], List[str], str, Optional[str]]:
    """Pull the verdict out of an answer, tolerating text around the JSON.

    Returns ``(ok, flaws, reason, parse_error)``.  ``ok`` is None when no
    usable JSON object was found, which the caller records as a failed row
    rather than guessing a verdict.
    """
    payload = None
    decoder = json.JSONDecoder()
    body = (text or "").strip()
    for position in range(len(body) - 1, -1, -1):
        if body[position] != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(body[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "ok" in candidate:
            payload = candidate
            break
    if payload is None:
        return None, [], "", "no JSON object with an 'ok' field"
    ok = payload.get("ok")
    if isinstance(ok, str):
        ok = ok.strip().lower() in {"true", "yes", "是", "通过"}
    if not isinstance(ok, bool):
        return None, [], str(payload.get("reason") or ""), f"'ok' is not a boolean: {payload.get('ok')!r}"
    flaws = payload.get("flaws") or []
    if isinstance(flaws, str):
        flaws = [flaws]
    flaws = [str(flaw).strip().lower() for flaw in flaws if str(flaw).strip()]
    reason = str(payload.get("reason") or "").strip()
    return ok, flaws, reason, None


async def run(args) -> None:
    selection = [json.loads(line) for line in Path(args.selection).open(encoding="utf-8")
                 if line.strip()]
    index = load_index(args.index, args.thumb_dir)
    done = load_done(args.out)
    rows = [row for row in selection
            if row["image_id"] in index
            and (row["image_id"], args.model) not in done
            and os.path.exists(index[row["image_id"]]["thumb_path"])]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("nothing to do")
        return

    client = CaptionClient(args.cache_dir, args.model, concurrency=args.concurrency,
                           timeout=args.timeout)
    print(f"{len(rows)} images x {args.model} (pricing {client.pricing_record()}, "
          f"concurrency {args.concurrency}); {len(done)} rows already judged")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    responses: List[Any] = []
    start = time.time()
    with out_path.open("a", encoding="utf-8") as sink:
        async with httpx.AsyncClient() as http:
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(row: Dict) -> Tuple[Any, Dict]:
                async with semaphore:
                    data = Path(index[row["image_id"]]["thumb_path"]).read_bytes()
                    return await client.generate(
                        http, image_bytes=data, prompt=FILTER_PROMPT,
                        prompt_version=PROMPT_VERSION, max_tokens=args.max_tokens,
                        temperature=args.temperature), row

            for wave_start in range(0, len(rows), args.batch_size):
                wave = rows[wave_start:wave_start + args.batch_size]
                tasks = [asyncio.create_task(one(row)) for row in wave]
                for coro in asyncio.as_completed(tasks):
                    response, row = await coro
                    ok, flaws, reason, parse_error = parse_verdict(response.text)
                    record = {
                        "image_id": row["image_id"],
                        "model": args.model,
                        "prompt_version": PROMPT_VERSION,
                        "ok": ok,
                        "flaws": flaws,
                        "reason": reason,
                        "parse_error": parse_error,
                        "raw": response.text,
                        "usage": response.usage,
                        "cost_usd": response.cost_usd,
                        "latency_s": response.latency_s,
                        "cached": response.cached,
                        "error": response.error,
                        "finish_reason": response.finish_reason,
                        "request_key": response.request_key,
                        "image_fingerprint": response.image_fingerprint,
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    responses.append((record, response))
                    finished = len(responses)
                    if finished % 10 == 0:
                        sink.flush()
                        rate = finished / max(time.time() - start, 1e-9)
                        print(f"  {finished}/{len(rows)}  {rate:.1f} req/s", flush=True)
        sink.flush()

    report([record for record, _ in responses], [response for _, response in responses], out_path)


def report(records: List[Dict], responses: List[Any], out_path: Path) -> None:
    judged = [record for record in records if record["ok"] is not None]
    ok = sum(1 for record in judged if record["ok"])
    print(f"\nwrote {out_path}")
    print(f"{len(judged)}/{len(records)} judged, {ok} ok ({ok / max(len(judged), 1):.1%}), "
          f"{len(judged) - ok} flawed, {len(records) - len(judged)} unparsed")
    flaws = Counter(flaw for record in judged for flaw in record["flaws"])
    if flaws:
        print("flaws:", dict(flaws.most_common()))
    unknown = sorted(set(flaws) - set(FLAW_LABELS))
    if unknown:
        print("flaw labels outside the taxonomy:", unknown)
    summary = summarise(responses)
    print("usage:", json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--index", required=True, help="thumbnail index JSONL")
    parser.add_argument("--thumb-dir", default=None,
                        help="directory holding the thumbnails, if they were copied "
                             "away from the paths recorded in the index")
    parser.add_argument("--out", required=True, help="output verdict JSONL (appended)")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/artflow_filter"))
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="images per wave; bounds peak memory from encoded images")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
