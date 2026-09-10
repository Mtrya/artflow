"""Generate long captions for a selection manifest through an API provider.

Reads the selection manifest (what to ask for) and the thumbnail index (where
the image is), sends each request, checks the answer against the non-visual
acceptance rules, and appends one JSONL record per request.  Re-running skips
rows already present in the output, and the client's own cache means an
identical request is never paid for twice.

CLI:
    python -m scripts.caption.generate \
        --selection selection.jsonl --index thumbs/index.jsonl \
        --out captions.jsonl --cache-dir ~/.cache/artflow_caption \
        --models google/gemini-3.5-flash-lite --concurrency 16 --limit 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from src.dataset.caption_client import CaptionClient, summarise
from src.dataset.caption_prompts import (
    CaptionRequest, check_caption, metadata_for_caption, normalize_caption,
)

# Output budget per length band, in the model's own tokens.  Measured on this
# corpus: English runs ~1.32 tokens per word, so the longest band needs room
# for ~950 words plus any reasoning tokens the provider bills.
MAX_TOKENS_BY_BAND = {
    "64-255": 400,
    "256-511": 900,
    "512-895": 1600,
    "896-1280": 2400,
}


def load_index(path: str, thumb_dir: Optional[str] = None) -> Dict[str, Dict]:
    index = {}
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error"):
            continue
        if thumb_dir:
            record["thumb_path"] = os.path.join(thumb_dir, os.path.basename(record["thumb_path"]))
        index[record["image_id"]] = record
    return index


def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add((record["image_id"], record["model"]))
    return done


def build_requests(rows: List[Dict], options: Dict[str, Dict]) -> List[Dict]:
    """Build one request per row, with per-band overrides where they exist.

    ``options`` maps a band to its settings, with ``"*"`` holding the model
    defaults.  A model can need a different length target on different bands:
    the same model asked for 256 tokens and for 1,200 tokens has a different
    systematic bias, and the target is the only knob that corrects it.
    """
    jobs = []
    for row in rows:
        settings = options.get(row["band"], options.get("*", {}))
        # Verified metadata the caption may use.  The prompt says it is fine to
        # leave the artist and title out, so this widens what the model may say
        # without forcing a catalogue voice.
        metadata = metadata_for_caption(row.get("artist"), row.get("title"),
                                        row["language"])
        request = CaptionRequest(
            image_id=row["image_id"],
            language=row["language"],
            format=row["format"],
            band=row["band"],
            domain=row["domain"],
            metadata=metadata,
            target_position=settings.get("target_position", 1.0 / 3.0),
        )
        jobs.append({
            "row": row,
            "request": request,
            "max_tokens": int(MAX_TOKENS_BY_BAND[request.band]
                              * settings.get("max_tokens_scale", 1.0)),
        })
    return jobs


async def run(args) -> None:
    selection = [json.loads(line) for line in Path(args.selection).open(encoding="utf-8")
                 if line.strip()]
    index = load_index(args.index, args.thumb_dir)
    model_config = json.loads(Path(args.model_config).read_text()) if args.model_config else {}
    done = load_done(args.out)

    # Thumbnails arrive in chunks, so a row whose image has not landed yet is
    # left for a later pass instead of failing the run.
    def ready(row: Dict) -> bool:
        return os.path.exists(index[row["image_id"]]["thumb_path"])

    # Either one list of models to compare on every row, or a per-band routing
    # table, which is how production runs: each band goes to the model that
    # measured best on it.
    if args.routing:
        routing = json.loads(Path(args.routing).read_text())
        by_model: Dict[str, List[Dict]] = {}
        for row in selection:
            if row["image_id"] not in index or not ready(row):
                continue
            model = routing.get(row["band"])
            if model is None:
                continue
            if (row["image_id"], model) in done:
                continue
            by_model.setdefault(model, []).append(row)
        if args.limit:
            by_model = {model: rows[:args.limit] for model, rows in by_model.items()}
        if not by_model:
            print("nothing to do")
            return
        print("routing: " + ", ".join(
            f"{model} <- {sum(1 for r in selection if routing.get(r['band']) == model)} rows"
            for model in by_model))
    else:
        models = args.models.split(",")
        rows = []
        for row in selection:
            if row["image_id"] not in index or not ready(row):
                continue
            if args.bands and row["band"] not in set(args.bands.split(",")):
                continue
            if args.languages and row["language"] not in set(args.languages.split(",")):
                continue
            if all((row["image_id"], model) in done for model in models):
                continue
            rows.append(row)
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            print("nothing to do")
            return
        by_model = {model: rows for model in models}
        print(f"{len(rows)} rows x {len(models)} model(s); {len(done)} (row, model) pairs already done")

    tokenizer = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_responses = []
    for model, rows in by_model.items():
        options = dict(model_config.get(model, {}))
        extra = options.pop("extra", {})
        per_band = options.pop("bands", {})
        defaults = {
            "max_tokens_scale": float(options.pop("max_tokens_scale", args.max_tokens_scale)),
            "target_position": float(options.pop("target_position", 1.0 / 3.0)),
        }
        options_by_band = {"*": defaults}
        for band, overrides in per_band.items():
            merged = dict(defaults)
            merged.update(overrides)
            options_by_band[band] = merged
        client = CaptionClient(args.cache_dir, model, concurrency=args.concurrency,
                              timeout=args.timeout)
        print(f"\n=== {model} (pricing {client.pricing_record()}, concurrency {args.concurrency}, "
              f"extra {extra or '{}'})")
        jobs = build_requests(rows, options_by_band)
        finished = 0
        start = time.time()
        with out_path.open("a", encoding="utf-8") as sink:
            async with httpx.AsyncClient() as http:
                semaphore = asyncio.Semaphore(args.concurrency)

                async def one(payload):
                    async with semaphore:
                        return await client.generate(
                            http, image_bytes=payload["image_bytes"], prompt=payload["prompt"],
                            prompt_version=payload["prompt_version"],
                            max_tokens=payload["max_tokens"], temperature=payload["temperature"],
                            extra=payload["extra"]), payload["job"]

                # Images are read one wave at a time.  Holding every encoded
                # image at once would be gigabytes for a full production run.
                for wave_start in range(0, len(jobs), args.batch_size):
                    wave = jobs[wave_start:wave_start + args.batch_size]
                    payloads = []
                    for job in wave:
                        image_id = job["row"]["image_id"]
                        data = Path(index[image_id]["thumb_path"]).read_bytes()
                        settings = {"max_tokens": job["max_tokens"],
                                    "temperature": args.temperature, **extra}
                        key = client.request_key(data, job["request"].prompt(),
                                                 job["request"].prompt_version, settings)
                        job["request_key"] = key
                        payloads.append({
                            "job": job,
                            "image_bytes": data,
                            "prompt": job["request"].prompt(),
                            "prompt_version": job["request"].prompt_version,
                            "max_tokens": job["max_tokens"],
                            "temperature": args.temperature,
                            "extra": extra or None,
                        })

                    tasks = [asyncio.create_task(one(payload)) for payload in payloads]
                    for coro in asyncio.as_completed(tasks):
                        response, job = await coro
                        request = job["request"]
                        normalized = normalize_caption(response.text)
                        check = check_caption(normalized, request, tokenizer)
                        record = {
                            "image_id": request.image_id,
                            "source": job["row"]["source"],
                            "domain": request.domain,
                            "model": model,
                            "language": request.language,
                            "format": request.format,
                            "band": request.band,
                            "target_tokens": request.target_tokens,
                            "prompt_version": request.prompt_version,
                            "text": normalized,
                            "retained_tokens": check.retained_tokens,
                            "in_band": check.in_band,
                            "accepted": check.ok,
                            "reject_reasons": check.reasons,
                            "evaluation_hits": check.evaluation_hits,
                            "hedging_hits": check.hedging_hits,
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
                        all_responses.append((record, response))
                        finished += 1
                        if finished % 25 == 0:
                            sink.flush()
                            rate = finished / max(time.time() - start, 1e-9)
                            print(f"  {finished}/{len(jobs)}  {rate:.1f} req/s", flush=True)
        summary = summarise([r for record, r in all_responses if record["model"] == model])
        print(f"  {model}: {json.dumps(summary)}")

    report(all_responses, out_path)


def report(all_responses, out_path: Path) -> None:
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for record, _ in all_responses:
        by_model[record["model"]].append(record)
    print(f"\nwrote {out_path}")
    print(f"{'model':>34s} {'n':>6s} {'accept':>7s} {'in-band':>8s} {'mean_tok':>9s} {'cost':>9s}")
    for model, records in by_model.items():
        accepted = sum(1 for r in records if r["accepted"])
        in_band = sum(1 for r in records if r["in_band"])
        mean_tokens = sum(r["retained_tokens"] for r in records) / len(records)
        cost = sum(r["cost_usd"] for r in records if not r["cached"])
        print(f"{model:>34s} {len(records):6d} {accepted / len(records):6.1%} "
              f"{in_band / len(records):7.1%} {mean_tokens:9.0f} {cost:8.4f}$")
    reasons = Counter(reason.split(":")[0] for _, records in by_model.items()
                      for r in records for reason in r["reject_reasons"])
    if reasons:
        print("reject reasons:", dict(reasons.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--index", required=True, help="thumbnail index JSONL")
    parser.add_argument("--thumb-dir", default=None,
                        help="directory holding the thumbnails, if they were copied "
                             "away from the paths recorded in the index")
    parser.add_argument("--out", required=True, help="output caption JSONL (appended)")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/artflow_caption"))
    parser.add_argument("--models", default="google/gemini-3.5-flash-lite")
    parser.add_argument("--routing", default=None,
                        help='JSON file mapping length band to model id; overrides --models')
    parser.add_argument("--model-config", default=None,
                        help='JSON file mapping model id to {"extra": {...}, '
                             '"max_tokens_scale": float, "target_position": float, '
                             '"bands": {band: {same keys}}}')
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="rows per wave; bounds peak memory from decoded images")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens-scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bands", default=None, help="comma separated length bands to keep")
    parser.add_argument("--languages", default=None, help="comma separated languages to keep")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="skip token counting; lengths are measured later "
                             "by a pass that has the tokenizer")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
