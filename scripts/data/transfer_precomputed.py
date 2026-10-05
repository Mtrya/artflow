"""Transfer precomputed datasets through Hugging Face or ModelScope.

Upload takes the training and evaluation directories from a run config's stage.
Dataset paths are resolved under the explicit --storage-root.
Download places a repository snapshot under <storage-root>/precomputed_dataset. Repositories must
already exist with the intended visibility.

Credentials: HF_TOKEN for Hugging Face; MS_TOKEN or MODELSCOPE_TOKEN_PATH
(a file containing the token) for ModelScope. Install the selected provider's
SDK in the transfer environment.

    python -m scripts.data.transfer_precomputed upload --provider hf \
        --repo-id OWNER/DATASET --config configs/hero.toml --stage 896p --storage-root /external/artflow
    python -m scripts.data.transfer_precomputed download --provider modelscope \
        --repo-id OWNER/DATASET --storage-root /external/artflow
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def stage_directories(config_path: str, stage_name: str,
                      storage_root: Path) -> list[Path]:
    from src.pretrain.config import load_config

    stage = load_config(config_path, storage_root=storage_root).stage(stage_name)
    directories = {}
    for value in [entry.path for entry in stage.datasets] + [stage.eval_dataset_path]:
        original = Path(value)
        if original.name in directories and directories[original.name] != original:
            raise ValueError(f"different directories share the name {original.name!r}")
        directories[original.name] = original
    paths = list(directories.values())
    missing = [str(path) for path in paths if not (path / "state.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing precomputed datasets: {missing}")
    return paths


def modelscope_token() -> str:
    token = os.environ.get("MS_TOKEN", "").strip()
    if not token and os.environ.get("MODELSCOPE_TOKEN_PATH"):
        token = Path(os.environ["MODELSCOPE_TOKEN_PATH"]).read_text().strip()
    if not token:
        raise ValueError("set MS_TOKEN or MODELSCOPE_TOKEN_PATH")
    return token


def upload(provider: str, repo_id: str, paths: list[Path]) -> None:
    if provider == "hf":
        from huggingface_hub import HfApi

        api, extra, workers = HfApi(), {}, 6
    else:
        from modelscope.hub.api import HubApi

        token = modelscope_token()
        api = HubApi(token=token)
        extra = dict(max_workers=8, use_cache=True, disable_tqdm=True, token=token)
        workers = 1  # ModelScope parallelizes files within upload_folder.

    def upload_one(path):
        for attempt in range(1, 4):
            try:
                api.upload_folder(repo_id=repo_id, repo_type="dataset",
                                  folder_path=str(path), path_in_repo=path.name, **extra)
                print(f"DONE {path.name}", flush=True)
                return
            except Exception:
                if attempt == 3:
                    raise
                print(f"RETRY {path.name} attempt={attempt}", flush=True)
                time.sleep(20 * attempt)

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(upload_one, path): path for path in paths}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failed.append((futures[future].name, error))
    if failed:
        raise RuntimeError(f"failed uploads: {[name for name, _ in failed]}") from failed[0][1]


def download(provider: str, repo_id: str, data_root: Path) -> None:
    if provider == "hf":
        from huggingface_hub import snapshot_download

        extra = {}
    else:
        from modelscope.hub import snapshot_download

        extra = {"token": modelscope_token()}
    data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(data_root),
                      max_workers=8, **extra)
    print(f"downloaded {repo_id} -> {data_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "download"))
    parser.add_argument("--provider", choices=("hf", "modelscope"), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--config", help="complete run TOML for upload")
    parser.add_argument("--stage", help="stage to upload, including its evaluation set")
    args = parser.parse_args()
    if args.action == "upload":
        if not args.config or not args.stage:
            parser.error("upload requires --config and --stage")
        paths = stage_directories(args.config, args.stage, args.storage_root)
        upload(args.provider, args.repo_id, paths)
    else:
        if args.config or args.stage:
            parser.error("download selects a repository; --config/--stage apply to upload")
        download(args.provider, args.repo_id, args.storage_root / "precomputed_dataset")


if __name__ == "__main__":
    main()
