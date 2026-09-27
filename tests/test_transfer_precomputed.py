from collections import Counter
from pathlib import Path
from types import ModuleType
import sys

import pytest

from scripts.data import transfer_precomputed as transfer
from src.pretrain.config import load_config


def test_upload_set_follows_selected_recipe_and_includes_eval(tmp_path):
    config_path = "configs/hero.toml"
    stage = load_config(config_path).stage("896p")
    expected = {Path(entry.path).name for entry in stage.datasets}
    expected.add(Path(stage.eval_dataset_path).name)
    for name in expected:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "state.json").write_text("{}")
    paths = transfer.stage_directories(config_path, "896p", tmp_path)
    assert {p.name for p in paths} == expected
    assert all(p.parent == tmp_path for p in paths)
    (tmp_path / Path(stage.eval_dataset_path).name / "state.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing precomputed"):
        transfer.stage_directories(config_path, "896p", tmp_path)


def test_hf_upload_retries_and_reports_exhausted_failures(monkeypatch, tmp_path):
    import huggingface_hub

    attempts = Counter()
    def upload_folder(self, **kwargs):
        name = kwargs["path_in_repo"]
        assert kwargs["folder_path"] == str(tmp_path / name)
        assert kwargs["repo_id"] == "owner/precomputed"
        attempts[name] += 1
        if name == "bad" or (name == "retry" and attempts[name] < 2):
            raise ConnectionError("test failure")
    monkeypatch.setattr(huggingface_hub, "HfApi", type("Api", (), {"upload_folder": upload_folder}))
    monkeypatch.setattr(transfer.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="bad"):
        transfer.upload("hf", "owner/precomputed",
                        [tmp_path / name for name in ("good", "retry", "bad")])
    assert attempts == {"good": 1, "retry": 2, "bad": 3}


def test_modelscope_credentials_and_provider_calls(monkeypatch, tmp_path, capsys):
    calls = []
    token_path = tmp_path / "token"
    token_path.write_text("test-token-from-file\n")
    monkeypatch.delenv("MS_TOKEN", raising=False)
    monkeypatch.setenv("MODELSCOPE_TOKEN_PATH", str(token_path))
    modules = {name: ModuleType(name) for name in
               ("modelscope", "modelscope.hub", "modelscope.hub.api")}
    class Api:
        def __init__(self, token):
            assert token == "test-token-from-file"
        def upload_folder(self, **kwargs):
            calls.append(kwargs)
    modules["modelscope.hub.api"].HubApi = Api
    modules["modelscope.hub"].snapshot_download = lambda **kw: calls.append(kw)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    transfer.upload("modelscope", "owner/data", [tmp_path / "d1@896p"])
    transfer.download("modelscope", "owner/data", tmp_path / "download")
    assert calls[0]["path_in_repo"] == "d1@896p"
    assert calls[0]["use_cache"] is True
    assert calls[1]["local_dir"] == str(tmp_path / "download")
    assert calls[1]["token"] == "test-token-from-file"
    assert "test-token-from-file" not in capsys.readouterr().out


def test_missing_modelscope_credentials_fail(monkeypatch):
    monkeypatch.delenv("MS_TOKEN", raising=False)
    monkeypatch.delenv("MODELSCOPE_TOKEN_PATH", raising=False)
    with pytest.raises(ValueError, match="MS_TOKEN"):
        transfer.modelscope_token()
