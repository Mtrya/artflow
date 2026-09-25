import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.pretrain.migrate_conditioning_decay import (
    check_recipe_change, migrate, sha256, split_adam_state,
)
from src.models.artflow import ArtFlow
from src.pretrain.config import load_config
from src.pretrain.muon import CONDITIONING_WEIGHTS, build_param_groups
from src.pretrain.stage_control import validate_checkpoint, write_checkpoint_record
from src.pretrain.state_verification import require_exact_state
from src.pretrain.train import build_linear_cosine_scheduler


def model_and_optimizers():
    model = ArtFlow(hidden_size=64, num_heads=4, double_stream_depth=1,
                    single_stream_depth=1, mlp_ratio=2.0)
    optimizers = build_param_groups(
        model, muon_lr=.02, muon_wd=.0015, adam_lr=1e-4,
        adam_wd=.01, adam_conditioning_wd=.4, adam_eps=1e-8,
        adam_betas=(.9, .95), muon_momentum=.95,
    )
    adam_ids = {id(p) for g in optimizers[1].param_groups for p in g['params']}
    historical = torch.optim.AdamW(
        [p for p in model.parameters() if id(p) in adam_ids],
        lr=1e-4, weight_decay=.01, betas=(.9, .95), eps=1e-8,
    )
    return model, optimizers, historical


def schedule(optimizer):
    return build_linear_cosine_scheduler(
        optimizer, num_warmup_steps=20000, num_training_steps=600000,
        min_learning_rate=5e-6, base_learning_rate=1e-4,
        start_learning_rate=1e-5,
    )


def test_native_group_routes_only_conditioning_matrices():
    model, (_, adam), _ = model_and_optimizers()
    names = {id(p): n for n, p in model.named_parameters()}
    assert {names[id(p)] for p in adam.param_groups[1]['params']} == CONDITIONING_WEIGHTS
    assert [g['weight_decay'] for g in adam.param_groups] == [.01, .4]
    assert {'c_mlp.0.bias', 'c_mlp.2.bias', 'txt_pooled_proj.bias'} <= {
        names[id(p)] for p in adam.param_groups[0]['params']
    }


def test_split_preserves_moments_schedule_and_matches_targeted_next_update():
    torch.manual_seed(19)
    model, (_, target), old = model_and_optimizers()
    old_scheduler, target_scheduler = schedule(old), schedule(target)
    for _ in range(3):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        old.step()
        old_scheduler.step()
    saved = copy.deepcopy(old.state_dict())
    state, scheduler, mapping = split_adam_state(model, target, saved, old_scheduler.state_dict())
    target.load_state_dict(state)
    target_scheduler.load_state_dict(scheduler)
    for ids in mapping.values():
        require_exact_state(saved['state'][ids['old_id']], target.state_dict()['state'][ids['new_id']], label='moments')

    # Independent reference: the qualified experimental pre-hook composes
    # decay on just these matrices before the historical AdamW update.
    reference = copy.deepcopy(model)
    by_name = dict(model.named_parameters())
    ref_names = dict(reference.named_parameters())
    # mapping follows new group order; historical IDs follow model order.
    old_names = sorted(mapping, key=lambda name: mapping[name]['old_id'])
    reference_opt = torch.optim.AdamW([ref_names[n] for n in old_names], lr=1e-4)
    reference_opt.load_state_dict(copy.deepcopy(saved))
    reference_scheduler = schedule(reference_opt)
    reference_scheduler.load_state_dict(old_scheduler.state_dict())
    reference_opt.param_groups[0]['lr'] = old.param_groups[0]['lr']
    for _ in range(3):
        for name in old_names:
            grad = torch.randn_like(by_name[name])
            by_name[name].grad = grad.clone()
            ref_names[name].grad = grad.clone()
        lr = reference_opt.param_groups[0]['lr']
        with torch.no_grad():
            for name in CONDITIONING_WEIGHTS:
                ref_names[name].mul_((1-lr*.4)/(1-lr*.01))
        reference_opt.step()
        target.step()
        reference_scheduler.step()
        target_scheduler.step()
        assert target_scheduler.get_last_lr() == reference_scheduler.get_last_lr() * 2
        for name in old_names:
            torch.testing.assert_close(by_name[name], ref_names[name], rtol=2e-6, atol=1e-7)


def test_recipe_migration_rejects_other_training_changes():
    new = asdict(load_config(Path(__file__).parents[1] / 'configs/hero.toml'))
    old = copy.deepcopy(new)
    del old['optim']['adam_conditioning_wd']
    check_recipe_change(old, new)
    for section, key, value in [('optim', 'adam_wd', .4), ('optim', 'muon_lr', .03),
                                ('train', 'seed', 12)]:
        changed = copy.deepcopy(new)
        changed[section][key] = value
        with pytest.raises(ValueError, match='only conditioning decay'):
            check_recipe_change(old, changed)


def test_offline_checkpoint_migration_publishes_preserved_full_state(tmp_path):
    config_path = tmp_path / 'run.toml'
    text = (Path(__file__).parents[1] / 'configs/hero.toml').read_text()
    for old, new in [('hidden_size = 1152', 'hidden_size = 64'),
                     ('num_heads = 16', 'num_heads = 4'),
                     ('single_stream_depth = 24', 'single_stream_depth = 1'),
                     ('mlp_ratio = 2.6666666666666665', 'mlp_ratio = 2.0')]:
        text = text.replace(old, new)
    config_path.write_text(text)
    config = load_config(config_path)
    old_config = asdict(config)
    del old_config['optim']['adam_conditioning_wd']
    model, (muon, _), adam = model_and_optimizers()
    sched = schedule(adam)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    adam.step()
    sched.step()
    source = tmp_path / 'original/checkpoint_step_000001'
    source.mkdir(parents=True)
    save_file(model.state_dict(), str(source / 'model.safetensors'))
    torch.save(model.state_dict(), source / 'ema_weights.pt')
    for name, value in {
        'optimizer.bin': muon.state_dict(), 'optimizer_1.bin': adam.state_dict(),
        'scheduler.bin': {'last_epoch': 1}, 'scheduler_1.bin': sched.state_dict(),
        'random_states_0.pkl': torch.get_rng_state(),
        'sampler_state_rank_00000.pt': {'next': 13},
        'npu_rng_state_rank_00000.pt': torch.get_rng_state(),
    }.items():
        torch.save(value, source / name)
    for name, value in {'run_config.json': old_config,
                        'transformer_config.json': model.get_config(),
                        'bucket_plan.json': {'unchanged': [32, 64]}}.items():
        (source / name).write_text(json.dumps(value))
    write_checkpoint_record(source, step=1, max_steps=600000, scheduler_count=2,
                            use_ema=True, world_size=1, device_type='npu')
    before = {p.name: sha256(p) for p in source.iterdir()}
    dest = tmp_path / 'migrated' / source.name
    report = migrate(source, dest, config_path, reason='test state-preserving split')
    assert before == {p.name: sha256(p) for p in source.iterdir()}
    assert validate_checkpoint(dest, max_steps=600000, require_record=True, device_type='npu') == 1
    assert json.loads((dest / 'run_config.json').read_text()) == asdict(config)
    state = torch.load(dest / 'optimizer_1.bin', weights_only=False)
    assert [g['weight_decay'] for g in state['param_groups']] == [.01, .4]
    assert report['source_sha256']['model.safetensors'] == report['destination_sha256']['model.safetensors']
    with pytest.raises(ValueError, match='new directory'):
        migrate(source, dest, config_path, reason='refuse overwrite')
