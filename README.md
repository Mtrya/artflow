# Inko: Flow Matching on Art Images

Inko is a bilingual text-to-image model with a 532M-parameter DiT and a
256p → 640p → 896p pretraining curriculum. Training targets Ascend; the
post-training release target is an 8-step student.

- [Design and training plan](notes/redesign_plan.md)
- [Pretraining recipe and checkpoint contracts](notes/pretrain_recipe.md)
- [Dataset composition and caption policy](notes/dataset_plan.md)
- [Infrastructure measurements and qualification limits](notes/infra_pretrain.md)
- [Post-training reward evidence and requirements](notes/posttrain_preflight.md)
- [Project tools](scripts/README.md)

The complete training configuration is [configs/pretrain.toml](configs/pretrain.toml).
Datasets, model weights and outputs use an explicit external storage root;
prompts and bucket plans are tracked in `configs/`.
