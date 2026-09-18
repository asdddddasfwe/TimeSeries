# Repository Guidelines

## Project Structure & Module Organization

`run.py` is the experiment entry point. `models/` contains model implementations; `layers/` holds reusable neural-network components. Task-specific training and evaluation live in `exp/`, dataset loading in `data_provider/`, and metrics and helpers in `utils/`. Reproduction scripts are grouped by task and dataset under `scripts/`. `tutorial/` contains notebooks and illustrations; `pic/` contains documentation assets. Keep downloaded datasets in `dataset/`.

## Build, Test, and Development Commands

Run commands from the repository root; use Bash or WSL for shell scripts. Create an isolated Python environment and install the hardware-compatible PyTorch package described in `README.md` before installing other dependencies.

- `python -m pip install -r requirements.txt`: install pinned dependencies. Follow the README for model-specific extras such as Mamba or Moirai.
- `python run.py --help`: inspect task, model, dataset, and device options.
- `bash scripts/short_term_forecast/DLinear_M4.sh`: run the M4 reproduction experiment after preparing its dataset.
- `docker compose up -d --build`: build and start the supplied development container.

For a one-epoch forecasting smoke test, prepare `dataset/ETT-small/ETTh1.csv`, then run:

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --model_id smoke_dlinear --model DLinear --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --train_epochs 1 --num_workers 0 --no_use_gpu
```

## Coding Style & Naming Conventions

Use four-space Python indentation and follow nearby code. Prefer `snake_case` for new functions and variables. Preserve model filenames such as `DLinear.py` and the exported `Model(nn.Module)` interface. Models are discovered automatically from `models/`; match `--model` to the filename stem. Document tensor shapes and non-obvious transformations. No formatter or linter configuration is currently checked in.

## Testing Guidelines

Model and logger regression tests use standard-library `unittest` in `tests/`. Run `python -m unittest discover -s tests -p 'test_*.py'`; TensorBoard event tests require the project dependencies. No coverage threshold is configured. Validate model changes with README quick tests and reproduction scripts. Report commands, datasets, devices, seeds, and metrics; check output shapes and finite losses.

## Commit & Pull Request Guidelines

Local history contains only the initialization commit (`初始化`), so no established commit convention can be inferred. Use concise, imperative subjects describing a focused change.

Follow `CONTRIBUTING.md`: describe bug fixes and reproduction steps in PRs. For new models, first open an issue linking the published paper and official code, then include the model implementation and reproduction scripts. Link relevant issues and provide validation results. Exclude datasets, checkpoints, generated results, and credentials from commits.
