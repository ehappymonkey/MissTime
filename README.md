# MissTime: State-Aware Retrieval for General Time Series Analysis with Missing Variables

Official code repository for **MissTime**, currently **under review at NeurIPS**.

MissTime is a unified framework for time-series learning under **missing variables**, with support for:

- Long-term forecasting
- Short-term forecasting
- Imputation
- Classification (UEA)
- Anomaly detection

The framework integrates multiple backbones (e.g., `TimesNet`, `TimeMixer`, `iTransformer`) and retrieval modes (`no_rag`, `feature_rag`, `latent_rag`).

---

## Repository Structure

```text
source_code/
├── run.py
├── data_provider/
├── exp/
├── layers/
├── models/
├── rag/
├── scripts/
│   ├── forecasting/
│   ├── imputation/
│   ├── anomaly_detection/
│   └── classification/
└── drawings/
```

- `run.py`: unified entry for training/testing
- `scripts/`: runnable experiment scripts grouped by task and dataset
- `data_provider/`: data loaders and missing-variable collation logic
- `exp/`: task-specific training/evaluation pipelines
- `rag/`: retrieval modules

---

## Environment

Recommended:

- Python 3.8+
- PyTorch (CUDA recommended)

Install common dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn tqdm
pip install einops sktime
pip install huggingface_hub datasets
```

Some optional models may require additional third-party packages.

---

## Data Source

This project uses datasets from the **Time Series Library** ecosystem.

- Time Series Library (GitHub): [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- Hugging Face dataset repo used by loaders: `thuml/Time-Series-Library`

For several datasets, loaders can auto-download from Hugging Face when local files are missing.

---

## Quick Start

Run from repository root (`source_code/`).

### 1) Long-term Forecasting (ETTh1, TimesNet)

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TimesNet \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --itr 1 \
  --top_k 5 \
  --mask_ratio 0.25 \
  --gpu 0 \
  --rag_type feature_rag \
  --retrieve_encoder iTransformer \
  --latent_dim 512 \
  --encoder_epochs 20 \
  --train_epochs 2
```

### 2) Short-term Forecasting (PEMS03, TimesNet)

```bash
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_12 \
  --model TimesNet \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.001 \
  --itr 1 \
  --mask_ratio 0.25 \
  --gpu 0 \
  --rag_type no_rag \
  --freq m
```

### 3) Imputation (Electricity, TimesNet)

```bash
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask \
  --mask_rate 0.125 \
  --model TimesNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001 \
  --mask_ratio 0.25 \
  --gpu 0
```

### 4) Classification (SelfRegulationSCP1, TimesNet)

```bash
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --mask_ratio 0.25 \
  --gpu 0
```

### 5) Anomaly Detection (SMD, TimesNet)

```bash
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model TimesNet \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 38 \
  --c_out 38 \
  --top_k 5 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 10 \
  --gpu 0 \
  --mask_ratio 0.25 \
  --rag_type latent_rag \
  --retrieve_encoder Typology \
  --contrastive_loss hard_negative
```

---

## Script-Based Reproduction

Prepared scripts are available in:

- `scripts/forecasting/`
- `scripts/imputation/`
- `scripts/classification/`
- `scripts/anomaly_detection/`

You can run the corresponding `.sh` scripts directly for dataset/model-specific settings.

---

## Important Arguments

- `--task_name`: `long_term_forecast | imputation | anomaly_detection | classification`
- `--model`: backbone model name
- `--rag_type`: `no_rag | feature_rag | latent_rag`
- `--mask_ratio`: missing-variable ratio used in batch-level masking
- `--root_path`, `--data_path`: dataset location
- `--gpu`: GPU id

Note: short-term forecasting is handled in the same forecasting pipeline with PEMS-style settings (e.g., `--data PEMS`, `--pred_len 12`).

---

## Citation

This work is currently under review at NeurIPS. Citation information will be released after publication.

---

## Acknowledgements

This project builds on open-source time-series ecosystems, especially Time Series Library and related benchmarks.
