# Gender and Race/Color Biases in LLM Recommendations for Brazilian Undergraduate Fields

This repository contains the code, prompts, configurations, generated outputs, and analysis notebooks for the paper:

> **Who Gets Recommended What? Gender and Race/Color Biases in LLM Recommendations for Brazilian Undergraduate Fields**  
> Submitted to IEEE LACCI 2026.

The study evaluates whether large language models (LLMs) produce demographically biased representations and recommendations for Brazilian undergraduate fields. We focus on gender and race/color categories in the Brazilian context and compare model outputs with external statistics from IBGE.

## Overview

The repository supports two main behavioral experiments plus an interpretability pipeline based on Natural Language Autoencoders (NLA).

### 1. Demographic attribution task

Models are prompted to generate structured profiles of Brazilian adults who completed a specific undergraduate field. Each generated profile includes demographic and socioeconomic attributes such as:

- name;
- age;
- Brazilian state;
- monthly income;
- attributed sex;
- attributed race/color.

The goal is to analyze how LLMs represent demographic groups across undergraduate fields and how these representations compare with IBGE statistics.

### 2. Educational recommendation task

Models are prompted as if the user were a final-year high-school student in Brazil seeking advice about undergraduate fields. The prompt varies demographic cues such as gender and race/color, and may also include an academic-interest signal based on ENEM knowledge areas.

The model is instructed to recommend exactly three undergraduate fields from a fixed list. We analyze whether demographic cues affect the exposure of different fields in the recommendations.

### 3. NLA interpretability pipeline

For local models served through SGLang, residual-stream activations can be extracted, verbalized, and reconstructed with Natural Language Autoencoder (NLA) actor/critic checkpoints. The pipeline supports:

1. **Activation extraction** — per-token residual activations at a chosen layer;
2. **Verbalization** — natural-language explanations of activation vectors via the NLA actor;
3. **Reconstruction** — critic scoring of how well verbalizations reconstruct the original activations;
4. **Analysis** — token-local, span-level, and outcome-linked summaries related to gender bias.

Artifacts are written as parquet files under `artifacts/nla/`.

## Repository structure

```text
.
├── conf/
│   ├── main_config.yaml
│   ├── nla_config.yaml
│   ├── profile_config.yaml
│   └── recommendation_config.yaml
│
├── data/
│   ├── generated_profiles.csv
│   ├── generated_profiles.jsonl
│   ├── generated_profiles.pkl
│   ├── generated_recommendations.csv
│   ├── generated_recommendations.jsonl
│   ├── generated_recommendations.pkl
│   ├── undergraduate_fields_for_profile.yaml
│   ├── undergraduate_fields_for_recommendation.yaml
│   └── tables/
│       └── ibge_undergraduate_fields.xlsx
│
├── scripts/
│   ├── extract_nla_activations.py
│   ├── run_nla_verbalization.py
│   ├── run_nla_reconstruction.py
│   └── analyze_nla_gender_bias.py
│
├── src/
│   ├── main/
│   │   ├── utils.py
│   │   ├── template_expansion.py
│   │   └── nla/
│   │       ├── activation_extraction.py
│   │       ├── analysis.py
│   │       ├── ids.py
│   │       ├── io.py
│   │       ├── nla_client.py
│   │       ├── reconstruction.py
│   │       ├── schemas.py
│   │       └── token_annotation.py
│   └── analysis/
│       ├── preprocessing_undergraduate_fields.ipynb
│       ├── processing_profile_results.ipynb
│       └── processing_recommendation_results.ipynb
│
├── nla_inference.py
├── run_main.py
├── pixi.toml
├── pixi.lock
├── LICENSE
└── README.md
```

## Data

### 1. The data/ directory includes:

- generated_profiles.csv: generated outputs for the demographic attribution task;
- generated_profiles.jsonl: JSONL version of the generated profile outputs;
- generated_profiles.pkl: cache file used during profile generation;
- generated_recommendations.csv: generated outputs for the recommendation task;
- generated_recommendations.jsonl: JSONL version of the generated recommendation outputs;
- generated_recommendations.pkl: cache file used during recommendation generation;
- tables/ibge_undergraduate_fields.xlsx: processed IBGE reference data used in the analyses;
- undergraduate_fields_for_profile.yaml: undergraduate-field list used in the profile-generation task;
- undergraduate_fields_for_recommendation.yaml: undergraduate-field list used in the recommendation task.

NLA run artifacts (activations, verbalizations, reconstructions, analysis CSVs) are written under `artifacts/nla/` and are not committed to the repository. Actor/critic checkpoints are expected under paths such as `checkpoints/` (see `conf/nla_config.yaml`).

## Configuration files

### 1. The experiments are controlled through Hydra configuration files in conf/.

- profile_config.yaml: configuration for the demographic attribution task.
- recommendation_config.yaml: configuration for the educational recommendation task.
- main_config.yaml: default configuration used by run_main.py.
- nla_config.yaml: configuration for the NLA interpretability pipeline.

### 2. The configuration files specify:

- models to evaluate;
- provider/backend for each model (including `local_openai` / `sglang` for local serving);
- temperature;
- number of repetitions;
- output paths;
- cache paths;
- system prompts;
- user prompts;
- demographic conditions;
- undergraduate-field lists;
- academic-interest conditions.

### 3. NLA-specific settings in nla_config.yaml include:

- base model and residual layer (`BASE_MODEL`, `NLA_LAYER`);
- actor/critic checkpoint paths (`AV_CHECKPOINT`, `AR_CHECKPOINT`);
- SGLang endpoint for verbalization (`SGLANG_NLA_URL`);
- verbalization tier and batching (`VERBALIZATION_TIER`, `VERBALIZATION_BATCH_SIZE`, `VERBALIZATION_CONCURRENCY`, `VERBALIZATION_FLUSH_ROWS`);
- reconstruction batching (`RECONSTRUCTION_BATCH_SIZE`, `RECONSTRUCTION_FLUSH_ROWS`);
- sampling (`TEMPERATURE`, `MAX_NEW_TOKENS`);
- artifact directory and optional generation JSONL for outcome-linked analysis.

## Environment setup

This repository uses pixi for environment management. Install the environment with:

```bash
pixi install
```
Then activate the environment:

```bash
pixi shell
```

Alternatively, commands can be run directly with:

```bash
pixi run <task-name>
```

## Running experiments

### Behavioral generation

The main experiment runner is:

```bash
python run_main.py
```

By default, this uses the Hydra configuration specified in run_main.py.

To run a specific configuration, use:

```bash
python run_main.py --config-name some_config
```

Local models can be served with SGLang and selected in the config via the `local_openai` (or `sglang`) provider, for example `["Qwen/Qwen2.5-7B-Instruct", "local_openai"]`. I only tested local runs with sglang (Qwen 2.5 7B).

### NLA pipeline

Prerequisites: a running SGLang server for the NLA actor, and actor/critic checkpoints at the paths in `nla_config.yaml`.

Run the stages in order (Hydra config: `conf/nla_config.yaml`):

```bash
pixi run nla-extract
pixi run nla-verbalize
pixi run nla-reconstruct
pixi run nla-analyze
```

Equivalent direct invocations:

```bash
python scripts/extract_nla_activations.py
python scripts/run_nla_verbalization.py
python scripts/run_nla_reconstruction.py
python scripts/analyze_nla_gender_bias.py
```

Override settings from the CLI as needed, for example:

```bash
python scripts/run_nla_verbalization.py VERBALIZATION_TIER=tier2 VERBALIZATION_BATCH_SIZE=16
```

`nla_inference.py` is a compatibility shim / smoke-test CLI for `NLAClient` and `NLACritic`:

```bash
python nla_inference.py ./checkpoints/nla-qwen2.5-7b-L20-av --sglang-url http://localhost:30001
```
