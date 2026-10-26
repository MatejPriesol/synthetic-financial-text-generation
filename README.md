# Synthetic Financial Data Generation and Evaluation Pipeline

**Master's Thesis in Machine Learning**  
**KTH Royal Institute of Technology**  

This repository contains the codebase for generating and evaluating synthetic financial text paired with relational Knowledge Graph (KG) triplets. It was developed to explore the viability of Large Language Models (LLMs) in synthesizing high-quality Relation Extraction (RE) data, investigating the critical trade-off between structural adequacy (following the instructions and constraints) and generative diversity (lexical and semantic).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Pipeline Execution](#pipeline-execution)
   - [1. Data Preparation](#1-data-preparation)
   - [2. Synthetic Data Generation](#2-synthetic-data-generation)
   - [3. Dataset Pruning](#3-dataset-pruning)
   - [4. Evaluation](#4-evaluation)
5. [Evaluation Metrics Explained](#evaluation-metrics-explained)

---

## Project Overview

In the financial domain, training robust information extraction models requires massive amounts of labeled relational data. However, human annotation is expensive and un-scalable, leading to limited availability of training data, creating a need for synthetic generation.

This project addresses this data bottleneck by building an automated pipeline to synthetically generate labeled financial sentences. Utilizing Llama-3.3-70B and Qwen-2.5-72B models, it evaluates four distinct strategies: unconstrained zero-shot and few-shot baselines, a vocabulary-restricted entity strategy, and a structurally constrained template-based approach.

A core focus of this repository is its comprehensive Evaluation Framework, which automatically assesses the generated datasets without human intervention along the four core dimensions: lexical and semantic diversity, adherence to structural constraints, grammatical correctness, and the statistical analysis of generative biases.

---

## Repository Structure

```text
MasterThesis/
├── data/                               # Datasets and evaluation reports
│   ├── 01_raw/                         # Raw FIRE dataset and types
│   ├── 02_processed/                   # Processed real-world FIRE reference data
│   ├── 03_synthetic/                   # Generated datasets and ontology constraints
│   ├── 04_reports/                     # JSON evaluation reports
│   └── 05_archive/                     # Final thesis datasets (final_fire, final_llama, final_qwen)
├── notebooks/                          # Jupyter notebooks for data visualization and results plotting
├── output/                             # Evaluation plots and dataset comparisons for thesis defense
├── scripts/                            # Utility scripts (data pruning, reference extraction, length analysis)
├── src/                                # Core Python package
│   ├── data_loader.py                  # Data parsing and loading utilities
│   ├── synthetic_data_generation.py    # Main LLM generation engine
│   ├── evaluation/                     # Evaluation Framework (Adequacy, Diversity, Grammar, Perplexity, Stats)
│   └── generators/                     # LLM client wrappers (Ollama/Groq) and Prompt Templates
├── requirements.txt                    # Python package dependencies
└── README.md                           # This file
```

---

## Setup & Installation

### 1. Python Environment
This project was developed using  Python 3.11 (though other Python 3.10+ versions should also work). Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. External Tools & API Keys
* **Local Generation (Ollama)**: To use local models (e.g., `qwen2.5:72b`), install [Ollama](https://ollama.com/) and ensure the server is running on `http://localhost:11434`.
* **API Generation (Groq)**: To use the Groq API, create a `.env` file in the root directory and add your API keys (the code supports key rotation for rate limits):
  ```env
  GROQ_API_KEY_1=your_key_here
  GROQ_API_KEY_2=your_second_key_here
  GROQ_API_KEY_3=your_third_key_here
  ```
  *(Note: The Llama-3.3-70B model originally used on Groq has been decommissioned. If you are re-running generation via the API, you will need to update the target model identifier to a currently supported model in your configuration.)*  
   
* **NLP Models**: The evaluation suite will automatically download necessary Hugging Face models (e.g., GPT-2, RoBERTa-CoLA) on their first run.

---

## Pipeline Execution

*(Note: You can use the `--help` flag on any script below to see the full list of available command-line arguments and parameters.)*

### 1. Data Preparation
Both the raw and processed reference datasets are already provided in this repository under `data/01_raw/` and `data/02_processed/`. You can proceed directly to data generation or evaluation. 

If you make changes to the data loader and wish to re-process the raw datasets from scratch, simply run:
```bash
python -m scripts.prepare_data --data all
```

### 2. Synthetic Data Generation
The generation engine creates synthetic sentences using one of four controlled prompting strategies. 

**Usage:**
```bash
python -m src.synthetic_data_generation --strategy <strategy_name> --provider <provider> --output <filename.json>
```

**Arguments:**
* `--strategy`: `zero-shot`, `few-shot`, `entity` (restricts generation to a specific entity pool), or `template` (forces generation of specific triplet structures).
* `--provider`: `local` (Ollama) or `api` (Groq).
* `--output`: (Optional) Name of the output JSON file.

**Example:**
```bash
python -m src.synthetic_data_generation --strategy template --provider local --output template_qwen.json
```
*Generated datasets are saved to `data/03_synthetic/`.*

### 3. Dataset Pruning
To ensure a strictly fair 1:1 comparison across all evaluation metrics, all datasets (both the real-world reference and the synthetic outputs) must be normalized to the exact same size. This script randomly samples exactly 2,000 entries using a fixed seed (`42`) for deterministic reproducibility.

**Usage:**
```bash
python -m scripts.prune_dataset --input <generated_file.json> --output <pruned_file.json>
```

**Arguments:**
- `--input`: Path to the raw generated JSON dataset.
- `--output`: Path to save the pruned JSON dataset.
- `--target_size`: (Optional) Target size, defaults to 2000.
- `--fire`: (Optional) Used only for preparing the real-world FIRE reference dataset. It filters out sentences outside the 1-5 triplet range to match the bounds requested during synthetic generation. 

**Example:**
```bash
python -m scripts.prune_dataset --input data/03_synthetic/template_qwen_raw.json --output data/03_synthetic/template_qwen.json --target_size 2000
```

### 4. Evaluation
The evaluation framework runs a comprehensive automated analysis on a specified dataset. You can run all evaluations at once or trigger them individually.

**Usage:**
```bash
python -m src.evaluation.run_evals --data <path_to_dataset> [EVAL_FLAGS] --verbose --save <report_name>
```

**Arguments:**
* `--data`: Path to the JSON dataset to evaluate.
* `--all`: Run all available evaluations.
* `[EVAL_FLAGS]`: Run specific modules individually by passing `--grammar`, `--stats`, `--diversity`, or `--adequacy`.
* `--run_name` / `-n`: A short identifier for the report (required if saving).
* `--save` / `-s`: Saves the evaluation metrics to `data/04_reports/<report_name>.json`.
* `--verbose` / `-v`: (Optional) Print detailed evaluation results.

**Example:**
```bash
python -m src.evaluation.run_evals --data data/03_synthetic/template_qwen.json --all --run_name Qwen_Template --save qwen_template_report
```

*(Note: You can also run specific modules standalone using `python -m src.evaluation.<module> --data <path>`)*

---

## Evaluation Metrics Explained

### 1. Adequacy (`adequacy.py`)
Measures the LLM's ability to follow instructions and strict prompt constraints without hallucinating relations.

- **Global Adequacy**: Evaluates Ontology Compliance (does every generated triplet exist in the allowed financial ontology?) and Blueprint Adherence (did the model generate the exact number of facts requested?).
- **Targeted Adherence**: Evaluates precision and recall specifically for the Entity and Template strategies to verify if the LLM actually utilized the exact entities or relation structures passed in the prompt.

### 2. Diversity (`diversity.py`)
Detects generative mode collapse and structural repetition, ensuring the dataset provides enough variance for robust downstream training.

- **Semantic Diversity**: Uses all-MiniLM-L6-v2 embeddings and pairwise Cosine Similarity to find semantic duplicates and measure global corpus variety.
- **Syntactic & Lexical Diversity**: Calculates Self-BLEU to measure structural overlap and tracks Unique N-Gram Ratios to assess vocabulary richness.

### 3. Grammatical Correctness (`grammatical_correctness.py`)
Measures the linguistic naturalness of the synthesized financial text.

- **Acceptability Confidence**: Utilizes a RoBERTa model fine-tuned on the Corpus of Linguistic Acceptability (CoLA) to flag malformed sentences, broken grammar, or awkward phrasing induced by strict prompt constraints.

### 4. Statistical Distributions (`statistical.py`)
Analyzes the structural properties and biases of the datasets.

- **Entity Bias & Canonicalization**: Tracks the most frequently generated subjects and objects. Features an alias-mapping tool to expose "generative shortcuts" (e.g., the model over-relying on generic placeholders like "John Doe" or major corporations like "Apple").
- **Profile Ontology**: Tracks the most frequently used relations and triplets.
- **Triplet Complexity**: Maps the density of relational facts per sentence to see if the LLM defaults to simple structures or successfully mimics the complexity of real financial reporting.
