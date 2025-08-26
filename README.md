# Dual Evaluation for Trust in Persuasive Dialogues (DuET-PD)

### **Authors:** [Bryan Chen Zhengyu Tan](https://scholar.google.com.sg/citations?user=86_eWK0AAAAJ&hl=en)<sup>1,2</sup>, [Daniel Wai Kit Chin](https://scholar.google.com/citations?user=Y701_bEAAAAJ&hl=en)<sup>1</sup>, [Zhengyuan Liu](https://scholar.google.com/citations?user=tqzidGsAAAAJ&hl=en)<sup>2</sup>, [Nancy F. Chen](https://scholar.google.com/citations?user=K3Z9UiAAAAAJ&hl=en)<sup>2</sup>, [Roy Ka-Wei Lee](https://scholar.google.com/citations?user=uQxdOlsAAAAJ&hl=en)<sup>1</sup>

<small><sup>1</sup>Singapore University of Technology and Design (SUTD)</small>  
<small><sup>2</sup>Institute for Infocomm Research (I2R), A*STAR, Singapore</small>

---

This repository contains the datasets, code, and analysis notebooks necessary to reproduce the results presented in the paper: ***[Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD](https://arxiv.org/abs/2508.17450)***.

We evaluate the susceptibility of Large Language Models (LLMs) to persuasion across multi-turn dialogues, focusing on the critical balance between resisting misinformation (negative persuasion) and accepting valid corrections (positive persuasion). We assess performance on knowledge-intensive (MMLU-Pro) and safety-critical (SALAD-Bench) benchmarks, analyse the effectiveness of various persuasion techniques, and evaluate mitigation strategies including prompting and Direct Preference Optimisation (DPO) using the LlamaFactory framework.

## 📝 Repository Structure

```
.
├── 1_initial_eval.py                 # Script for initial model stance evaluation
├── 2_stance_change_eval.py           # Script for multi-turn persuasion evaluation
├── 3_analysis.py                     # Script to process evaluation results into metrics
├── 4_analysis_tables.ipynb           # Notebook to generate paper tables/figures from analysis
├── 5_generate_refutation_and_affirmation.py # Script to generate ideal responses for DPO
├── 6_create_dpo_datasets.ipynb       # Notebook to construct DPO preference datasets
├── 7_unzip_and_cp_to_dataset_dir.py  # Utility to prepare DPO data for LlamaFactory
├── LLaMA-Factory/                    # Submodule or copy of LlamaFactory framework
│   ├── data/                         # LlamaFactory data dir (DPO datasets copied here)
│   ├── examples/                     # LlamaFactory examples
│   ├── llama3_*.yaml                 # YAML configs for DPO training experiments
│   └── ...                           # Other LlamaFactory files
├── README.md                         # This README file
├── analysis_outputs_test_only/       # Directory for analysis script outputs (CSV/JSON)
├── data/
│   ├── generated_appeal_withID.json  # Core MCQ dataset with generated persuasions
│   └── qn_id_split_50_50_increments_llama31_8b.json # Train/test split IDs
├── dpo_datasets/                     # Compressed DPO datasets (.json.gz, .csv.gz)
│   ├── baseline*.json.gz
│   ├── resist_10*.json.gz            # Corresponds to 20% training data in paper
│   ├── resist_20*.json.gz            # Corresponds to 40% training data in paper
│   ├── resist_30*.json.gz            # Corresponds to 60% training data in paper
│   ├── resist_40*.json.gz            # Corresponds to 80% training data in paper
│   ├── resist_50*.json.gz            # Corresponds to 100% training data in paper
│   ├── holistic_10*.json.gz          # Corresponds to 20% training data in paper
│   ├── ...                           # etc.
│   └── ...
├── figures/                          # Directory for figures used in paper/notebooks
├── persuasion_taxonomy.jsonl         # Definitions of persuasion techniques
├── prompts.py                        # Helper module for prompt templates
├── requirements.txt                  # Python dependencies
├── results/                          # Raw initial evaluation outputs for baseline models
│   └── initial_eval_results_*.json
├── results_after_train/              # Raw evaluation outputs for DPO-trained models
│   ├── initial_eval_results_*.json
│   └── stance_change_*/              # Raw stance change outputs for DPO models
└── utils.py                          # Utility functions (API calls, parsing, etc.)
```

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Social-AI-Studio/DuET-PD
    cd DuET-PD
    ```
2.  **Set up Conda Environment:** We recommend using Conda for environment management. Create a new environment (Python 3.11 recommended) and install dependencies:
    ```bash
    conda create -n persuasion_env python=3.11 -y
    conda activate persuasion_env
    pip install -r requirements.txt
    ```
3.  **Environment Variables:** Create a `.env` file in the root directory of the project and add your API keys and desired port for local models:
    ```dotenv
    # .env file
    OPENAI_API_KEY=your_openai_api_key_here
    API_PORT=8000 # Default port for local vLLM server
    ```
    The scripts will load these variables using `python-dotenv`.
4.  **LlamaFactory:** This repository includes [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) as directory. **Use the version specificied in the current repo's `requirements.txt` file. (`llamafactory==0.9.2.dev0`)**.

## 💾 Data

*   **Core Evaluation Dataset:** `data/generated_appeal_withID.json` contains the combined MMLU-Pro and SALAD-Bench MCQs, along with the pre-generated positive and negative persuasive appeals for each technique.
*   **Train/Test Split:** `data/qn_id_split_50_50_increments_llama31_8b.json` defines the question IDs used for the training and testing splits, stratified by source, category, and initial Llama-3.1-8B correctness. It also contains incremental subsets used for data scaling experiments.
*   **DPO Datasets:** The `dpo_datasets/` directory contains compressed (`.json.gz`) preference pair datasets (Baseline, Resist, Holistic, and their size increments) generated by `6_create_dpo_datasets.ipynb`. These need to be unzipped and placed into `LLaMA-Factory/data/` for training (see DPO section below).

## 🚀 Reproducing Evaluation Results

The following scripts reproduce the stance change evaluations for baseline and fine-tuned models.

### 1. Initial Stance Evaluation (`1_initial_eval.py`)

This script evaluates the baseline performance (Turn 0 accuracy and confidence) of a model on the core dataset.

**For Local Models (e.g., Mistral-7B, Base Llama-3.1):**

*   **Important:** First, start a vLLM server for your model in a separate terminal. Ensure the port matches `API_PORT` in your `.env` file (default 8000).
    ```bash
    # Example for Mistral-7B-Instruct-v0.3
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000
    ```
*   Then, run the evaluation script:
    ```bash
    python -u 1_initial_eval.py \
        --input_json data/generated_appeal_withID.json \
        --output_json results/initial_eval_results.json \
        --model_name mistralai/Mistral-7B-Instruct-v0.3 \
        --max_tokens 1 \
        --top_logprobs 20 \
        --use_local
    ```
    *(Output filename will be automatically suffixed with the model name)*

**For OpenAI API Models (e.g., GPT-4o-mini):**

```bash
python -u 1_initial_eval.py \
    --input_json data/generated_appeal_withID.json \
    --output_json results/initial_eval_results.json \
    --model_name gpt-4o-mini \
    --max_tokens 1 \
    --top_logprobs 20
```

### 2. Stance Change Evaluation (`2_stance_change_eval.py`)

This script runs the multi-turn persuasion dialogues using the output from Step 1. It generates 7 output files per model (one for each persuasion technique + repetition) in the specified output directory.

**For Local Models:**

*   Ensure the vLLM server from Step 1 is still running.
*   Run the script:
    ```bash
    python 2_stance_change_eval.py \
      --input_json results/initial_eval_results_Mistral-7B-Instruct-v0.3.json \
      --output_dir results/stance_change_Mistral-7B-Instruct-v0.3 \
      --persuasion_taxonomy_file persuasion_taxonomy.jsonl \
      --model_name mistralai/Mistral-7B-Instruct-v0.3 \
      --use_local \
      --top_logprobs 20
    ```

**For OpenAI API Models:**

```bash
python 2_stance_change_eval.py \
  --input_json results/initial_eval_results_gpt-4o-mini.json \
  --output_dir results/stance_change_gpt-4o-mini \
  --persuasion_taxonomy_file persuasion_taxonomy.jsonl \
  --model_name gpt-4o-mini \
  --top_logprobs 20
```

**Evaluating on Test Set Only:** To run evaluations only on the test split defined in `data/qn_id_split_*.json`, add the `--test_only` flag to the commands for both `1_initial_eval.py` and `2_stance_change_eval.py`.

**Using Cautious Prompt:** To evaluate with the cautious system prompt modification, add the `--system_prompt_type cautious` flag to the `2_stance_change_eval.py` command:
```bash
# Example for local Llama-3.1-8B with cautious prompt
nohup python -u 2_stance_change_eval.py \
  --input_json results/initial_eval_results_Llama-3.1-8B-Instruct.json \
  --output_dir results/stance_change_Llama-3.1-8B-Instruct_cautiousPrompt \
  --persuasion_taxonomy_file persuasion_taxonomy.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --use_local \
  --system_prompt_type cautious \
  --top_logprobs 20 > out_cautious.log 2>&1 &
```

### 3. Analysis (`3_analysis.py`)

This script processes the raw stance change results (output by Step 2) to compute the metrics used in the paper (Acc@n, Flip@n, confidence trends).

```bash
# Example for gpt-4o-mini results, analyzing only the test set data
python 3_analysis.py \
    --model_name gpt-4o-mini \
    --results_dir results/stance_change_gpt-4o-mini \
    --analysis_output_dir analysis_outputs_test_only \
    --test_only \
    --qn_split_path data/qn_id_split_50_50_increments_llama31_8b.json
```
This generates consolidated CSV and JSON files in the `--analysis_output_dir`.

### 4. Generating Paper Tables/Figures (`4_analysis_tables.ipynb`)

Open and run the Jupyter notebook `4_analysis_tables.ipynb`. This notebook reads the outputs from `3_analysis.py` (e.g., from `analysis_outputs_test_only/`) and generates the summary tables and plots presented in the paper.

## 🧬 Reproducing DPO Training

Follow these steps to generate the DPO datasets and train the mitigated Llama-3.1-8B-Instruct models.

### 1. Generate Ideal Responses (`5_generate_refutation_and_affirmation.py`)

This script reads the stance change results for the baseline model (Llama-3.1-8B-Instruct) and uses GPT-4o-mini (via API) to generate ideal refutations (for NEG scenarios) and affirmations (for POS scenarios).

```bash
# Ensure OPENAI_API_KEY is set in .env
python 5_generate_refutation_and_affirmation.py \
    --input_dir results/stance_change_Llama-3.1-8B-Instruct \
    --output_dir results/stance_change_Llama-3.1-8B-Instruct_intended-response
```
*(Note: This uses OpenAI API calls and may incur costs.)*

### 2. Create DPO Preference Datasets (`6_create_dpo_datasets.ipynb`)

Run the Jupyter notebook `6_create_dpo_datasets.ipynb`. This notebook processes the outputs from Step 1 and constructs the preference pairs for the Baseline, Resist, and Holistic datasets (including incremental sizes). It saves these datasets as compressed `.json.gz` files in the `dpo_datasets/` directory.

### 3. Prepare Data for LlamaFactory (`7_unzip_and_cp_to_dataset_dir.py`)

This utility script unzips the required DPO datasets from `dpo_datasets/` and copies the resulting `.json` files into the `LLaMA-Factory/data/` directory.

```bash
python 7_unzip_and_cp_to_dataset_dir.py
```

### 4. Run DPO Training (LlamaFactory)

Navigate into the LlamaFactory directory (or ensure `llamafactory-cli` is in your path) and use the provided YAML configuration files to launch training jobs.

**Important Naming Convention Note:** The paper refers to DPO dataset sizes as percentages of the *training set* (e.g., 20%, 40%, ..., 100%). However, the repository datasets, YAML files, and adapter names use suffixes (`_10`, `_20`, ..., `_50`) which correspond to these percentages. The mapping is:
*   Paper 20% = Repo `_10`
*   Paper 40% = Repo `_20`
*   Paper 60% = Repo `_30`
*   Paper 80% = Repo `_40`
*   Paper 100% = Repo `_50`
**Please use the repository naming convention (`_10` to `_50`) when running the commands below.**

```bash
# Example commands (run from the main project directory or ensure paths are correct)
cd LLaMA-Factory

# To train specific models from the paper (using repo naming):
llamafactory-cli train llama3_resist_50.yaml   # Corresponds to Resist-100% in paper
llamafactory-cli train llama3_holistic_50.yaml # Corresponds to Holistic-100% in paper
llamafactory-cli train llama3_resist_20.yaml   # Corresponds to Resist-40% in paper
llamafactory-cli train llama3_holistic_20.yaml # Corresponds to Holistic-40% in paper
# ... and so on for other sizes (_10, _30, _40)
```
Refer to the YAML files in `LLaMA-Factory/` for specific configurations. Ensure sufficient GPU resources (e.g., A40, A6000).

### 5. Evaluate DPO-Trained Models

After training, evaluate the resulting DPO-finetuned models.

*   **Start vLLM Server with LoRA Adapters:** Launch the vLLM server for the base model (`meta-llama/Llama-3.1-8B-Instruct`) and load the trained LoRA adapters. Replace `<HUGGINGFACE_USERNAME>` with the appropriate Hugging Face username where the adapters are hosted. **Use the repository naming convention (`_10` to `_50`) for the adapter names.** Run this in a separate terminal using `nohup`:
    ```bash
    # Example loading all adapters (adjust GPU visibility if needed)
    # Using repository naming convention for adapters (resist_10, holistic_10, etc.)
    CUDA_VISIBLE_DEVICES=1 nohup vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max_model_len 8192 \
        --enable-lora \
        --lora-modules \
        resist_10=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_resist_10 \
        holistic_10=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_holistic_10 \
        resist_20=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_resist_20 \
        holistic_20=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_holistic_20 \
        resist_30=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_resist_30 \
        holistic_30=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_holistic_30 \
        resist_40=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_resist_40 \
        holistic_40=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_holistic_40 \
        resist_50=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_resist_50 \
        holistic_50=<HUGGINGFACE_USERNAME>/Llama-3.1-8B-Instruct_holistic_50 \
        --port 8000 > vllm_server.log 2>&1 &
    ```

*   **Run Initial Evaluation (Step 1):** Run `1_initial_eval.py` for *each* trained adapter, specifying the adapter name (e.g., `resist_50`, `holistic_50`) as the `--model_name`. Store outputs in `results_after_train/`.
    ```bash
    # Example for resist_50 (corresponds to Resist-100% in paper)
    python -u 1_initial_eval.py \
        --input_json data/generated_appeal_withID.json \
        --output_json results_after_train/initial_eval_results.json \
        --model_name resist_50 \
        --max_tokens 1 \
        --top_logprobs 20 \
        --use_local

    # Example for holistic_50 (corresponds to Holistic-100% in paper)
    python -u 1_initial_eval.py \
        --input_json data/generated_appeal_withID.json \
        --output_json results_after_train/initial_eval_results.json \
        --model_name holistic_50 \
        --max_tokens 1 \
        --top_logprobs 20 \
        --use_local
    # Repeat for other resist/holistic suffixes (_10, _20, _30, _40)...
    ```

*   **Run Stance Change Evaluation (Step 2):** Run `2_stance_change_eval.py` for *each* trained adapter, using its corresponding initial evaluation file and specifying the adapter name (e.g., `resist_50`, `holistic_50`). Add `--test_only` if desired.
    ```bash
    # Example for resist_50 (test only)
    nohup python -u 2_stance_change_eval.py \
      --input_json results_after_train/initial_eval_results_resist_50.json \
      --output_dir results_after_train/stance_change_resist_50 \
      --persuasion_taxonomy_file persuasion_taxonomy.jsonl \
      --model_name resist_50 \
      --use_local \
      --test_only \
      --top_logprobs 20 > resist_50_eval.log 2>&1 &

    # Example for holistic_50 (test only)
    nohup python -u 2_stance_change_eval.py \
      --input_json results_after_train/initial_eval_results_holistic_50.json \
      --output_dir results_after_train/stance_change_holistic_50 \
      --persuasion_taxonomy_file persuasion_taxonomy.jsonl \
      --model_name holistic_50 \
      --use_local \
      --test_only \
      --top_logprobs 20 > holistic_50_eval.log 2>&1 &
    # Repeat for other resist/holistic suffixes (_10, _20, _30, _40)...
    ```

*   **Run Analysis (Steps 3 & 4):** Use `3_analysis.py` and `4_analysis_tables.ipynb`, pointing to the results directories in `results_after_train/` to generate metrics and figures for the fine-tuned models. Remember the naming convention difference when interpreting results against the paper.

## 📄 Citation (TODO)

If you find this work useful, please cite our paper:

```bibtex
@misc{tanPersuasionDynamicsLLMs2025,
  title = {Persuasion Dynamics in {{LLMs}}: Investigating Robustness and Adaptability in Knowledge and Safety with {{DuET-PD}}},
  shorttitle = {Persuasion Dynamics in {{LLMs}}},
  author = {Tan, Bryan Chen Zhengyu and Chin, Daniel Wai Kit and Liu, Zhengyuan and Chen, Nancy F. and Lee, Roy Ka-Wei},
  year = {2025},
  month = aug,
  number = {arXiv:2508.17450},
  eprint = {2508.17450},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2508.17450},
  abstract = {Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32\% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21\% to 76.54\%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.},
  archiveprefix = {arXiv},
  langid = {english},
}
```
