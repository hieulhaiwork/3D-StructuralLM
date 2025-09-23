# Fine-tuning with LLaMA-Factory

This guide explains how to set up and run fine-tuning for models using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which is included as a Git submodule in this project.

---

## Installation

### 1. Initialize the submodule

If you cloned the repo without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

From the **project root directory**, run:

```bash
pip install -e externals/LLaMA-Factory
```

## Fine-tune

1. Configure the training parameters in yaml file. Example
[`LLM\finetune\scripts\finetune_llama-3.2-1b_sft.yaml`](LLM\finetune\scripts\finetune_llama-3.2-1b_sft.yaml).  
   - Set the model checkpoint (e.g., `memeta-llama/Llama-3.2-1B`)
   - Adjust dataset paths or names
   - Add saving directory path
   - Tune hyperparameters (batch size, learning rate, LoRA settings, etc.)

2. Run the fine-tuning script. Example:

```bash
bash finetune/run_finetune.sh
```
