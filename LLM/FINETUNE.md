# Fine-tuning with LLaMA-Factory

This guide explains how to set up and run fine-tuning for models using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which is included as a Git submodule in this project.

---

## 1. Initialize the submodule

If you cloned the repo without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

## 2. Install dependencies

From the **project root directory**, run:

```bash
pip install -e externals/LLaMA-Factory
```
