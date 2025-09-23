#!/bin/bash

set -x 

# Run from project root
cd "$(dirname "$0")/.."

llamafactory-cli train LLM/finetune/scripts/finetune_llama-3.2-1b_sft.yaml