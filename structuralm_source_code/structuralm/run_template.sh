#!/usr/bin/env bash
set -u 
set -o pipefail

start_time=$(date +%s)

export MY_LLM_API_KEY=""
export MY_LLM_BASE_URL=""
export SCENE_MODEL_NAME=""
export MY_HUGGINGFACE_TOKEN=""

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

PYTHONPATH=src:. python -m complete_pipeline.main "$@"
status=$?

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

hours=$(( elapsed / 3600 ))
mins=$(( (elapsed % 3600) / 60 ))
secs=$(( elapsed % 60 ))

printf "[PIPELINE] Total elapsed time: %02d:%02d:%02d (exit code %d)\n" \
  "$hours" "$mins" "$secs" "$status"

exit $status