#!/bin/bash
set -euo pipefail

mkdir -p logs results

SIZES=("small" "medium" "large" "xl" "2.7B")

echo "Using python: $(which python)"
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)"

for mode in forward forward_backward; do
  for size in "${SIZES[@]}"; do
    echo "Running size=${size} mode=${mode}"
    python benchmark.py \
      --model_size "${size}" \
      --context_length 512 \
      --batch_size 4 \
      --num_warmup 5 \
      --num_iters 10 \
      --mode "${mode}" \
      | tee "logs/${size}_${mode}.log"
  done
done

echo "Done."