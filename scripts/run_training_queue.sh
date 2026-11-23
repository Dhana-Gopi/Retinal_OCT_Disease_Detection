#!/usr/bin/env bash
# Sequential GPU training queue (training agent). One job at a time.
# Each run logs to logs/<run>.log. A run is skipped if models/<run>_history.json
# already has status complete or early_stopped. Stops on the first error.
# Usage: bash scripts/run_training_queue.sh      (from anywhere)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/opt/anaconda3/envs/Retinal_OCT_Project/bin/python}
mkdir -p logs models
VAL="--val-limit 4000"

if pgrep -f "src\.train" >/dev/null; then
  echo "[queue] another src.train job is running; refusing to start" >&2; exit 1
fi

is_done() {  # $1 = run name
  local h="models/$1_history.json"
  [[ -f "$h" ]] && "$PY" -c "import json,sys; s=json.load(open('$h')).get('status'); sys.exit(0 if s in ('complete','early_stopped') else 1)"
}

run() {  # $1 = run name, rest = src.train args
  local name=$1; shift
  if is_done "$name"; then echo "[queue] skip $name (already finished)"; return 0; fi
  echo "[queue] $(date '+%F %T') start $name: $*"
  "$PY" -m src.train --run-name "$name" $VAL "$@" > "logs/$name.log" 2>&1 || {
    echo "[queue] $(date '+%F %T') FAILED $name (see logs/$name.log)" >&2; exit 1; }
  echo "[queue] $(date '+%F %T') done  $name"
}

# 0. ResNet18 fine-tune re-run on the stretch-resize cache (pad version archived by lead)
run resnet18_finetune --model resnet18 --mode finetune --epochs 6 --samples-per-epoch 16000 --lr 1e-4 --batch-size 64

# 1. Custom CNN batch-size / learning-rate sweep (SHORT: 2 ep x 6000 scans)
run cnn_bs32_lr1e-3  --model custom_cnn --epochs 2 --samples-per-epoch 6000 --patience 5 --batch-size 32  --lr 1e-3
run cnn_bs64_lr1e-3  --model custom_cnn --epochs 2 --samples-per-epoch 6000 --patience 5 --batch-size 64  --lr 1e-3
run cnn_bs64_lr3e-4  --model custom_cnn --epochs 2 --samples-per-epoch 6000 --patience 5 --batch-size 64  --lr 3e-4
run cnn_bs128_lr1e-3 --model custom_cnn --epochs 2 --samples-per-epoch 6000 --patience 5 --batch-size 128 --lr 1e-3

# 2. Custom CNN regularisation ablation (4 ep x 8000, lr 1e-3, bs 64)
ABL="--model custom_cnn --epochs 4 --samples-per-epoch 8000 --batch-size 64 --lr 1e-3"
run cnn_no_regularisation $ABL --dropout 0 --weight-decay 0 --no-bn
run cnn_no_batchnorm      $ABL --no-bn
run cnn_no_dropout        $ABL --dropout 0

# 3. Main custom CNN with best bs/lr from the sweep (by best_val_macro_f1)
read BS LR < <("$PY" - <<'PYEOF'
import json, pathlib
best = (-1.0, 64, 1e-3)
for name, bs, lr in [("cnn_bs32_lr1e-3", 32, 1e-3), ("cnn_bs64_lr1e-3", 64, 1e-3),
                     ("cnn_bs64_lr3e-4", 64, 3e-4), ("cnn_bs128_lr1e-3", 128, 1e-3)]:
    p = pathlib.Path("models") / f"{name}_history.json"
    if p.exists():
        f1 = json.loads(p.read_text()).get("best_val_macro_f1", -1.0)
        if f1 > best[0]:
            best = (f1, bs, lr)
print(best[1], best[2])
PYEOF
)
echo "[queue] sweep winner: batch size $BS, lr $LR"
run custom_cnn_finetune --model custom_cnn --epochs 8 --samples-per-epoch 16000 \
    --batch-size "$BS" --lr "$LR" --dropout 0.4 --weight-decay 1e-4

# 4-7. Transfer learning
run mobilenet_v2_finetune    --model mobilenet_v2 --epochs 6 --samples-per-epoch 16000 --lr 1e-4 --batch-size 64
run resnet18_feature_extract --model resnet18 --mode feature_extract --epochs 4 --samples-per-epoch 16000 --lr 1e-3 --batch-size 64
run densenet121_finetune     --model densenet121 --epochs 5 --samples-per-epoch 16000 --lr 1e-4 --batch-size 32
run vgg16_finetune           --model vgg16 --epochs 4 --samples-per-epoch 10000 --lr 5e-5 --batch-size 32

echo "[queue] $(date '+%F %T') all runs finished"
