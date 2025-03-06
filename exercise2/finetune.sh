#!/bin/bash


__conda_setup="$('/home/quent/.local/bin/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/quent/.local/bin/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/quent/.local/bin/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/quent/.local/bin/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate hf


export WANDB_PROJECT=stack-overflow

BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=8
GAS=$(( $BATCH_SIZE / $PER_DEVICE_BATCH_SIZE ))
EPOCHS=2
STEPS_PER_EPOCH=$(( 45000 / $BATCH_SIZE ))
NUM_STEPS=$(( $STEPS_PER_EPOCH * $EPOCHS ))
LOGGING_STEPS=$(( $STEPS_PER_EPOCH / 4 ))
EVAL_STEPS=$(( $LOGGING_STEPS ))

python run_classification.py \
    --model-name-or-path roberta-title-results/train \
    --text-column-names Title \
    --label-column-name Y \
    --train-file stack_overflow_questions_train.csv \
    --validation-file stack_overflow_questions_valid.csv \
    --output-dir roberta-title-results/finetune \
    --do-train --do-eval \
    --eval-strategy steps \
    --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
    --per-device-eval-batch-size 256 \
    --gradient-accumulation-steps $GAS \
    --eval-steps $EVAL_STEPS \
    --logging-steps $LOGGING_STEPS \
    --learning-rate 1e-5 \
    --weight-decay .9 \
    --no-pad-to-max-length \
    --num-train-epochs $EPOCHS \
    --lr-scheduler-type cosine \
    --log-level info \
    --logging-strategy steps \
    --save-strategy best \
    --save-total-limit 5 \
    --metric-for-best-model accuracy \
    --optim adamw_torch \
    --report-to wandb



