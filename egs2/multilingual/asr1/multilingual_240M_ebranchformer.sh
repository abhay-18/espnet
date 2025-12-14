#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
# export NCCL_DEBUG_SUBSYS=ALL    # show all NCCL subsystems
export NCCL_DEBUG_FILE=nccl_%r.log
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=1

train_set="train"
valid_set="valid"
test_sets="valid"

asr_config=conf/multilingual_ebranchformer.yaml
inference_config=conf/decode_transducer.yaml

# --skip_stages \
./asr_new.sh \
    --lang hi_bn_mr_ta \
    --stage 11 \
    --stop_stage 11 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 16 \
    --nbpe 4096 \
    --bpe_char_cover 0.99995 \
    --bpe_nlsyms "<|hi|>,<|ta|>,<|mr|>,<|bn|>" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" 
    # --skip_stages "5" \
    # --pretrained_model "${pretrained_model}"
