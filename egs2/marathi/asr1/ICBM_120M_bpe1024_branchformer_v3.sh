#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="valid"

asr_config=conf/marathi_opensource_120M_bpe1024_branchformer.yaml
inference_config=conf/decode_transducer.yaml

# --skip_stages \
./asr_new.sh \
    --lang mr \
    --stage 11 \
    --stop_stage 1000 \
    --ngpu 3 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 16 \
    --nbpe 1024 \
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
    # --pretrained_model "${pretrained_model}"
