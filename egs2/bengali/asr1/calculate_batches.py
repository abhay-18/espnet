# save as compute_batches.py and run: python compute_batches.py /path/to/shape.txt 16000000 --feat-dim 80 --ngpu 6 --batch_type numel

import sys
import math
from pathlib import Path
import argparse

def load_shape(shape_path):
    # expects lines like: utt_id <n_frames>  OR utt_id <n_frames> <feat_dim> depending on format.
    total_frames = []
    # speech_path = "/nfs/abhay/espnet/egs2/bengali/asr1/exp/bn_opensource_bf_120M/asr_stats_raw_bn_bpe1024_sp/train/speech_shape"
    with open(shape_path, "r") as f:
        # i = 1
        for line in f:
            if not line.strip():
                continue
            # print(i)
            utt_id, nos = line.strip().split(maxsplit=1)
            # print(int(nos))
            frames = ((int(nos) - 400 )// 160) + 1
            # print(frames) if i == 1 else None
            total_frames.append(frames)
            # i += 1
    return total_frames

def simulate_batches(frames, batch_bins, batch_type='numel', feat_dim=80):
    # Sort descending (ESPnet often sorts by length buckets — descending gives a good estimate)
    # frames_sorted = sorted(frames, reverse=True)
    frames_sorted = frames
    batches = []
    cur_sum = 0
    for f in frames_sorted:
        units = f * (feat_dim if batch_type=='numel' else 1)
        if cur_sum + units <= batch_bins:
            cur_sum += units
        else:
            batches.append(cur_sum)
            cur_sum = units
            # if a single item > batch_bins, it still becomes its own batch
            if cur_sum > batch_bins:
                # allow single-big-utterance batch
                batches.append(cur_sum)
                cur_sum = 0
    if cur_sum > 0:
        batches.append(cur_sum)
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shape_path')
    parser.add_argument('batch_bins', type=int)
    parser.add_argument('--feat-dim', type=int, default=80)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_type', choices=['numel','frames'], default='numel')
    args = parser.parse_args()

    frames = load_shape(args.shape_path)
    if len(frames)==0:
        print("No frames loaded. Check file format.")
        return
    batches = simulate_batches(frames, args.batch_bins, args.batch_type, args.feat_dim)
    print(f"Total utterances: {len(frames)}")
    print(f"Total batches: {len(batches)}")
if __name__ == '__main__':
    main()
