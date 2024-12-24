import os

commands = [
    f"torchrun --nnodes=1 --nproc_per_node=8 generate.py \
        --model SiT-XL/2 \
        --num-fid-samples 5000 \
        --path-type=linear \
        --encoder-depth=8 \
        --projector-embed-dims=768 \
        --per-proc-batch-size=64 \
        --mode=sde \
        --num-steps=250 \
        --cfg-scale={cfg} \
        --skip=6 \
        --guidance-high={ghi}"
    for ghi in [0.5, 0.4, 0.3]
    for cfg in [2.0, 1.8]
]

for c in commands:
    print(c)

for cmd in commands:
    os.system(cmd)

'''

cfg 2.0 - ghi 0.5 0.4 0.3 0.2 0.1
cfg 1.8 - ghi 0.5 0.4 0.3 0.2 0.1
cfg 1.65 - ghi 1.0 0.9 0.8 0.7 0.6 0.5 0.4
cfg 1.35 - ghi 1.0 0.9 0.8 0.7 0.6 0.5 0.4

'''