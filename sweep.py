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
        --cfg-scale={j} \
        --skip {i} \
        --guidance-high=0.7"
    for j in [1.5, 1.8, 2.0, 1.25]
    for i in range(28)
]

for c in commands:
    print(c)

for cmd in commands:
    os.system(cmd)