import os

def suffix(s):
    layers = [i for i in range(s, 28)]
    return ' '.join([str(x) for x in layers])

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
        --skip {skip} \
        --guidance-high=0.7"
    for skip in [suffix(i) for i in [27, 26, 25, 24, 23, 22, 21, 20, 19, 18]]
    for cfg in [1.25, 1.5, 1.8]
]

for c in commands:
    print(c)

for cmd in commands:
    os.system(cmd)