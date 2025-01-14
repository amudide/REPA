from tqdm import tqdm
import os
from PIL import Image
import numpy as np

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    missing = 0
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        try:
            sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
        except:
            missing += 1
    samples = np.stack(samples)
    #assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    print(samples.shape)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    print(f"Missing {missing}")
    return npz_path

#create_npz_from_sample_folder("samples/EXP2_SiT-XL-2-SiT-XL-2-256x256-size-256-vae-ema-cfg-1.6-glo-0.0-ghi-0.9-skip-[6]-samples-50000-seed-0-sde")


sample_folder_dir = "samples/EXP2_SiT-XL-2-SiT-XL-2-256x256-size-256-vae-ema-cfg-1.6-glo-0.0-ghi-0.9-skip-[6]-samples-50000-seed-0-sde.npz"
ref_path = "VIRTUAL_imagenet256_labeled.npz"

commands = [
    f'python evaluations/evaluator.py evaluations/{ref_path} "{sample_folder_dir}" > "results/{sample_folder_dir[8:-4]}.txt"'
]

for c in commands:
    print(c)

for cmd in commands:
    os.system(cmd)
