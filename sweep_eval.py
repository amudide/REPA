import os

ref_path = "VIRTUAL_imagenet256_labeled.npz"

folder_path = "samples/npzs/"
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

sample_folder_dirs = []
for f in files:
    sample_folder_dirs.append(folder_path + f)

commands = [
    f'python evaluations/evaluator.py evaluations/{ref_path} "{sample_folder_dir}" > "results/{sample_folder_dir[13:-4]}.txt"'
    for sample_folder_dir in sample_folder_dirs
]

for c in commands:
    print(c)

for cmd in commands:
    os.system(cmd)