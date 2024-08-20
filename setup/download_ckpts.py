import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download

output_folder = 'experiments/omages64/null2geo/checkpoints/'
os.makedirs(output_folder, exist_ok=True)

part1 = hf_hub_download(repo_id='3dlg-hcvc/omages_ABO', filename='N2G_dit_allcate.ckpt', repo_type="dataset", local_dir=output_folder)


output_folder = 'experiments/omages64/geo2mat_imagen/checkpoints/'
os.makedirs(output_folder, exist_ok=True)

part2 = hf_hub_download(repo_id='3dlg-hcvc/omages_ABO', filename='G2M_imagen_allcate.ckpt', repo_type="dataset", local_dir=output_folder)
