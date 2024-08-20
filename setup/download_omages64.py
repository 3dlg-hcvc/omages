import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download

output_folder = 'datasets/ABO/omages/'
os.makedirs(output_folder, exist_ok=True)

part1 = hf_hub_download(repo_id='3dlg-hcvc/omages_ABO', filename='df_p64_m02_res64.h5', repo_type="dataset", local_dir=output_folder)
part2 = hf_hub_download(repo_id='3dlg-hcvc/omages_ABO', filename='df_p64_m02_res64_meta.json', repo_type="dataset", local_dir=output_folder)
