import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download

output_folder = 'datasets/ABO/omages/'
os.makedirs(output_folder, exist_ok=True)

part1 = hf_hub_download(repo_id='3dlg-hcvc/omages_ABO', filename='omages_ABO_p64_m02_1024.tar_partaa', repo_type="dataset", local_dir=output_folder)
part2 = hf_hub_download(repo_id='3dlg-hcvc/omages_ABO', filename='omages_ABO_p64_m02_1024.tar_partab', repo_type="dataset", local_dir=output_folder)

dest = os.path.join(output_folder, 'omages_ABO_p64_m02_1024.tar')
src  = os.path.join(output_folder, 'omages_ABO_p64_m02_1024.tar_parta*')
command = "cat {} > {}".format(src, dest)
result = subprocess.run(command, shell=True, check=True)
# then untar the file


