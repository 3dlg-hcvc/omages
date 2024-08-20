# utils for hugging face
import re
import os
import sys
import glob
import numpy as np
from xgutils import nputil, sysutil

from huggingface_hub import upload_file, list_repo_files, hf_hub_download

def split_and_upload(local_path, target_dir, repo_id, repo_type, split_size='45G'):
    """
    Split the file and upload to hugging face
    """
    # split the file
    file_dir  = os.path.dirname(local_path)
    file_name = os.path.basename(local_path)
    local_split_target = file_dir + '/' + file_name + '_part_'
    target_name = file_name + '_part_'
    command = f"split -d -a 5 -b {split_size} {local_path} {local_split_target}"
    """
        split: The command used to split files.
        -d: Use numeric suffixes starting from 00001 (default is 0000).
        -a 5: Specifies that the suffix length is 5 characters.
        -b 100M: Specifies the size of each split file. You can adjust 100M to the desired size for each part.
        local_path: The file you want to split.
        local_split_target: The prefix for the split files. The parts will be named ..._part_00001, etc.
    """
    os.system(command)
    # upload the file
    splited_files = glob.glob(local_split_target + '*')
    for i, file in sysutil.progbar(enumerate(splited_files)):
        part_name = f'{target_name}{str(i).zfill(5)}'
        target_path = os.path.join(target_dir, part_name)
        upload_file(path_or_fileobj = file, 
                    path_in_repo    = target_path, 
                    repo_id         = repo_id, 
                    repo_type       = repo_type
                    )
    # remove the split files
    for file in splited_files:
        print(f"Removing {file}")
        os.remove(file)
    return splited_files
def download_and_merge(hub_path, local_target_dir, repo_id, repo_type):
    """
    Download the files and merge them
    """
    os.makedirs(local_target_dir, exist_ok=True)
    all_repo_file_paths = list_repo_files(repo_id = repo_id, repo_type = repo_type)
    # find the files that match the pattern
    matched_files = []
    print(len(all_repo_file_paths))
    for file in all_repo_file_paths:
        if hub_path in file:
            matched_files.append(file)
    print(matched_files)
    # download the files
    local_files = []
    for i, file in sysutil.progbar(enumerate(matched_files)):
        local_file = os.path.join(local_target_dir, os.path.basename(file))
        hf_hub_download(filename=file, local_dir=local_target_dir, repo_id=repo_id, repo_type=repo_type,)
        local_files.append(local_file)
    
    # merge the files
    file_name = os.path.basename(hub_path)
    hub_path_pattern = hub_path + '_part_*' 
    src = os.path.join(local_target_dir, hub_path_pattern)
    dest = os.path.join(local_target_dir, file_name)
    command = f"cat {src} > {dest}"
    print(command)
    os.system(command)
    # remove the split files
    for file in local_files:
        print(f"Removing {file}")
        os.remove(file)
    return dest
    
if __name__ == '__main__':
    pass
    #split_and_upload('/studio/temp/test2/test.blend', './', '3dlg-hcvc/omages_ABO', 'dataset', split_size='10M')
    # download_and_merge('test.blend', '/studio/temp/test3/', '3dlg-hcvc/omages_ABO', 'dataset')
