"""This module contains useful methods for numpy and math.

The methods are gathered in these groups:
    Math utilities
    Array operations
    Linear algebra
    H5 dataset utilities
"""
from __future__ import print_function
import re
import os
import sys
import shutil
import traceback

from collections.abc import Iterable

import h5py
import numpy as np
import pandas as pd
import tables as tb

import torch

from xgutils import nputil, sysutil
from torch.utils.data import Dataset

import glob
def gen_datalist(dset_root, data_dir, name_list=None, list_name="dset_list", splits=[.8,.1,.1], shuffle=True, seed=314):
    """ 
        dset_root/data_dir/ stores the data items
        dset_root/dset_list.txt stores the list of data files
    """
    split_names = ["train", "val", "test"]
    if name_list is None:
        ditem_paths = glob.glob(os.path.join(dset_root, data_dir, "*"))
        ditem_rel_paths = [os.path.relpath(ditem_path, dset_root) for ditem_path in ditem_paths]
    else:
        ditem_paths = [dset_root+"/"+data_dir+"/"+name for name in name_list]
        ditem_rel_paths = [data_dir+"/"+name for name in name_list]
    splits = np.array(splits)
    assert np.sum(splits) == 1
    with nputil.temp_seed(seed):
        if shuffle:
            np.random.shuffle(ditem_paths)
        total_num = len(ditem_paths)
        split_num = np.round(splits * total_num).astype(int)
        part_num = np.floor(splits * total_num).astype(int)
        part_num[-1] = total_num - np.sum(part_num[:-1]) # make sure the sum is correct
        part_bias = np.zeros(len(part_num)+1, dtype=int)
        part_bias[1:] = np.cumsum(part_num)
        
    np.savetxt(os.path.join(dset_root, f"{list_name}.txt"), ditem_rel_paths, fmt="%s")
    for i, split in enumerate(splits):
        name = os.path.join(dset_root, f"{list_name}_{split_names[i]}.txt".format())
        ditem_name = ditem_rel_paths[part_bias[i]:part_bias[i+1]]
        # sort the list
        ditem_name = sorted(ditem_name)
        np.savetxt(name, ditem_name, fmt="%s")

def generate_split(N, splits=[.8,.1,.1], shuffle=True, seed=314):
    """ return a list of splits of np.arange(N)
    """
    # split np.arange(N) into splits
    splits = np.array(splits)
    assert np.sum(splits) == 1
    index = np.arange(N)
    with nputil.temp_seed(seed):
        if shuffle:
            np.random.shuffle(index)
        accu_splits = np.round(np.cumsum(splits) * N).astype(int)
        assert accu_splits[-1] == N
        splits = np.split(index, accu_splits)[:-1]
    return splits

def batch_generator_for_ranks(dataset, ind_list=None, batch_size=1, global_rank=0, world_size=1):
    """ return a batch iterator for ranks
    """
    if ind_list is None:
        N = len(dataset)
        ind_list = np.arange(N)
    else:
        N = len(ind_list)
    # extend N to make it divisible by world_size * batch_size
    wb = world_size * batch_size
    # find the next multiple of wb
    N_ext = (N + wb - 1) // wb * wb
    # extend ind_list to N_ext with random padding
    ind_list_ext = np.concatenate([ind_list, np.random.choice(ind_list, N_ext - N, replace=True)])
    
    # Split the index list into equal parts, one for each rank
    ind_list_splits = np.array_split(ind_list_ext, world_size)
    
    # Get the part of the index list that corresponds to the current rank
    rank_indices = ind_list_splits[global_rank]

    for i in range(0, len(rank_indices), batch_size):
        batch_inds = rank_indices[i:i + batch_size]
        batch = [dataset[idx] for idx in batch_inds]
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch["ditem_ind"] = batch_inds
        yield batch

def npyfolder2dict(folder_path, keys=None):
    ditem = dict()
    if not folder_path.endswith("/"):
        folder_path += "/"
    for rkey in glob.glob(folder_path+"*.npy"): # will ignore names starting with .
        key = rkey.split("/")[-1].replace(".npy","")
        if keys is not None and key not in keys:
            continue
        ditem[key] = np.load(rkey, allow_pickle=True)
        # check if loaded data is pickled
        if isinstance(ditem[key], np.ndarray) and ditem[key].dtype == np.dtype('O'):
            ditem[key] = ditem[key].item()
    return ditem
def dict2npyfolder(folder_path, ditem, keys=None, compress=False):
    sysutil.mkdirs([folder_path])
    save_paths = dict()
    for key, val in ditem.items():
        if keys is not None and key not in keys:
            continue
        if compress == False:
            save_path = os.path.join(folder_path, key+".npy")
            np.save(save_path, val, allow_pickle=True)
        else:
            save_path = os.path.join(folder_path, key+".npz")
            np.savez_compressed(save_path, arr_0=val, allow_pickle=True)
        save_paths[key] = save_path
    return save_paths
def html5_imgGrid(image_folder_path, target_dir, columns=5, height=400, mode="video", inplace=True):
    # Get the list of image filenames
    tar_img_folder = target_dir+"/imgs/"
    sysutil.mkdirs([image_folder_path, tar_img_folder])
    image_filenames = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

    import glob
    if mode == "video" or mode == "gif" or mode == "all":
        suffix = "mp4"
        out_suffix = "mp4" if mode == "video" else "gif"
        oimage_filenames = glob.glob(image_folder_path+f"/*.{suffix}")
    if mode == "image" or mode == "all":
        out_suffix = suffix = "png"
        oimage_filenames = glob.glob(image_folder_path+f"/*.{suffix}")
    # elif mode == "gif":
    #     suffix = "mp4"
    #     oimage_filenames = glob.glob(image_folder_path+f"/*.{suffix}")
   
    image_filenames = []
    for fpath in oimage_filenames:
        if "summary" in fpath:
            continue
        # copy to tar_img_folder
        fname = os.path.basename(fpath)
        # check if is mp4 file, if in gif mode, convert .mp4 to .gif
        if mode == "gif" and fname.endswith(".mp4"):
            tar_name = "imgs/"+fname.replace(".mp4", ".gif")
            target_path = target_dir + tar_name
            os.system(f"ffmpeg -i {fpath} -vf scale=-1:-1 -r 10 {target_path}")
            image_filenames.append(tar_name)
        else:
            copyto = tar_img_folder + fname
            os.system(f"cp {fpath} {tar_img_folder}")
            image_filenames.append("imgs/"+fname)

    image_filenames = sorted(image_filenames)

    # Open or create HTML file
    with open(f'{target_dir}/images_table.html', 'w') as html_file:
        # Write the HTML header
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html lang="en">\n')
        html_file.write('<head>\n')
        html_file.write('    <meta charset="UTF-8">\n')
        html_file.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        html_file.write('    <title>Image Gallery</title>\n')
        html_file.write('    <style>\n')
        html_file.write('       figure {\n')
        html_file.write('           display: inline-block;\n')
        html_file.write('           text-align: center;\n')
        html_file.write('           margin: 2px; \n')
        html_file.write('       }\n')
        html_file.write('       figcaption {\n')
        html_file.write('           text-align: center;\n')
        html_file.write('       }\n')
        html_file.write('    </style>\n')
        html_file.write('</head>\n')
        html_file.write('<body>\n')
        html_file.write('    <table>\n')

        # Iterate through the images and create table rows
        img_per_row = columns
        for row in range(0, len(image_filenames), img_per_row):  # 4 images per row
            html_file.write('        <tr>\n')
            for col in range(row, min(row + img_per_row, len(image_filenames))):
                html_file.write(f'            <td style = "border-right: 1px dashed green;"><div>')
                img_path = image_filenames[col]
                if img_path.endswith(".gif") or img_path.endswith(".png"):
                    html_file.write(f'<img src="{img_path}" alt="Image {col}" height="{height}">')
                elif img_path.endswith(".mp4"):
                    html_file.write(f'<video height="{height}" controls autoplay loop muted><source src="{img_path}" type="video/mp4">Your browser does not support the video tag.</video>')
                caption = os.path.basename(img_path)
                html_file.write(f"<figcaption>{caption}</figcaption>")
                html_file.write(f'</div></td>\n')
            html_file.write('        </tr>\n')

        # Write the HTML footer
        html_file.write('    </table>\n')
        html_file.write('</body>\n')
        html_file.write('</html>\n')

    # os.chdir("/studio/")
    # # Path to the directory containing images
    # image_folder_path = "dl_template/experiments/v31_cos-schedule_condpolyfusion_onshape/results/LoopfusionVisualizer/visual"
    # html5_imgGrid(image_folder_path, "/studio/dl_template/temp/v31_cos_polyloopfusion_gif/", columns=5, height=300, mode="gif")

def npzdset2npydset(data_path, tar_path):
    # get last non-empty folder name of data_path
    data_path = data_path.rstrip("/")
    data_folder_name = os.path.basename(data_path)
    par_dir = os.path.dirname(data_path)
    assert data_folder_name != ""

    dfiles = glob.glob(data_path+"/*.npz")
    if tar_path is None or tar_path == data_path:
        tar_path = os.path.join(par_dir, data_folder_name+"_npy/")
    for dfile in sysutil.progbar(dfiles):
        ditem_name = os.path.basename(dfile).replace(".npz", "")
        tar_ditemp = os.path.join(tar_path, ditem_name+"/")
        ditem = np.load(dfile, allow_pickle=True)
        dict2npyfolder(tar_ditemp, ditem )

    #ditem = np.load(npz_path, allow_pickle=True)
    #np.save(npy_path, ditem)

def df2dset(df, dset_root, description=""):
    """
    df: pandas dataframe with columns: path, caption, ...
    dset_root: the root folder of the dataset
    """
    sysutil.mkdirs([dset_root])
    for i in range(len(df)):
        ditem = df.iloc[i].to_dict()
        ditem_dir = os.path.join(dset_root, str(df.index[i]))
        sysutil.mkdirs([ditem_dir])
        dict2npyfolder(ditem_dir, ditem)
    df['mask'] = 1
    save_df = df['mask']
    save_df.to_jason(os.path.join(dset_root, "df_index.json"))
    meta_df = pd.DataFrame([dict(description=description, num_items=len(df), columns=df.columns)])
    meta_df.to_json(os.path.join(dset_root, "df_meta.json"), orient='records', indent=4)
class NpyListDataset(Dataset):
    def __init__(self, dset_list=None, duplicate = 1, keys=dict(), **kwargs):
        """ 
            The following is the structure of the dset folder:
            dset_root/
                dset_list.txt
                dset/
                    0000/
                        *.npy
                    0001/
                        *.npy
                    ...
        """
        super().__init__()
        self.__dict__.update(locals())
        #self.ditem_list = np.loadtxt(dset_list, dtype=str)
        self.ditem_index = pd.read_json(dset_list)
        self.list_dir = os.path.dirname(dset_list)
    def __len__(self):
        return len(self.ditem_list) * self.duplicate
    def ind2name(self, ind):
        ind = ind % len(self.ditem_list)
        return str(self.ditem_list[ind]).split('/')[-1].split('.')[0]
    def ind2path(self, ind):
        ind = ind % len(self.ditem_list)
        return os.path.join(self.list_dir, self.ditem_list[ind])
    def __getitem__(self, ind, keys=None):
        ind = ind % len(self.ditem_list)
        ditem_name = self.ditem_list.iloc[ind] 
        ditem_path = os.path.join(self.list_dir, ditem_name)
        ditem = {}
        if keys is None:
            keys = self.keys
        for key in keys:
            dkey_path = os.path.join(ditem_path, key + '.npy')
            ditem[key] = np.load(dkey_path, allow_pickle=True)
            if ditem[key].dtype == np.dtype('O'):
                ditem[key] = ditem[key].item()
        return ditem

# data vis
def path2html(path, height=100):
    if path.endswith('.png'):
        html = f'<img src="{path}" height="{height}" alt="image">'
    elif path.endswith('.mp4'):
        html = f'<video controls playsinline autoplay muted loop> <source src="{path}" height="{height}" alt="video" type="video/mp4"> </video>'
    elif path.endswith('.glb'):
        splits = path.split('|')
        if len(splits) == 1: # no preview
            html = f'<model-viewer src="{path}" alt="3D model" auto-rotate camera-controls style="height: {height};"></model-viewer>'
        else:
            preview_path = splits[0]
            glb_path = splits[1]
            # html = f"""<div style="height: {10};"> <img src="{preview_path}" alt="Image" style="height: '{height}'; cursor: pointer;" onclick="this.style.display='none'; this.nextElementSibling.style.display='block';"> <model-viewer src="{glb_path}" alt="A 3D model" auto-rotate camera-controls style="height: {height}; display: none;"></model-viewer></div>"""
            html = f'<div class="container" onclick="this.querySelector("model-viewer").dismissPoster();"><model-viewer style="height: {height}px;" poster="{preview_path}" reveal="interaction" src="{glb_path}" alt="3D model" auto-rotate camera-controls style="height: {height};"></model-viewer></div>'
            html = f'<model-viewer style="height: {height}px;" poster="{preview_path}" reveal="interaction" src="{glb_path}" alt="3D model" auto-rotate camera-controls style="height: {height};"></model-viewer><button id="load-model-button" onclick="this.previousElementSibling.dismissPoster()">Load Model</button>'
    else:
        return path
    return html
def itablize(df, show=True, maxBytes='1MB'):
    import itables 
    itables.init_notebook_mode(all_interactive=True)
    from IPython.core.display import HTML
    HTML("""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    """)
    idf = df.applymap( lambda x: path2html(x) if type(x) is str else x)
    itables.show(idf, maxBytes=maxBytes)
    return idf
def itable2html(html_path, idf, maxBytes='1MB'):
    import itables
    
    html_str = itables.to_html_datatable(idf, maxBytes=maxBytes)
    modelviewer_html = """<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js">tesst</script>"""
    html_str = html_str + modelviewer_html 
    if html_path is not None:
        with open(html_path, 'w') as f:
            f.write(html_str)
    return html_str

def dataset_to_hdf5(dset, hdf5_path):
    """
    Dump a PyTorch Dataset into an HDF5 file.

    :param dset: Instance of torch.utils.data.Dataset
    :param hdf5_file: Path to the output HDF5 file.
    """
    with tb.open_file(hdf5_path, "w") as h5file:
        for idx in sysutil.progbar(range(len(dset))):
            data_item = dset[idx]
            group = h5file.create_group("/", f"ditem_{idx}", f"Data item {idx}")
            for key, data in data_item.items():
                h5file.create_array(group, key, data)
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.__dict__.update(locals())
    def __len__(self):
        return len(self.h5file.root)
    def __getitem__(self, idx):
        self.h5file = tb.open_file(self.hdf5_path, "r")
        group = self.h5file.get_node(f"/ditem_{idx}")
        data_dict = {}
        for array in group:
            data_dict[array.name] = np.array(array.read())
        self.h5file.close()
        return data_dict
    def close(self):
        self.h5file.close()
    def __del__(self):
        self.close()

class DatasetProcessor():
    """
        DataProcessor turns a table of input to another table of input. By table we mean pandas dataframe.

        The table entry has these types: file path (string), text, number, bool.

        We think of any datasets as such table. The data content like images, polygonal meshes are regarded as file path. We also assume that for each row files are in the same folder of that path 'ditem_dir'

        We categorize the columns into "meta" and "data" columns. The "meta" columns are meta-information of the data, like image size, whether the data is valid, etc.
        And the "data" columns are the file paths toward the actual data. Or sometimes when the dataset is small, we store numpy arrays directly in the "data" columns.

        DataProcessor is designed to be robust, parallelizable, and easy to use. 
        In dataset processing, we often encounter failures for some certain data items.
        Even one failure will ruin the whole process, if using a simple for loop for processing.
        And manytimes we want to process the data in parallel, to greatly speed up the process.
        Also we might want to keep track of the version of the processed dataset and the description of the processing.
        We also want the data processing script to be aligned with the pytorch dataset script.
        DataProcessor is designed to solve all these problems.

        Key components of DataProcessor are "Generator" "Processor" "Recorder" "Handler".
        
        The Generator takes a config dict and create the initial pandas table.
        
        The Processor is the DatasetProcessor at the row level, that is, a function that takes an input data row and returns an output data row.
        Usually it reads the file paths in the input data row into memory (like images, meshes, ...)

        Recorder record the meta info of the parser output, like whether the parsing is successful (no errors or exceptions), the meta-info of the output, like the number of faces of the processed meshes. Also, recorder is able to know whether an item is processed or not and is able to check if some previously processed items follow some new criterions.

        Handler deals with errors and exceptions encountered during the process. In this way we can skip and record the error message for the failed data items without ruining the whole process.
    """
    def __init__(self, config, force_overwrite=False, filemode=True, debug=False):
        self.__dict__.update(locals())
        self.ditems=dict()
        self.target_root = None # not none for filemode
    def generator(self,):
        # base_df = pd.read_pickle("/studio/datasets/ABO/df_meta.pkl")
        # input_df = input_df.sort_values(by="path")
        # glb_names = input_df.path
        # ids = input_df.index
        # return input_df
        self.force_overwrite = force_overwrite
        input_df = []
        for i in range(20):
            input_df.append( dict(testv=i) )
        input_df = pd.DataFrame(input_df)
        return input_df

    def row_processor(self, glb_path, **kwargs):
        pass
    
    def verifier(self, output_ditem):
        if output_ditem is None:
            return False
        return output_ditem["_success"]
    def load_ditem(self, index):
        if self.filemode==True:
            if os.path.exists(ditem_path):
                ditem = pd.read_json(ditem_path)
                ditem['_valid'] = self.verifier(output_ditem)
            else:
                ditem['_valid'] = False
        return ditem
    def save_ditem(self):
        pass
    def process_row(self, index, input_ditem):
        try:
            loaded_ditem = None
            if self.force_overwrite==False:
                if self.filemode==True:
                    ditem_dir = input_ditem['target_dir']
                    ditem_path = ditem_dir + '/ditem.json'
                    if os.path.exists(ditem_path):
                        loaded_ditem = pd.read_json(ditem_path).iloc[0].to_dict()
                else:
                    if index in self.ditems:
                        loaded_ditem = self.ditems[index]
                if_valid = self.verifier(loaded_ditem)
                if if_valid == False:
                    loaded_ditem = None

            if loaded_ditem is not None:
                output_ditem = loaded_ditem
            else:
                output_ditem = self.row_processor(**input_ditem)
                output_ditem.update( dict(_success=True, _error_message="") )
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_message = traceback.format_exc()#.replace("\n", " ")
            print(error_message)
            output_ditem = dict(_success=False, _error_message=error_message, _valid=False)
            if self.debug:
                import pdb
                pdb.post_mortem(exc_traceback)  # Pass the traceback to post_mortem
                exit()

        self.ditems[index] = output_ditem

        # save the output_ditem
        if self.filemode==True:
            ditem_dir = input_ditem['target_dir']
            ditem_path = ditem_dir + '/ditem.json'
            sysutil.mkdirs([ditem_dir])
            df = pd.DataFrame([output_ditem], index=[index])
            df.to_json(ditem_path, orient='records', indent=4)

        return output_ditem
    def process(self):
        input_df = self.generator()
        output_df = []
        for i in sysutil.progbar(range(len(input_df))):
            input_ditem   = input_df.iloc[i].to_dict()
            index = input_df.index[i]
            output_ditem = self.process_row(index, input_ditem)
            output_df.append( output_ditem )
        output_df = pd.DataFrame(output_df)
        if self.filemode and self.target_root is not None:
            output_df.to_json(self.target_root + '/dataframe.json', orient='records', indent=4)
        return output_df
    
    # def error_handler(self, func, func_kwargs):
    #     try:
    #         output_ditem = func(**func_kwargs)
    #         output_ditem.update(dict(_success=True, _error_message=""))
    #     except Exception as e:
    #         error_message = traceback.format_exc().replace("\n", " ")
    #         print(error_message)
    #         output_ditem = dict(_success=False, _error_message=error_message)
    # def data_reader(self,):
    #     pass
    # def data_writer(self,):
    #     pass
    # def recorder(self, ditem_dir, func, func_kwargs):
    #     """
    #         The difficult part is to split the read/save logic from the process script, since they are highly intertwined. Split the io enables easy switch between io with files and io with pandas dataframe.
    #     """
    #     if self.force_overwrite==False:
    #         if os.path.exists(ditem_dir + '/ditem.json'):
    #             ditem = pd.read_json(ditem_dir + '/ditem.json')
    #             if self.verifier(ditem) == True:
    #                 return ditem
    #             else:
    #                 print("The ditem is not valid, reprocessing...")
        
    #     ditem = func()
