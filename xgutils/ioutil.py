import h5py
import numpy as np
def write_list_of_list(f5file, name, array):
    grp=f5file.create_group(name)
    for i, subarray in enumerate(array):
        grp.create_dataset('%d'%i,data=subarray)
def read_list_of_list(f5file, name):
    length = len(f5file[name].keys())
    array_list = []
    for i in range(length):
        array_list.append( np.array(f5file[name]['%d'%i]))
    return np.array(array_list)
