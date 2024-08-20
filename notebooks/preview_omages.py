''' 
This is an interactive notebook for previewing the omages dataset.
For more information for using .py file as jupyter notebooks, please refer to:
https://code.visualstudio.com/docs/python/jupyter-support-py

'''
#%% First run this command, and then select the dlt enviroment
from ipynb_header import *
# set cuda 
cwd = os.getcwd()
print( cwd )
# torch.cuda.set_device(0)
#%%
from xgutils import omgutil 
example_omage_1024 = 'assets/B0742FHDJF_objamage_tensor_1024.npz'
omage = omgutil.load_omg(example_omage_1024)
vomg, rdimg, _ = omgutil.preview_omg(omage)
visutil.showImg( vomg )
#%% #################### You can also save the decoded omage to .blend and .glb files
os.makedirs(f'{cwd}/temp', exist_ok=True)
vomg, rdimg, _ = omgutil.preview_omg(omage, decoder_kwargs=dict(render_mode='segcolor', save_path=f'{cwd}/temp/omage_preview_B0742FHDJF.blend'))
# a .glb file with the same name will be saved in the same directory
# You can use the 'glTF Model Viewer' plugin in vscode to directly see it!

#%% #################### See the patch segmentation of the omage
vomg, rdimg, _ = omgutil.preview_omg( omage, geometry_only=True, 
                            render_kwargs=dict(camera_position=(2., -2., 1), shadow_catcher=True), decoder_kwargs=dict(render_mode='segcolor', save_path=None),
                        )
visutil.showImg( vomg )


#%% #################### Downsample the omage to 64 with *sparse pooling*
omage64_dict = omgutil.downsample_omg(omage, factor=16) # 1024 to 64
omage64 = omage64_dict['omg_down_star']
vomg64, rdimg64, _ = omgutil.preview_omg(omage64, geometry_only=True, decoder_kwargs=dict(render_mode='segcolor', save_path=None))
visutil.showImg( vomg64 )

#%% #################### %% Direct downsampling without *sparse pooling*
omage64_direct = omage[::16, : :16]
vomg64_direct, rdimg64_direct, _ = omgutil.preview_omg(omage64_direct, geometry_only=True, decoder_kwargs=dict(render_mode='segcolor', save_path=None))
visutil.showImg( vomg64_direct )

# %%
