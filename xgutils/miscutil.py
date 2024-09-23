import os
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont
#from xgutils import *




ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))+"/assets/"
preset_glb = preset_blend = os.path.join(ASSETS_DIR, "preset_glb.blend")
def load_mesh(name="Utah.obj"):
    import igl
    face, vert = igl.read_triangle_mesh(os.path.join(ASSETS_DIR, name) )
    return face, vert
def load_imgfont(name="candarab.ttf", fsize=32):
    font = ImageFont.truetype(os.path.join(ASSETS_DIR, name), fsize)
    return font
def get_fontpath(name="candarab.ttf", fsize=32):
    return os.path.join(ASSETS_DIR, name)
    

