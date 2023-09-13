import os
import glob
import sys
import numpy as np
from PIL import Image
import torch

def read_image(img_path):
    img = Image.open(img_path)
    return img