import imageio
import torch
from PIL import Image
import numpy as np

def enhancer(filePath, funcHandler = lambda x: x ** 0.5):
    im = imageio.imread(filePath)
    i = torch.FloatTensor(im) / 255
    i = funcHandler(i)
    image = Image.fromarray(np.uint8(255*i.numpy())).convert('RGB')
    return image

result_path = 'results/Hessian/images/'
im = enhancer(result_path + 'error_750.png', lambda x: x**0.5)
im.save(result_path + 'tmp.png')