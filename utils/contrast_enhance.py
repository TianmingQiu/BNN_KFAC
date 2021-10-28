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
func = lambda x:  3761.649 + (0.001531456 - 3761.649)/(1 + (x/453399700)**0.4157757)
im = enhancer(result_path + 'error_15k_foot.png', func)
im.save(result_path + 'reinforced/' + 'error_15k_foot.png')
