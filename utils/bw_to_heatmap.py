import imageio
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

result_path = '/root/Documents/BNN_KFAC/results/Hessian/images/'
img_name = 'H_750_dense.png'
file_path = result_path + img_name

cmap = LinearSegmentedColormap.from_list('', [(0,'k'),(0.1,'purple'),(0.2,'b'),(0.3,'cyan'),(0.4,'g'),(0.5,'yellow'),(1,'salmon')])
im = imageio.imread(file_path)
i = torch.FloatTensor(im) / 255
print(i.max())
plt.figure(figsize = (12,8))

# plot = sns.heatmap(i[:,:,0], xticklabels=False, yticklabels=False, square=True, cbar=False)
plot = sns.heatmap(i[:,:,0], vmin=0, vmax=1, xticklabels=False, yticklabels=False, square=True, cbar=0, cmap=cmap)

# plot.figure.savefig(result_path + 'heatmap/' + img_name)
# plt.savefig(result_path + 'heatmap/' + img_name)
print(result_path + 'heatmap/' + img_name)

# /root/Documents/BNN_KFAC/results/Hessian/images
# sns.heatmap(i[:,:,0], vmin=0, vmax=0.6, xticklabels=False, yticklabels=False)
# plt.savefig('/root/Documents/BNN_KFAC/results/Hessian/images/heatmaps/' + 'error_15k_foot.png')