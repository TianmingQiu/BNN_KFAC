import imageio
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

result_path = '/root/Documents/BNN_KFAC/results/Hessian/images/'
img_name = 'H_inv_15k_dense_foot.png'
file_path = result_path + img_name

a = 0.04
# cmap = LinearSegmentedColormap.from_list('', [(0,'k'),(0.05,'purple'),(0.10,'b'),(0.15,'cyan'),(0.20,'g'),(0.25,'yellow'),(1,'salmon')])
cmap = LinearSegmentedColormap.from_list('', [(0,'k'),(a,'purple'),(2*a,'b'),(3*a,'cyan'),(4*a,'g'),(5*a,'yellow'),(1,'salmon')])

im = imageio.imread(file_path)
i = torch.FloatTensor(im) / 255
print(i.max())
plt.figure(figsize = (12,8))

# plot = sns.heatmap(i[:,:,0], xticklabels=False, yticklabels=False, square=True, cbar=False)
plot = sns.heatmap(i[:,:,0], vmin=0, vmax=0.6, xticklabels=False,\
     yticklabels=False, square=True, cbar=1, cmap=cmap)

# plot.figure.savefig(result_path + 'heatmap/' + img_name)
# plt.savefig(result_path + 'heatmap/' + img_name)
print(result_path + 'heatmap/' + img_name)

# /root/Documents/BNN_KFAC/results/Hessian/images
# sns.heatmap(i[:,:,0], vmin=0, vmax=0.6, xticklabels=False, yticklabels=False)
# plt.savefig('/root/Documents/BNN_KFAC/results/Hessian/images/heatmaps/' + 'error_15k_foot.png')

# H_15k_foot
# H_750_dense
# error_750
# error_15k_foot
# H_inv_750_dense
# H_inv_15k_dense_foot