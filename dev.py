import numpy as np
import torch

split = 'dev'
data_path = '/home/ubuntu/Students/zhoutao/data/coco_precomp/%s_boxes.npy' % split
ims = np.load(data_path)
ims = torch.tensor(ims)
ims = ims.repeat_interleave(5, 0)
print(ims.shape)
np.save(data_path, ims)
