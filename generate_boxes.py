import torch
import numpy as np

# precompute bboxes
bboxes = np.load('/home/ubuntu/Students/zhoutao/data/coco_precomp/test_ims_bbx.npy', allow_pickle=True)
ims_size = np.load('/home/ubuntu/Students/zhoutao/data/coco_precomp/test_ims_size.npy', allow_pickle=True)
bboxes = torch.tensor(bboxes, dtype=torch.float64)
ims_size_tensor = torch.zeros(len(ims_size), 2)
for idx, e in enumerate(ims_size):
    ims_size_tensor[idx][0] = e['image_h']
    ims_size_tensor[idx][1] = e['image_w']
    
ims_size = ims_size_tensor.unsqueeze(1).repeat_interleave(36, 1)
bboxes[:, :, 0::2] = bboxes[:, :, 0::2] / ims_size[:, :, 1].unsqueeze(2)
bboxes[:, :, 1::2] = bboxes[:, :, 1::2] / ims_size[:, :, 0].unsqueeze(2)
bboxes = bboxes.numpy()

# generate boxes
def compute_area(box, i_x, i_y, split_size):
        #compute the area between box and [(i_x,i_y),(i_x+1,i_y+1)]
        p1_x, p1_y, p2_x, p2_y = box
        one_wh = 1.0/split_size
        p3_x, p3_y, p4_x, p4_y = i_x * one_wh, i_y * one_wh, (i_x+1) * one_wh, (i_y+1) * one_wh
        if p1_x > p4_x or p2_x < p3_x or p1_y > p4_y or p2_y < p3_y:
            return 0.0
        len = min(p2_x,p4_x) - max(p1_x,p3_x)
        wid = min(p2_y, p4_y) - max(p1_y, p3_y)
        if len < 0 or wid < 0:
            return 0.0
        return len * wid
    
split_size = 12

boxes_whole = []
for i_bs in range(bboxes.shape[0]):
    boxes = []
    for i_r in range(bboxes.shape[1]):
        areas = []
        bbox = bboxes[i_bs][i_r]
        for i_x in range(split_size):
            for i_y in range(split_size):
                area = compute_area(bbox, i_x, i_y, split_size)
                areas.append(area)
        areas = torch.tensor(areas)
        value, index = torch.sort(areas, descending=True, dim=0)
        value = value[:15]
        value = value / value.sum()
        index = index[:15]
        box = torch.cat([index, value], dim=-1)
        boxes.append(box)
    boxes = torch.stack(boxes, dim=0)
    boxes_whole.append(boxes)
boxes_whole = torch.stack(boxes_whole, dim=0)
boxes_whole = boxes_whole.numpy()
print(boxes_whole.shape)
np.save('/home/ubuntu/Students/zhoutao/code/PFAN/data/coco_test/test_boxes_12.npy', boxes_whole)



    



