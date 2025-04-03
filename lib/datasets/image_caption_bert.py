"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2

import logging

logger = logging.getLogger(__name__)


class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = os.path.join(data_path,data_name)
        loc_image = os.path.join(data_path,data_name)
        loc_boxes = os.path.join(data_path, data_name)
        loc_bbox = os.path.join(data_path, data_name)
        loc_tags = os.path.join(data_path, data_name)
        loc_sizes = os.path.join(data_path, data_name)

        # Captions
        self.captions = []
        self.tags = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        with open(osp.join(loc_tags, '%s_tags.txt' % data_split), 'r') as f:
            for line in f:
                self.tags.append(line.strip())
        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))
        self.boxes = np.load(os.path.join(loc_boxes, '%s_boxes.npy' % data_split))
        self.bbox = np.load(os.path.join(loc_bbox, '%s_ims_bbx.npy' % data_split))
        self.sizes = np.load(os.path.join(loc_sizes, '%s_ims_size.npy' % data_split), allow_pickle=True)

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        tag = self.tags[img_index]
        tag = process_tags(self.tokenizer, tag)
        # Convert caption (string) to word ids (with Size Augmentation at training time)
        target = process_caption(self.tokenizer, caption_tokens, self.train)
        image = self.images[img_index]
        bboxes = self.bbox[img_index]
        imsize = self.sizes[img_index]
        k = image.shape[0]
        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize['image_w']
            bbox[1] /= imsize['image_h']
            bbox[2] /= imsize['image_w']
            bbox[3] /= imsize['image_h']
            bboxes[i] = bbox
        box = torch.tensor(self.boxes[img_index])
        if self.train:  # Size augmentation for region feature
            if self.opt.schema == 'random':
                num_features = image.shape[0]
                rand_list = np.random.rand(num_features)
                image = image[np.where(rand_list > 0.20)]
                box = box[np.where(rand_list > 0.20)]
            elif self.opt.schema == 'tags':
                unique_tag = {}
                for i, v in enumerate(tag):
                    if v not in unique_tag:
                        unique_tag[i] = 1
                    else :
                        unique_tag[i] += 1
                image = image[[key for key in unique_tag.keys()]]
                box = box[[key for key in unique_tag.keys()]]
                tag = tag[[key for key in unique_tag.keys()]]
            elif self.opt.schema == 'center':
                num_features = image.shape[0]
                n_r = 26
                bb_size = (bboxes[:, 2:] - bboxes[:, :2])
                bb_centre = bboxes[:, :2] + 0.5 * bb_size
                bb_centre = torch.tensor(bb_centre)
                centre_coord = bb_centre.mean(dim=0, keepdim=True).repeat_interleave(num_features, 0)
                pseudo_coord = bb_centre - centre_coord
                rho = torch.sqrt(pseudo_coord[:, 0]**2 + pseudo_coord[:, 1]**2)
                rho_index = rho.sort(descending=False, dim=-1)[1]
                image = image[rho_index[:n_r]]
                box = box[rho_index[:n_r]]
        image = torch.Tensor(image)
        box = torch.tensor(box)
        return image, target, index, img_index, box

    def __len__(self):
        return self.length


def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target

def process_tags(tokenizer, tags):
    words = tags.strip().split(',')[:36]
    caption = list()
    caption.extend([tokenizer.convert_tokens_to_ids(token) for token in words])
    target = torch.tensor(caption, dtype=torch.long)
    return target


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, ids, img_ids, boxes = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Sort a data list by caption length
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)
        img_lengths = [len(image) for image in images]
        box_lengths = [len(box) for box in boxes]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        all_boxes = torch.zeros(len(boxes), max(box_lengths), boxes[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        for i, box in enumerate(boxes):
            end = box_lengths[i]
            all_boxes[i, :end] = box[:end]
        img_lengths = torch.Tensor(img_lengths)
        box_lengths = torch.Tensor(box_lengths)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return all_images, img_lengths, targets, lengths, all_boxes, ids
    else:  # raw input image
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths, ids


def get_loader(data_path, data_name, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.precomp_enc_type in ["basic","selfattention","transformer"]:
        dset = PrecompRegionDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(opt.precomp_enc_type))
    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, True, workers, train=opt.drop)
    val_loader = get_loader(data_path, data_name, 'dev', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader
