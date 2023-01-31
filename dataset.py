import os
import json
import torch

import numpy as np
import random
import math
import cv2
import torch.nn as nn
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import torchvision.transforms as transform

# from .aug_strategy import imgaug_mask
# from .aug_strategy import pipe_sequential_rotate
# from .aug_strategy import pipe_sequential_translate
# from .aug_strategy import pipe_sequential_scale
# from .aug_strategy import pipe_someof_flip
# from .aug_strategy import pipe_someof_blur
# from .aug_strategy import pipe_sometimes_mpshear
# from .aug_strategy import pipe_someone_contrast
def visualize(text_dict, level='word'):
    h, w, n_mask = text_dict['gt_masks'].shape
    if not n_mask:
        return np.zeros((h, w, 3), np.float32)
    palette = np.random.uniform(0.0, 1., (n_mask, 3))
    colored = np.reshape(np.matmul(
            np.reshape(text_dict['gt_masks'], (-1, n_mask)), palette), (h, w, 3))
    dont_care_mask = (np.reshape(np.matmul(
            np.reshape(text_dict['gt_masks'], (-1, n_mask)),
            np.reshape(1.- text_dict['gt_weights'], (-1, 1))), (h, w, 1)) > 0).astype(np.float32)

    filtered = np.clip(dont_care_mask * 1. + (1. - dont_care_mask) * colored, 0., 1.)

    plt.figure(figsize=(15, 15))
    plt.imshow(filtered)
    plt.savefig(level + '_mask.png')

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)
        
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, opt, **kwargs):
        # parse options        
        self.imgSizes = opt.INPUT.CROP.SIZE
        self.imgMaxSize = 1024
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 2**5 # resnet 总共下采样5次

        # parse the input list
        # self.parse_input_list(odgt, **kwargs)
        self.pixel_mean = np.array(opt.DATASETS.PIXEL_MEAN)
        self.pixel_std = np.array(opt.DATASETS.PIXEL_STD)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        # print(img.shape)
        img = img / 255.   
        img = (img - torch.from_numpy(self.pixel_mean).unsqueeze(1).unsqueeze(2)) / torch.from_numpy(self.pixel_std).unsqueeze(1).unsqueeze(2)
        # img = img.transpose((2, 0, 1))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long()
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p         

    
    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio

    def resize_padding(self, img, outsize, Interpolation=Image.BILINEAR):
        w, h = img.size
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)
        ow, oh = round(w * ratio), round(h * ratio)
        img = img.resize((ow, oh), Interpolation)
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
        return img

    def padding_torch(self, img, outsize, Interpolation=transform.InterpolationMode.NEAREST):
        c, h, w = img.shape
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)
        ow, oh = round(w * ratio), round(h * ratio)
        resize = transform.Resize((oh, ow), interpolation=Interpolation)
        img = resize(img)
        # print(target_h, target_w, img.shape)
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        zeroPad = nn.ZeroPad2d((left, right, top, bottom))

        img = zeroPad(img)
        # print(img.shape)

        return img

    
    def is_cross_text(self, start_loc, length, vertices):
        '''check if the crop image crosses text regions
        Input:
            start_loc: left-top position
            length   : length of crop image
            vertices : vertices of text regions <numpy.ndarray, (n,8)>
        Output:
            True if crop image crosses text region
        '''
        if len(vertices) == 0:
            return False
        start_w, start_h = start_loc
        a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                    start_w, start_h + length]).reshape((4, 2))
        p1 = Polygon(a).convex_hull
        for vertice in vertices:
            # print(vertice.shape)
            p2 = Polygon(vertice).convex_hull
            # print(p1,p2)
            # print(p2.area)
            inter = p1.intersection(p2).area
            if 0.01 <= inter / (p2.area + 1e-6) <= 0.99:
                return True
        return False
    


class HierTextDataset(BaseDataset):
    def __init__(self, opt, dynamic_batchHW = False, is_training = True, **kwargs):
        super(HierTextDataset, self).__init__(opt, **kwargs)
        
        if is_training:
            SUBSET = 'train'
        else:
            SUBSET = 'validation'

        self.IMAGE_SUBSET_DIR_PATH = os.path.join(opt.DATASETS.IMAGE_DIR_PATH, SUBSET)
        ANN_PATH = os.path.join(opt.DATASETS.ANN_DIR_PATH, f'{SUBSET}.jsonl')
        self.training = is_training
        self.annotations = json.load(open(ANN_PATH, 'r'))['annotations']
        self.detect_level = 'line'
        # self.img_list =  [ann['image_id'] for ann in self.annotations]
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.MODEL.SEM_SEG_HEAD.COMMON_STRIDE # 网络输出相对于输入缩小的倍数
        self.dynamic_batchHW = dynamic_batchHW  # 是否动态调整batchHW, cswin_transformer需要使用固定image size
        self.max_num_instances = opt.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES - 1
        # self.visualize = ADEVisualize()

        # self.aug_pipe = self.get_data_aug_pipe()


    def get_batch_size(self, batch_records, dynamic_HW = False):
        batch_width, batch_height = self.imgMaxSize, self.imgMaxSize

        if dynamic_HW:            
            if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
                this_short_size = np.random.choice(self.imgSizes)
            else:
                this_short_size = self.imgSizes

            batch_widths = np.zeros(len(batch_records), np.int32)
            batch_heights = np.zeros(len(batch_records), np.int32)
            for i, item in enumerate(batch_records):
                img_height, img_width = item['image'].shape[0], item['image'].shape[1]
                this_scale = min(
                    this_short_size / min(img_height, img_width), \
                    self.imgMaxSize / max(img_height, img_width))
                batch_widths[i] = img_width * this_scale
                batch_heights[i] = img_height * this_scale
            
            batch_width = np.max(batch_widths)
            batch_height = np.max(batch_heights)
            
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        return batch_width, batch_height
    
    @staticmethod
    def draw_mask(vertices, w, h):
        mask = np.zeros((h, w, 3), dtype=np.float32)
        mask = cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), [1.] * 3)[:, :, 0]
        return mask

    @staticmethod
    def stack_word_masks(mask, word_mask):
        return (mask + word_mask > 0.).astype(np.float32)


    def load_img(self,image_id):
        # return Image.open('/home/hdong/Mask2Former-Simplify/output/0006289e4f292bcd.jpg').convert('RGB')
        return Image.open(os.path.join(self.IMAGE_SUBSET_DIR_PATH, f'{image_id}.jpg')).convert('RGB')


    def random_crop(self, img, vertices, length=1024):
        '''crop img patches to obtain batch and augment
        Input:
            img         : PIL Image
            vertices    : vertices of text regions <numpy.ndarray, (n,8)>
            labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
            length      : length of cropped image region
        Output:
            region      : cropped image region
            new_vertices: new vertices in cropped region
        '''
        h, w = img.height, img.width
        # confirm the shortest side of image >= length
        if h >= w and w < length:
            img = img.resize((length, int(h * length / w)), Image.BILINEAR)
        elif h < w and h < length:
            img = img.resize((int(w * length / h), length), Image.BILINEAR)
        ratio_w = img.width / w
        ratio_h = img.height / h
        assert(ratio_w >= 1 and ratio_h >= 1)

        # print(ratio_w, ratio_h)
        for idx in range(len(vertices)):
            vertices[idx][:,0] = ratio_w * vertices[idx][:,0]
            vertices[idx][:,1] = ratio_h * vertices[idx][:,1]

        # find random position
        remain_h = img.height - length
        remain_w = img.width - length
        flag = True
        cnt = 0
        while flag and cnt < 100:
            cnt += 1
            start_w = int(np.random.rand() * remain_w)
            start_h = int(np.random.rand() * remain_h)
            flag = self.is_cross_text([start_w, start_h], length, vertices)
        box = (start_w, start_h, start_w + length, start_h + length)

        for idx in range(len(vertices)):
            vertices[idx][:,0] -= start_w
            vertices[idx][:,1] -= start_h
            vertices[idx] = np.clip(vertices[idx], 0, length-1)
        img = img.crop(box)
        # print(new_vertices.max(), new_vertices.min())
        crop_info = (ratio_w, ratio_h, start_w, start_h)
        return img, crop_info, vertices
    

    def parse_annotation_dict(self, ann):
        image_id = ann['image_id']
        # print(f'Processing image {image_id}')
        image = self.load_img(image_id)

        gt_word_masks = []
        gt_word_weights = []
        gt_word_texts = []
        gt_word_para_ids = []

        gt_line_masks = []
        gt_line_weights = []
        gt_line_texts = []
        gt_line_para_ids = []

        gt_paragraph_masks = []
        gt_paragraph_weights = []
        # dont_care_masks = []
        not_croped = [1] * 1000

        if self.training:
            all_vertices=[]
            for paragraph in ann['paragraphs']:
                for line in paragraph['lines']:
                    for word in line[ 'words']:
                        # print(word['vertices'], np.array(word['vertices']).shape)
                        all_vertices.append(np.array(word['vertices'], dtype=np.int32))
            # print(all_vertices)
            image, crop_info, vertices = self.random_crop(image, all_vertices, length=1024)
            not_croped = []
            for v in vertices:
                not_croped.append(Polygon(v).convex_hull.area > 10)
        else:
            crop_info = (1,1,0,0)
            # print(not_croped)
            # print(crop_box)
            # print(np.array(image).shape)
            # print(vertices)

        h, w = image.height, image.width
        dont_care_regions = np.zeros((h,w), dtype=np.float32)
        # print(h,w)
        para_id = 1
        word_idx = -1
        for paragraph in ann['paragraphs']:
            gt_paragraph_mask = np.zeros((h, w), dtype=np.float32)
            for line in paragraph['lines']:
                gt_line_mask = np.zeros((h, w), dtype=np.float32)
                for word in line['words']:
                    word_idx += 1
                    if not not_croped[word_idx] and self.training:
                        continue
                    gt_word_weights.append(1.0 if word['legible'] else 0.0)
                    gt_word_texts.append(word['text'])

                    if self.training:
                        new_vertices = vertices[word_idx]
                    else:
                        new_vertices = np.array(word['vertices'])
                    
                    gt_word_para_ids.append(para_id)
                    gt_word_mask = self.draw_mask(new_vertices, w, h)
                    gt_word_masks.append(gt_word_mask)
                    gt_line_mask = self.stack_word_masks(gt_line_mask, gt_word_mask)
                    gt_paragraph_mask = self.stack_word_masks(gt_paragraph_mask,
                                                    gt_word_mask)
                if gt_line_mask.sum() == 0:
                    continue
                gt_line_masks.append(gt_line_mask)
                gt_line_para_ids.append(para_id)
                gt_line_weights.append(1.0 if line['legible'] else 0.0)
                gt_line_texts.append(line['text'])
            if not paragraph['legible']:
                para_poly = np.array(paragraph['vertices'])
                para_poly[:,0] = para_poly[:,0] * crop_info[0] - crop_info[2]
                para_poly[:,1] = para_poly[:,1] * crop_info[1] - crop_info[3]
                if Polygon(para_poly).convex_hull.area > 0:
                    dont_care_regions += self.draw_mask(np.array(para_poly), w, h)              #TODO: dont care regions: mask? or just dont care
            if gt_paragraph_mask.sum() == 0:
                continue
            gt_paragraph_masks.append(gt_paragraph_mask)
            gt_paragraph_weights.append(1.0 if paragraph['legible'] else 0.0)
            para_id += 1

        # image = self._erase(image, torch.from_numpy(dont_care_regions))       # replace unlegible region with noises

        num_gt_words = len(gt_word_masks)
        word_dict = {
            'gt_weights': (np.array(gt_word_weights) if num_gt_words else np.zeros(
                (0,), np.float32)),
            'gt_para_ids': (np.array(gt_word_para_ids) if num_gt_words else np.zeros(
                (0,), np.float32)),
            'gt_masks': (np.stack(gt_word_masks, -1) if num_gt_words else np.zeros(
                ((h + 1) // 2, (w + 1) // 2, 0), np.float32)),
            'gt_texts': (np.array(gt_word_texts) if num_gt_words else np.zeros(
                (0,), str)),
        }

        num_gt_lines = len(gt_line_masks)
        line_dict = {
            'gt_weights': (np.array(gt_line_weights) if num_gt_lines else np.zeros(
                (0,), np.float32)),
            'gt_para_ids': (np.array(gt_line_para_ids) if num_gt_words else np.zeros(
                (0,), np.float32)),
            'gt_masks': (np.stack(gt_line_masks, -1) if num_gt_lines else np.zeros(
                ((h + 1) // 2, (w + 1) // 2, 0), np.float32)),
            'gt_texts': (np.array(gt_line_texts) if num_gt_lines else np.zeros(
                (0,), str)),
        }

        # print(gt_line_para_ids)

        num_gt_paragraphs = len(gt_paragraph_masks)
        paragraph_dict = {
            'gt_weights':
                (np.array(gt_paragraph_weights) if num_gt_paragraphs else np.zeros(
                    (0,), np.float32)),
            'gt_masks':
                (np.stack(gt_paragraph_masks, -1) if num_gt_paragraphs else np.zeros(
                    ((h + 1) // 2, (w + 1) // 2, 0), np.float32)),
        }

        # print(image.shape)
        # image.save('./cropped.jpg')
        # cv2.imwrite('./cropped.jpg', image)
        # visualize(word_dict, 'word')
        # visualize(line_dict, 'line')
        # visualize(paragraph_dict, 'para')
        return np.array(image), word_dict, line_dict, paragraph_dict

    def _erase(self, image, erase_masks):
        '''
        Replace ignored region with random noises
        '''
        image = torch.from_numpy(np.array(image))
        h, w, c = image.shape
        if len(erase_masks.shape) == 2:
            return torch.where(erase_masks.unsqueeze(-1).repeat(1,1,3) > 0., torch.ones((h,w,3)).uniform_(0., 255.), image).numpy()

        for mask in erase_masks:
            image = torch.where(mask.unsqueeze(-1).repeat(1,1,3) > 0., torch.ones((h,w,3)).uniform_(0., 255.), image)

        return image.numpy()

    def __getitem__(self, index):        
        
        labels = dict()
        image, word_dict, line_dict, paragraph_dict = self.parse_annotation_dict(self.annotations[index])
        # print(line_dict['gt_masks'].shape)

        h, w, c = image.shape
        segmentation_mask = np.zeros((h, w), dtype=np.float32)      #A (h, w) float32 mask for textness score. 1 for word,0 for bkg.
        num_instances = 0
        instances_mask_ids = np.zeros((h, w), dtype=np.float32)     #A (h, w) int32 mask for entity IDs. The value of each pixel is the id of the entity it belongs to. A value of `0` means the bkg mask.
        instance_masks = []
        para_ids = []
        if self.detect_level == 'line':
            for idx, weight in enumerate(line_dict['gt_weights']):
                if weight == .0:            # ignore non-legible masks
                    image = self._erase(image, torch.from_numpy(line_dict['gt_masks'][:,:,idx]))
                    continue
                num_instances += 1
                segmentation_mask += line_dict['gt_masks'][:,:,idx]
                instances_mask_ids += line_dict['gt_masks'][:,:,idx] * num_instances
                instance_masks.append(line_dict['gt_masks'][:,:,idx])
                para_ids.append(line_dict['gt_para_ids'][idx])
        
        if num_instances == 0:
            labels['num_instances'] = 0
            return labels
        instance_masks = torch.stack([x for x in (torch.from_numpy(np.array(instance_masks)))], dim=0)       # list to tensor
        para_ids = torch.from_numpy(np.array(para_ids))
        instance_classes = torch.ones(num_instances)
        segmentation_mask = torch.from_numpy(segmentation_mask)

        pad_num = np.maximum(self.max_num_instances - num_instances, 0)
        if pad_num > 0:
            kept_instance_masks = torch.cat([instance_masks, torch.zeros(pad_num, h,w)], dim=0)
            instance_classes = torch.cat([instance_classes, torch.zeros(pad_num)], dim=0)
            para_ids = torch.cat([torch.from_numpy(np.array(para_ids)), torch.zeros(pad_num)], dim=0)
            # print(kept_instance_masks.shape, instance_classes.shape, para_ids)
            # print(instance_classes)
        else:
            sample_idx = np.random.choice(np.linspace(0, num_instances-1, num_instances, dtype=np.int32), self.max_num_instances)
            erase_idx = np.setdiff1d(np.linspace(0, num_instances-1, num_instances, dtype=np.int32), sample_idx)
            # print(erase_idx, num_instances)

            kept_instance_masks = instance_masks[sample_idx]
            instance_classes = instance_classes[sample_idx]
            para_ids = para_ids[sample_idx]
            segmentation_mask *= torch.sum(kept_instance_masks, dim=0)

            erase_masks = instance_masks[erase_idx]
            image = self._erase(image, erase_masks)             

        kept_instance_and_bkg_masks = torch.cat([torch.logical_not(segmentation_mask).unsqueeze(0), kept_instance_masks], dim=0)           # N * H * W
        instance_and_bkg_classes = torch.cat([torch.ones(1)*2, instance_classes], dim=0)                                                   # N * 1
        para_ids = torch.cat([torch.zeros(1), para_ids], dim=0)                                                                            # 0 for bkg and non-object
        

        labels['image'] = image
        labels['segmentation_mask'] = segmentation_mask
        labels['instance_masks'] = kept_instance_and_bkg_masks
        labels['instance_classes'] = instance_and_bkg_classes                       #[2,1,1,1,1,..,1,0,0,...,0] 2 for bkg, 1 for words/lines , 0 for none-obj

        labels['layout_labels'] = para_ids

        labels['num_instances'] = np.minimum(num_instances, self.max_num_instances)

        # print(image.shape, segmentation_mask.shape, kept_instance_and_bkg_masks.shape, instance_and_bkg_classes.shape)

        return labels


    # Customization of batchify process, mainly for dynamic scale training
    def collate_fn(self, batch):
        
        out = {}
        images = []
        seg_masks = []
        ins_masks = []
        ins_classes = []
        layout_labels = []
        num_instances = []

        if not self.training:
            batch_width, batch_height = self.get_batch_size(batch)
            for item in batch:

                img = self.padding_torch(torch.from_numpy(item['image']).permute(2,0,1), (batch_width, batch_height),transform.InterpolationMode.BILINEAR)
                img = self.img_transform(img)                                                           # Normalize
                segm = self.padding_torch(item['segmentation_mask'].unsqueeze(0), (batch_width, batch_height))
                insm = self.padding_torch(item['instance_masks'], (batch_width, batch_height))

                images.append(img)
                seg_masks.append(segm[:,0])
                ins_masks.append(insm)
                ins_classes.append(item['instance_classes'].long())
                layout_labels.append(item['layout_labels'].long())
                num_instances.append(item['num_instances'])
            
            out['images'] = torch.stack(images)
            out['segmentation_masks'] = torch.stack(seg_masks)
            out['instance_masks'] = torch.stack(ins_masks)
            out['instance_classes'] = torch.stack(ins_classes)
            out['layout_labels'] = torch.stack(layout_labels)
            out['num_instances'] = num_instances
 
        else:
            for item in batch:
                if item['num_instances'] == 0:
                    continue                            # ignore images without test instances
                
                # img = Image.fromarray(np.uint8(item['image']))
                img = self.img_transform(torch.from_numpy(item['image']).permute(2,0,1))

                images.append(img)
                seg_masks.append(item['segmentation_mask'])
                ins_masks.append(item['instance_masks'])
                ins_classes.append(item['instance_classes'].long())
                layout_labels.append(item['layout_labels'].long())
                num_instances.append(item['num_instances'])
            
            dynamic_HW = np.random.choice([224, 320, 480, 512])
            # from torchvision.transforms import Resize
            # torch_resize = Resize([dynamic_HW, dynamic_HW])

            out['images'] = torch.stack(images)
            out['segmentation_masks'] = torch.stack(seg_masks)
            out['instance_masks'] = torch.stack(ins_masks)
            out['instance_classes'] = torch.stack(ins_classes)
            out['layout_labels'] = torch.stack(layout_labels)
            out['num_instances'] = num_instances

        return out        

    def __len__(self):
        return len(self.annotations)



if __name__ == '__main__':

    import argparse
    from fvcore.common.config import CfgNode
    import collections
    import os.path as osp
    import sys
    from argparse import ArgumentParser
    from importlib import import_module

    from addict import Dict


    class ConfigDict(Dict):

        def __missing__(self, name):
            raise KeyError(name)

        def __getattr__(self, name):
            try:
                value = super(ConfigDict, self).__getattr__(name)
            except KeyError:
                ex = AttributeError("'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name))
            except Exception as e:
                ex = e
            else:
                return value
            raise ex


    def add_args(parser, cfg, prefix=''):
        for k, v in cfg.items():
            if isinstance(v, str):
                parser.add_argument('--' + prefix + k)
            elif isinstance(v, int):
                parser.add_argument('--' + prefix + k, type=int)
            elif isinstance(v, float):
                parser.add_argument('--' + prefix + k, type=float)
            elif isinstance(v, bool):
                parser.add_argument('--' + prefix + k, action='store_true')
            elif isinstance(v, dict):
                add_args(parser, v, k + '.')
            elif isinstance(v, collections.Iterable):
                parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
            else:
                print('connot parse key {} of type {}'.format(prefix + k, type(v)))
        return parser


    class Config(object):
        """A facility for config and config files.
        It supports common file formats as configs: python/json/yaml. The interface
        is the same as a dict object and also allows access config values as
        attributes.
        Example:
            >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
            >>> cfg.a
            1
            >>> cfg.b
            {'b1': [0, 1]}
            >>> cfg.b.b1
            [0, 1]
            >>> cfg = Config.fromfile('tests/data/config/a.py')
            >>> cfg.filename
            "/home/kchen/projects/mmcv/tests/data/config/a.py"
            >>> cfg.item4
            'test'
            >>> cfg
            "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
            "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
        """

        @staticmethod
        def fromfile(filename):
            filename = osp.abspath(osp.expanduser(filename))
            if filename.endswith('.py'):
                module_name = osp.basename(filename)[:-3]
                if '.' in module_name:
                    raise ValueError('Dots are not allowed in config file path.')
                config_dir = osp.dirname(filename)
                sys.path.insert(0, config_dir)
                mod = import_module(module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
            elif filename.endswith(('.yml', '.yaml', '.json')):
                import mmcv
                cfg_dict = mmcv.load(filename)
            else:
                raise IOError('Only py/yml/yaml/json type are supported now!')
            return Config(cfg_dict, filename=filename)

        @staticmethod
        def auto_argparser(description=None):
            """Generate argparser from config file automatically (experimental)
            """
            partial_parser = ArgumentParser(description=description)
            partial_parser.add_argument('config', help='config file path')
            cfg_file = partial_parser.parse_known_args()[0].config
            cfg = Config.fromfile(cfg_file)
            parser = ArgumentParser(description=description)
            parser.add_argument('config', help='config file path')
            add_args(parser, cfg)
            return parser, cfg

        def __init__(self, cfg_dict=None, filename=None):
            if cfg_dict is None:
                cfg_dict = dict()
            elif not isinstance(cfg_dict, dict):
                raise TypeError('cfg_dict must be a dict, but got {}'.format(
                    type(cfg_dict)))

            super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
            super(Config, self).__setattr__('_filename', filename)
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    super(Config, self).__setattr__('_text', f.read())
            else:
                super(Config, self).__setattr__('_text', '')

        @property
        def filename(self):
            return self._filename

        @property
        def text(self):
            return self._text

        def __repr__(self):
            return 'Config (path: {}): {}'.format(self.filename,
                                                self._cfg_dict.__repr__())

        def __len__(self):
            return len(self._cfg_dict)

        def __getattr__(self, name):
            return getattr(self._cfg_dict, name)

        def __getitem__(self, name):
            return self._cfg_dict.__getitem__(name)

        def __setattr__(self, name, value):
            if isinstance(value, dict):
                value = ConfigDict(value)
            self._cfg_dict.__setattr__(name, value)

        def __setitem__(self, name, value):
            if isinstance(value, dict):
                value = ConfigDict(value)
            self._cfg_dict.__setitem__(name, value)

        def __iter__(self):
            return iter(self._cfg_dict)
  
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='../configs/maskformer_ake150.yaml')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument("--ngpus", default=1, type=int)

        args = parser.parse_args()
        cfg_ake150 = Config.fromfile(args.config)

        cfg_base = CfgNode.load_yaml_with_base(args.config, allow_unsafe=True)    
        cfg_base.update(cfg_ake150.__dict__.items())

        cfg = cfg_base
        for k, v in args.__dict__.items():
            cfg[k] = v

        cfg = Config(cfg)
        # print(cfg)

        cfg.ngpus = 1
        # cfg.ngpus = torch.cuda.device_count()
        # if torch.cuda.device_count() > 1:
        #     cfg.local_rank = torch.distributed.get_rank()
        #     torch.cuda.set_device(cfg.local_rank)
        return cfg

    cfg = get_args()
    dataset_train = HierTextDataset(cfg, dynamic_batchHW=True, is_training=True)
    train_sampler = None
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,  
        collate_fn=dataset_train.collate_fn,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    dataset_val = HierTextDataset(cfg, dynamic_batchHW=True, is_training=False)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,  
        collate_fn=dataset_val.collate_fn,
        num_workers=1,)

    for i, batch in enumerate(loader_train):  
        print(batch['images'].shape, batch['instance_masks'].shape, batch['instance_classes'].shape,
            batch['segmentation_masks'].shape, batch['layout_labels'].shape, batch['num_instances'])
        break

    for i, batch in enumerate(loader_val):  
        print(batch['images'].shape, batch['instance_masks'].shape, batch['instance_classes'].shape,
            batch['segmentation_masks'].shape, batch['layout_labels'].shape, batch['num_instances'])
        break