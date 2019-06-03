import logging
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
import os
import cv2
import mmcv
import numpy as np
import torch.nn as nn
import pycocotools.mask as maskUtils

from mmdet.core import tensor2imgs, get_classes


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_track(self):
        return hasattr(self, 'track_head') and self.track_head is not None
    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)
    
    def gen_colormask(self, N=256):
        def bitget(byteval, idx):
          return ((byteval & (1 << idx)) != 0)

        dtype = 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
          r = g = b = 0
          c = i
          for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

          cmap[i] = np.array([r, g, b])

        self.color_mask = cmap[3:]
    
    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset='coco',
                    is_video=False,
                    save_vis=False,
                    save_path='vis',
                    score_thr=0.1):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]

        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)) or dataset is None:
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))
        # use colors to denote different object instances in videos
        # for YTVOS, only refresh color_mask at the first frame of each video
        if not hasattr(self, 'color_mask'):
          self.gen_colormask()
        if isinstance(bbox_result, dict) and len(bbox_result.keys()) == 0:
            return
        assert len(imgs) == 1, "only support mini-batch size 1"
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            #img_show = img[:h, :w, :]
            img_show = np.zeros((h,w,3)) 
            if not is_video:
              bboxes = np.vstack(bbox_result)
            else:
              bboxes = np.vstack([x['bbox'] for x in bbox_result.values()])
              obj_ids = list(bbox_result.keys())
            # draw segmentation masks
            if segm_result is not None:
                if not is_video:
                  segms = mmcv.concat_list(segm_result)
                else:
                  segms = list(segm_result.values())
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    mask = mask[:h, :w]
                    if not is_video:
                      color_id = i
                    else:
                      color_id = obj_ids[i]
                    img_show[mask] = self.color_mask[color_id,:]
            if save_vis:
              show = False 
              save_path = '{}/{}/{}.png'.format(save_path, img_meta['video_id'], img_meta['frame_id'])
            else:
              show = True
              save_path = None
            # draw bounding boxes
            if dataset == 'coco':
              labels = [
                  np.full(bbox.shape[0], i, dtype=np.int32)
                  for i, bbox in enumerate(bbox_result)
              ]
              labels = np.concatenate(labels)
            else:
              labels = [ x['label'] for x in bbox_result.values()]
              labels = np.array(labels)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes[:,:4],
                labels,
                class_names=class_names,
                show=show,
                text_color ='white',
                out_file=save_path)
