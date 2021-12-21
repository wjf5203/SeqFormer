# ------------------------------------------------------------------------
# SeqFormer data loader
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------


"""
 augment coco image to generate a n-frame pseudo video
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_clip as T
from datasets.image_to_seq_augmenter import ImageToSeqAugmenter
import numpy as np
from PIL import Image
from util.box_ops import masks_to_boxes
import random
from util import box_ops


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.num_frames = 5
        self.augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-20, 20), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                             motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                             translate_range=(-0.1, 0.1))


    def apply_random_sequence_shuffle(self, images, instance_masks):
        perm = list(range(self.num_frames))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        return images, instance_masks
    
    def __getitem__(self, idx):

        instance_check = False
        while not instance_check:   

            img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)         
            seq_images, seq_instance_masks = [img], [target['masks'].numpy()]
            numpy_masks = target['masks'].numpy()
    
            numinst = len(numpy_masks)
            for t in range(self.num_frames-1):
                im_trafo, instance_masks_trafo = self.augmenter(np.asarray(img), numpy_masks)
                im_trafo = Image.fromarray(np.uint8(im_trafo))
                seq_images.append(im_trafo)
                seq_instance_masks.append( np.stack(instance_masks_trafo, axis=0))
            seq_images, seq_instance_masks = self.apply_random_sequence_shuffle(seq_images, seq_instance_masks)
            output_inst_masks = []
            for inst_i  in range(numinst):
                inst_i_mask = []
                for f_i in range(self.num_frames):
                    inst_i_mask.append(seq_instance_masks[f_i][inst_i])
                output_inst_masks.append( np.stack(inst_i_mask, axis=0) )
            
            output_inst_masks = torch.from_numpy( np.stack(output_inst_masks, axis=0) )         
            target['masks'] = output_inst_masks.flatten(0,1)
            target['boxes'] = masks_to_boxes(target['masks'])

            if self._transforms is not None:
                img, target = self._transforms(seq_images, target, self.num_frames)
            if len(target['labels']) > 0 and len(target['labels']) <= 25: 
                instance_check=True
            else:
                idx = random.randint(0,self.__len__()-1) # None instance or too much instances
              
        for inst_id in range(len(target['boxes'])):
            if target['masks'][inst_id].max()<1:
                target['boxes'][inst_id] =  torch.zeros(4).to(target['boxes'][inst_id])
       
        target['boxes']=target['boxes'].clamp(1e-6)
        return torch.cat(img,dim=0), target   


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=768),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=768),
                    T.Check(),
                ])
            ),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            T.RandomResize([360], max_size=640),
            normalize,
        ])
        
    raise ValueError(f'unknown {image_set}')



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks
        #ytvis19
        self.category_map = {1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18, 21:15, 22:32, 23:20, 24:30, 25:22, 36:33, 41:5, 42:27, 43:40, 74:24}
        # ytvis21  
        # self.category_map = {1:26, 2:23, 3:5, 4:23, 5:1, 7:36, 8:37, 9:4, 16:3, 17:6, 18:9, 19:19, 21:7, 22:12, 23:2, 24:40, 25:18, 36:31, 41:29, 42:33, 43:34, 74:24}



    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # classes = [obj["category_id"] for obj in anno]  # map coco category id to YTVIS-2019 category id
        classes = [self.category_map[obj["category_id"]] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

    
def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    dataset_type = args.dataset_type
    if args.dataset_file == 'coco' or args.dataset_file == 'Seq_coco' or args.dataset_file == 'jointcoco':
        PATHS = {
            # "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "train": (root / "train2017", './datasets/coco_keepfor_ytvis19.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        }
    if args.dataset_file == 'YoutubeVIS':
        PATHS = {
            "train": (root / f'{dataset_type}' / "train" / "JPEGImages", root / "annotations" / f'{dataset_type}' / f'{mode}_train_sub_coco.json'),
            "val": (root / f'{dataset_type}' / "valid" / "JPEGImages", root / "annotations" / f'{dataset_type}' / f'{mode}_val_sub_coco.json'),
        }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset


