import copy

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from oneformer.data.tokenizer import SimpleTokenizer, Tokenize

REDUCED_CLASS_MAP = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 2,
    10: 3,
    11: 3,
    12: 4,
    13: 4,
    14: 5,
    15: 6,
    16: 7,
    17: 8,
    18: 8,
    19: 9,
    20: 10,
    21: 11,
    22: 10,
    23: 10,
    24: 12,
    25: 0,
    26: 13,
    27: 13,
}

__all__ = ["WaymoLidarSemanticDatasetMapper"]


class WaymoLidarSemanticDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        depth_on=True,
        depth_in_log_scale=True,
        *,
        meta,
        modalities,
        condition_text_entries,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        task_seq_len,
        max_seq_len,
        num_queries,
    ):
        self.is_train = is_train
        self.depth_on = depth_on
        self.depth_in_log_scale = depth_in_log_scale
        self.meta = meta
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.num_queries = num_queries
        self.main_modality = "CAMERA"
        self.modalities = modalities
        self.condition_text_entries = condition_text_entries
        self.class_names = self.meta.stuff_classes
        self.text_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seq_len)
        self.task_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=task_seq_len)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        if cfg.INPUT.INTERP.upper() == "NEAREST":
            interp = Image.NEAREST
        elif cfg.INPUT.INTERP.upper() == "BILINEAR":
            interp = Image.BILINEAR
        else:
            raise NotImplementedError

        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                    interp,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())
        else:
            augs = []

        dataset_names = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST_SEMANTIC
        meta = MetadataCatalog.get(dataset_names[0])
        modalities = [cfg.DATASETS.MODALITIES.MAIN_MODALITY.upper()]
        for modality in cfg.DATASETS.MODALITIES.ORDER:
            modality_key = modality.upper()
            if modality_key != modalities[0] and cfg.DATASETS.MODALITIES[modality_key].LOAD:
                modalities.append(modality_key)
        return {
            "is_train": is_train,
            "depth_on": cfg.MODEL.DEPTH_HEAD.ENABLED if is_train else cfg.MODEL.TEST.DEPTH_ON,
            "depth_in_log_scale": cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE,
            "meta": meta,
            "modalities": modalities,
            "condition_text_entries": cfg.MODEL.CONDITION_CLASSIFIER.CONDITION_TEXT_ENTRIES,
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES - cfg.MODEL.TEXT_ENCODER.N_CTX,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": meta.ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }

    def _get_texts(self, classes, num_class_obj):
        texts = ["an semantic photo"] * self.num_queries
        for class_id in list(np.array(classes)):
            cls_name = self.class_names[class_id]
            num_class_obj[cls_name] += 1

        cursor = 0
        for cls_name in self.class_names:
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if cursor >= len(texts):
                        break
                    texts[cursor] = f"a photo with a {cls_name}"
                    cursor += 1
        return texts

    def _apply_reduced_mapping(self, class_map):
        remapped = np.zeros_like(class_map, dtype=np.int64)
        for old_id, new_id in REDUCED_CLASS_MAP.items():
            remapped[class_map == old_id] = new_id
        return remapped

    def _load_sparse_lidar(self, lidar_path, height, width):
        lidar_npz = np.load(lidar_path)
        u = np.asarray(lidar_npz["u"], dtype=np.float32)
        v = np.asarray(lidar_npz["v"], dtype=np.float32)
        depth = np.asarray(lidar_npz["depth"], dtype=np.float32)

        depth_map = np.full((height, width), np.inf, dtype=np.float32)
        uu = np.clip(np.rint(u).astype(np.int64), 0, width - 1)
        vv = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
        depth[~np.isfinite(depth)] = np.inf
        np.minimum.at(depth_map, (vv, uu), depth)
        depth_map[np.isinf(depth_map)] = 0.0
        return depth_map

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        lidar_map = self._load_sparse_lidar(
            dataset_dict["lidar_file_name"], image.shape[0], image.shape[1]
        )
        lidar_image = np.repeat(lidar_map[..., None], 3, axis=2)

        sem_seg_npz = np.load(dataset_dict.pop("sem_seg_file_name"))
        sem_seg_gt = None
        for key in ("seg", "label", "labels", "mask", "gt", "semseg", "semantic", "semantics", "class_map"):
            if key in sem_seg_npz:
                sem_seg_gt = sem_seg_npz[key]
                break
        if sem_seg_gt is None:
            sem_seg_gt = sem_seg_npz[sem_seg_npz.files[0]]
        if sem_seg_gt.ndim == 3:
            sem_seg_gt = sem_seg_gt[..., 0]
        sem_seg_gt = self._apply_reduced_mapping(sem_seg_gt).astype("double")

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        if self.tfm_gens:
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg
            lidar_image = transforms.apply_segmentation(lidar_image)

        gt_depth = lidar_image[..., 0].astype(np.float32, copy=True)
        if self.depth_in_log_scale:
            valid = gt_depth > 0
            gt_depth[valid] = np.log(gt_depth[valid])
        gt_depth = np.expand_dims(gt_depth, axis=2)

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        lidar_image = torch.as_tensor(np.ascontiguousarray(lidar_image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        gt_depth = torch.as_tensor(np.ascontiguousarray(gt_depth.transpose(2, 0, 1)))

        if self.size_divisibility > 0 and self.is_train:
            height, width = image.shape[-2], image.shape[-1]
            pad_h = (self.size_divisibility - height % self.size_divisibility) % self.size_divisibility
            pad_w = (self.size_divisibility - width % self.size_divisibility) % self.size_divisibility
            if pad_h > 0 or pad_w > 0:
                padding_size = [0, pad_w, 0, pad_h]
                image = F.pad(image, padding_size, mode="reflect").contiguous()
                lidar_image = F.pad(lidar_image, padding_size, mode="constant").contiguous()
                gt_depth = F.pad(gt_depth, padding_size, mode="constant").contiguous()
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])
        dataset_dict["image"] = image
        dataset_dict["CAMERA"] = image
        dataset_dict["LIDAR"] = lidar_image
        for modality in self.modalities:
            if modality in {"CAMERA", "LIDAR"}:
                continue
            dataset_dict[modality] = torch.zeros_like(image)
        dataset_dict["modalities"] = list(self.modalities)
        dataset_dict["sem_seg"] = sem_seg_gt.long()
        dataset_dict["orig_shape"] = image_shape
        dataset_dict["task"] = "The task is semantic"
        condition_defaults = {
            "condition": "clear",
            "time_of_day": "daytime",
            "precipitation_text": "no precipitation",
            "ground_condition": "dry road",
            "sun_level": "normal sunlight",
            "text": "projected lidar scene",
        }
        dataset_dict["condition_text"] = [
            condition_defaults.get(entry, "unknown")
            for entry in self.condition_text_entries
        ]

        if self.depth_on:
            dataset_dict["gt_depth"] = gt_depth

        classes = np.unique(sem_seg_gt.numpy())
        classes = classes[classes != self.ignore_label]
        instances = Instances(image_shape)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        masks = [(sem_seg_gt.numpy() == class_id) for class_id in classes]
        if len(masks) == 0:
            instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
        else:
            instances.gt_masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(mask.copy())) for mask in masks])
            ).tensor

        num_class_obj = {name: 0 for name in self.class_names}
        dataset_dict["instances"] = instances
        dataset_dict["text"] = self._get_texts(instances.gt_classes, num_class_obj)
        return dataset_dict
