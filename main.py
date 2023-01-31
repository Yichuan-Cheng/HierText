from utils import get_args
from dataset import HierTextDataset
import torch

cfg=get_args()
dataset_train = HierTextDataset(cfg, dynamic_batchHW=True)
train_sampler = None
loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=cfg.DATASETS.BATCH_SIZE,
    shuffle=False,  
    collate_fn=dataset_train.collate_fn,
    num_workers=cfg.DATASETS.NUMWORKS,
    drop_last=True,
    pin_memory=True,
    sampler=train_sampler
    )
for i, batch in enumerate(loader_train):  
    print(batch['images'].shape, batch['instance_masks'].shape, batch['instance_classes'].shape,
        batch['segmentation_masks'].shape, batch['layout_labels'].shape, batch['num_instances'])
    break

dataset_val = HierTextDataset(cfg, is_training=False, dynamic_batchHW=True)
val_sampler = None
loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=cfg.DATASETS.BATCH_SIZE,
    shuffle=False,  
    collate_fn=dataset_val.collate_fn,
    num_workers=cfg.DATASETS.NUMWORKS,
    drop_last=True,
    pin_memory=True,
    sampler=val_sampler
    )
for i, batch in enumerate(loader_val):  
    print(batch['images'].shape, batch['instance_masks'].shape, batch['instance_classes'].shape,
        batch['segmentation_masks'].shape, batch['layout_labels'].shape, batch['num_instances'])
    break


# print(dataset_train)