from model import UnifiedDetector
from criterion import Criterion

from statistics import mean
import torch
import numpy as np
import os
import time
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch import distributed as dist
from collections import OrderedDict
import sys
import itertools
from tqdm import tqdm

from summary import create_summary
# from utils.solver import maybe_add_gradient_clipping
from misc import load_parallal_model

class Hier_trainer():
    def __init__(self, im_size=128,n_classes=1,pretrained_path=None,save_dir='output/save/',local_rank=0, ngpus=1,lr=0.001,log_dir='output/log/'):
        super().__init__()
        self.model=UnifiedDetector(im_size=im_size,n_classes=n_classes)
        self.lr=lr
        
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_model(pretrained_path)
            print("loaded pretrain mode:{}".format(pretrained_path))
            
        self.device = torch.device("cuda", local_rank)
        self.model = self.model.to(self.device)

        if ngpus > 1:
            # self.model = nn.parallel.DistributedDataParallel(self.model, broadcast_buffers=False, find_unused_parameters=True)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank) 

        # self._training_init(cfg)
        self.optimizer=self.build_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=10)
        self.criterion=Criterion()
        self.summary_writer = create_summary(0, log_dir=log_dir)
        self.n_epochs=500
        self.epoch=0
        self.save_dir=save_dir

    def build_optimizer(self):
        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = 0.01
            enable = True
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)
            return FullModelGradientClippingOptimizer if enable else optim
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(self.model.parameters(), self.lr)
        return optimizer

    def load_model(self, pretrain_weights):
        state_dict = torch.load(pretrain_weights, map_location='cuda:0')
        print('loaded pretrained weights form %s !' % pretrain_weights)

        ckpt_dict = state_dict['model']
        self.optimizer = state_dict['optimizer']
        self.scheduler = state_dict['scheduler']
        self.epoch = state_dict['epoch']
        self.model = load_parallal_model(self.model, ckpt_dict)

    def train(self, train_sampler, data_loader, eval_loder):
        max_score = 0.0
        for epoch in range(self.epoch,self.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.train_epoch(data_loader, epoch)
            evaluator_score = self.evaluate(eval_loder)
            self.scheduler.step(evaluator_score)
            self.summary_writer.add_scalar('val_score', evaluator_score, epoch)
            if evaluator_score > max_score:
                max_score = evaluator_score
                ckpt_path = os.path.join(self.save_folder, 'HierTextModel{0}_PQ{1}.pth'.format(epoch, max_score))
                save_state = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler,
                    'epoch': self.epoch
                    }
                torch.save(save_state, ckpt_path)
                print('weights {0} saved success!'.format(ckpt_path))
            self.epoch+=1
        self.summary_writer.close()

    def train_epoch(self,data_loader, epoch):
        self.model.train()
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2,3])
        self.criterion.train()
        load_t0 = time.time()
        for i, batch in enumerate(data_loader):                     
            images = batch['images'].to(device=self.device, non_blocking=True)
            mask_gt = batch['mask_gt'].to(device=self.device, non_blocking=True)
            classes_gt = batch['classes_gt'].to(device=self.device, non_blocking=True)
            affinity_matrix_gt = batch['affinity_matrix_gt'].to(device=self.device, non_blocking=True)
            gt_size = batch['gt_size'].to(device=self.device, non_blocking=True)
            pixel_features, mask_out, classes, semantic_mask, affinity_matrix=self.model(images)  
            loss=self.criterion(classes.sigmoid().squeeze(dim=-1),classes_gt,F.softmax(mask_out,dim=1),mask_gt,affinity_matrix,affinity_matrix_gt,semantic_mask.sigmoid(),mask_gt.sum(dim=1),pixel_features,gt_size)
            # print(loss)
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
            self.criterion.zero_grad()
            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            curent_lr = self.optimizer.param_groups[0]['lr']
            progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{loss:.6f} lr:{curent_lr:.2e} '
            print(progress, end=' ')
            sys.stdout.flush()                
        self.summary_writer.add_scalar('loss', loss.item(), epoch)
    
    def inference(self,eval_loder, t_m=0.4, t_A=0.5, t_c=0.5):
        masks,As,masks_gt,As_gt,cls_score,gt_sizes=[],[],[],[],[],[]
        for i, batch in enumerate(eval_loder):                     
            images = batch['images'].to(device=self.device, non_blocking=True)
            mask_gt = batch['mask_gt'].to(device=self.device, non_blocking=True)
            classes_gt = batch['classes_gt'].to(device=self.device, non_blocking=True)
            mask_gt = batch['mask_gt'].to(device=self.device, non_blocking=True)
            affinity_matrix_gt = batch['affinity_matrix_gt'].to(device=self.device, non_blocking=True)
            gt_size = batch['gt_size'].to(device=self.device, non_blocking=True)
            pixel_features, mask_out, classes, semantic_mask, affinity_matrix=self.model(images)  
            mask_out=F.softmax(mask_out,dim=1)
            mask=(mask_out>t_m).long()*F.one_hot(mask_out.argmax(dim=1),num_classes=100).permute(0, 3, 1, 2)
            affinity_matrix=(affinity_matrix>t_A).long()*affinity_matrix
            mask=(mask.sum(dim=[-1,-2],keepdim=True)>32).long()*mask
            classes=classes.squeeze(dim=-1)
            classes=(classes>t_c).long()*classes
            masks.append(mask)
            As.append(affinity_matrix)
            masks_gt.append(mask_gt)
            As_gt.append(affinity_matrix_gt)
            cls_score.append(classes)
            gt_sizes.append(gt_size)
        return torch.cat(masks),torch.cat(As),torch.cat(masks_gt),torch.cat(As_gt),torch.cat(cls_score),torch.cat(gt_sizes)
    
    def cal_PQ_stat(self,mask,mask_gt):
        mask=mask.unsqueeze(dim=1)
        mask_gt=mask_gt.unsqueeze(dim=0) #[N,K,H,W]
        matches=(mask*mask_gt).sum(dim=[-1,-2])/((mask_gt+mask)>0).long().sum(dim=[-1,-2])
        TP_cnt1=(matches>0.5).long().sum()
        FP_cnt1=len(mask)-TP_cnt1
        FN_cnt1=len(mask_gt)-TP_cnt1
        IoU_sum1=((matches>0.5).long()*matches).sum()
        return IoU_sum1,TP_cnt1,FP_cnt1,FN_cnt1

    def union_find(self,masks,As):
        rs=[{i}for i in range(len(masks))]
        for i in range(len(masks)):
            for j in range(i+1,len(masks)):
                if(As[i][j]==1):
                    rs[i].add(j)

        for i in range(len(rs)):
            tmp=rs[-i-1]
            for ele in tmp:
                tmp.union(rs[ele])
                del rs[ele]

        masks_rs=torch.cat([masks[list(i)].sum(dim=0) for i in rs])
        return masks_rs

    @torch.no_grad()
    def evaluate(self, eval_loder):
        masks,As,masks_gt,As_gt,cls_score,gt_sizes=self.inference(eval_loder)
        IoU_sum1,TP_cnt1,FP_cnt1,FN_cnt1=0,0,0,0
        IoU_sum2,TP_cnt2,FP_cnt2,FN_cnt2=0,0,0,0
        for (mask,A,mask_gt,A_gt,cls,gt_size) in zip(masks,As,masks_gt,As_gt,cls_score,gt_sizes):
            indexs=[]
            for i in range(mask):
                if(all(mask[i]==0) or cls[i]==0):
                    continue
                else:
                    indexs.append(i)
            mask=mask[indexs]
            A=A[indexs][:,indexs]
            mask_gt=mask_gt[gt_size]
            A_gt=A_gt[:gt_size][:,:gt_size]
            IoU_sum1_tmp,TP_cnt1_tmp,FP_cnt1_tmp,FN_cnt1_tmp=self.cal_PQ_stat(mask,mask_gt)
            IoU_sum1,TP_cnt1,FP_cnt1,FN_cnt1=IoU_sum1+IoU_sum1_tmp,TP_cnt1+TP_cnt1_tmp,FP_cnt1+FP_cnt1_tmp,FN_cnt1+FN_cnt1_tmp

            mask=self.union_find(masks,A)
            mask_gt=self.union_find(mask_gt,A_gt)
            IoU_sum2_tmp,TP_cnt2_tmp,FP_cnt2_tmp,FN_cnt2_tmp=self.cal_PQ_stat(mask,mask_gt)
            IoU_sum2,TP_cnt2,FP_cnt2,FN_cnt2=IoU_sum2+IoU_sum2_tmp,TP_cnt2+TP_cnt2_tmp,FP_cnt2+FP_cnt2_tmp,FN_cnt2+FN_cnt2_tmp
        PQ1=IoU_sum1/(TP_cnt1+0.5*FP_cnt1+0.5*FN_cnt1)
        PQ2=IoU_sum2/(TP_cnt2+0.5*FP_cnt2+0.5*FN_cnt2)
        return (PQ1+PQ2)/2





