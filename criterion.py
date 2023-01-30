import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from einops import rearrange, repeat
import torch.nn.functional as F

# def dice_coeff(
#     m_p:torch.Tensor,
#     m_gt:torch.Tensor,
#     ):
#     """
#     Compute the dice coefficient
#     Args:
#         m_p: The predictions of masks [B,N,H,W]
#         m_gt: The ground truth of masks [B,K,H,W]
#     """
#     m_p=m_p.unsqueeze(2).flatten(3) #[B,N,1,HW]
#     m_gt=m_gt.unsqueeze(1).flatten(3) #[B,1,K,HW]
#     numerator = 2 * (m_p * m_gt).sum(-1) #[B,N,K]
#     denominator = m_p.square().sum(-1) + m_gt.square().sum(-1) #[B,N,K]
    
#     return numerator / denominator #[B,N,K]
class HungarianMatcher(nn.Module):
    def __init__(self):
        super(HungarianMatcher, self).__init__()
    
    def matching_sim(
        self,
        y_p:torch.Tensor,
        y_gt:torch.Tensor,
        m_p:torch.Tensor,
        m_gt:torch.Tensor,
        ):
        """
        Compute the simlarity between predictions and ground truth
        Args:
            y_p: The predictions of classification scores. [B,N]
            y_gt: The ground truth of classification scores. [B,K]
            m_p: The predictions of masks [B,N,H,W]
            m_gt: The ground truth of masks [B,K,H,W]
        """
        y_p=y_p.unsqueeze(2) #[B,N,1]
        y_gt=y_gt.unsqueeze(1) #[B,1,K]
        term1=y_p*y_gt+(1-y_p)*(1-y_gt) #[B,N,K]

        m_p=m_p.unsqueeze(2).flatten(3) #[B,N,1,HW]
        m_gt=m_gt.unsqueeze(1).flatten(3) #[B,1,K,HW]
        numerator = 2 * (m_p * m_gt).sum(-1) #[B,N,K]
        denominator = m_p.square().sum(-1) + m_gt.square().sum(-1) #[B,N,K]
        term2= numerator / denominator #[B,N,K]

        return term1*term2

    @torch.no_grad()
    def forward(
        self,
        y_p:torch.Tensor,
        y_gt:torch.Tensor,
        m_p:torch.Tensor,
        m_gt:torch.Tensor,
        gt_size:torch.Tensor,
        ):
        """
        Args:
            y_p: The predictions of classification scores. [B,N]
            y_gt: The gts of classification scores. [B,K]
            m_p: The predictions of masks [B,N,H,W]
            m_gt: The gts of masks [B,K,H,W]
            gt_size: The number of gts [B,]
        """
        device = y_p.device
        B, N = y_p.size()[:2]
        K = y_gt.size(-1)
        
        sim = self.matching_sim(y_p,y_gt,m_p,m_gt).cpu() #(B, N, K)
        indices = [linear_sum_assignment(s[:, :e], maximize=True) for s,e in zip(sim, gt_size)]

        input_pos_indices = []
        target_pos_indices = []
        input_neg_indices = []
        input_indices = np.arange(0, N)
        for i, (inp_idx, tgt_idx) in enumerate(indices):
            input_pos_indices.append(torch.as_tensor(inp_idx, dtype=torch.long, device=device))
            target_pos_indices.append(torch.as_tensor(tgt_idx, dtype=torch.long, device=device))
            input_neg_indices.append(torch.as_tensor(np.setdiff1d(input_indices, inp_idx), dtype=torch.long, device=device))

        batch_pos_idx = torch.cat([torch.full_like(pos, i) for i, pos in enumerate(input_pos_indices)])
        batch_neg_idx = torch.cat([torch.full_like(neg, i) for i, neg in enumerate(input_neg_indices)])
        input_pos_indices = torch.cat(input_pos_indices)
        target_pos_indices = torch.cat(target_pos_indices)
        input_neg_indices = torch.cat(input_neg_indices)

        inp_pos_indices = (batch_pos_idx, input_pos_indices)
        gt_pos_indices = (batch_pos_idx, target_pos_indices)
        inp_neg_indices = (batch_neg_idx, input_neg_indices)
        return inp_pos_indices, gt_pos_indices, inp_neg_indices

class Criterion(nn.Module):
    def __init__(self,alpha=0.5,alpha_L=0.5,lambda_1=3.0,lambda_2=1.0,lambda_3=1.0,lambda_4=1.0):
        super(Criterion, self).__init__()
        self.temp = 0.3
        self.eps = 1e-5
        self.matcher=HungarianMatcher()
        self.alpha=alpha
        self.alpha_L=alpha_L
        self.lambda_1=lambda_1
        self.lambda_2=lambda_2
        self.lambda_3=lambda_3
        self.lambda_4=lambda_4
        self.xentropy = nn.CrossEntropyLoss()

    def cal_det_loss(
        self,
        y_p_pos:torch.Tensor,
        y_gt_pos:torch.Tensor,
        y_p_neg:torch.Tensor,
        m_p_pos:torch.Tensor,
        m_gt_pos:torch.Tensor,
        m_p_neg:torch.Tensor=None
    ):
        """
        Args:
            y_p_pos: The positive sample of predictions of classification scores. [number_of_positive]
            y_gt_pos: The positive sample of gts of classification scores. [number_of_positive]
            y_p_neg: The negative sample of predictions of scores [number_of_negative]
            m_p_pos: The positive sample of predictions of masks [number_of_positive,H,W]
            m_gt_pos: The positive sample of gts of masks [number_of_positive,H,W]
            m_p_neg: The negative sample of predictions of masks [number_of_negative,H,W]
        """
        term1=((1-self.alpha)*(-torch.log(1-y_p_neg))).sum() # loss for negative sample

        y_p_pos_constant=y_p_pos.detach() #[number_of_positive]

        m_p_pos_tmp=m_p_pos.flatten(1) #[number_of_positive,HW]
        m_gt_pos_tmp=m_gt_pos.flatten(1) #[number_of_positive,HW]
        numerator = 2 * (m_p_pos_tmp * m_gt_pos_tmp).sum(-1) #[number_of_positive]
        denominator = m_p_pos_tmp.square().sum(-1) + m_gt_pos_tmp.square().sum(-1) #[number_of_positive]
        dice_coeff= numerator / denominator #[number_of_positive]
        dice_coeff_constant=dice_coeff.detach()

        term2=(self.alpha*y_gt_pos*(-y_p_pos_constant*dice_coeff-dice_coeff_constant*y_p_pos.log())).sum() # loss for positive sample
        # print(y_gt_pos,y_p_pos_constant,)

        return (term1+term2)/(y_p_pos.shape[0]+y_p_neg.shape[0])

    def cal_lay_loss(
        self,
        inp_pos_indices, 
        gt_pos_indices, 
        inp_neg_indices,
        A_p:torch.Tensor,
        A_gt:torch.Tensor,
        gt_size:torch.Tensor,
    ):
        """
        Args:
            inp_pos_indices: the index of positive sample of predictions
            gt_pos_indices: the index of positive sample of gts
            inp_neg_indices: the index of negative sample of predictions
            A_p: predictions of affinity matrix [B,N,N]
            A_gt: gts of affinity matrix [B,K,K]
            gt_size: The number of gts [B,]
        """
        B,N=A_p.shape[:2]
        K=A_gt.shape[1]
        # y_gt=torch.zeros([B,N,1])

        # omega_p=1/A_gt.sum(dim=[1,2]) #[B,]
        # omega_n=1/(gt_size*gt_size-A_gt.sum(dim=[1,2]))
        index_=0
        loss=0
        for i,gt_size_ in enumerate(gt_size):
            inp_pos_indices_tmp=inp_pos_indices[1][index_:index_+gt_size_]
            A_p_tmp=A_p[i,inp_pos_indices_tmp][:,inp_pos_indices_tmp] #[number of positive, number of positive]
            gt_pos_indices_tmp=gt_pos_indices[1][index_:index_+gt_size_]
            A_gt_tmp=A_gt[i,gt_pos_indices_tmp][:,gt_pos_indices_tmp] #[number of positive, number of positive]
            omega_p=1/(A_gt_tmp.sum()+self.eps)
            omega_n=1/((1-A_gt_tmp).sum()+self.eps)
            loss=loss+(self.alpha_L*omega_p*A_gt_tmp*(-(A_p_tmp+self.eps).log())).sum()+((1-self.alpha_L)*omega_n*(1-A_gt_tmp)*(-(1-A_p_tmp+self.eps).log())).sum()
        loss/=B

        return loss
    
    # def cal_semantic_segmentation_loss(
    #     self,
    #     input_mask,
    #     target_mask
    #     ):
    #     """
    #     Args:
    #         input_mask: [B, 1, H, W]
    #         target_mask: [B,H,W]
    #     """
    #     input_mask=input_mask.permute(0,2,3,1).repeat(1,1,1,2)
    #     input_mask=input_mask.flatten(end_dim=-2) #[BHW,2]
    #     input_mask[:,0]=1-input_mask[:,0]
    #     target_mask=target_mask.flatten(end_dim=-1) #[BHW]
    #     # print(input_mask,target_mask)
    #     return self.xentropy(input_mask, target_mask)
    #     # return 0

    def cal_semantic_segmentation_loss(
        self,
        input_mask,
        target_mask
        ):
        """
        Args:
            input_mask: [B, 1, H, W]
            target_mask: [B,H,W]
        """
        input_mask=input_mask.squeeze(dim=1) #[B,H,W]
        # print(input_mask,target_mask)
        # print(input_mask,target_mask)
        return (-target_mask*(input_mask.log())-(1-target_mask)*(1-input_mask).log()).mean()
        # return 0

    def cal_instance_dis_loss(self, mask_features, target_mask, target_sizes):
        """
        mask_features: (B, D, H/4, W/4) #g
        target_mask: (B, K, H, W) #m
        """

        #downsample input and target by 4 to get (B, H/4, W/4)
        # mask_features = mask_features[..., ::4, ::4]
        target_mask = target_mask[..., ::4, ::4]
        

        device = mask_features.device
        target_mask=target_mask.float()

        #eqn 16
        t = torch.einsum('bdhw,bkhw->bkd', mask_features, target_mask)
        t = F.normalize(t, dim=-1) #(B, K, D)

        #get batch and mask indices from target_sizes
        batch_indices = []
        mask_indices = []
        for bi, size in enumerate(target_sizes):
            mindices = torch.arange(0, size, dtype=torch.long, device=device)
            mask_indices.append(mindices)
            batch_indices.append(torch.full_like(mindices, bi))

        batch_indices = torch.cat(batch_indices, dim=0) #shape: (torch.prod(target_sizes), )
        mask_indices = torch.cat(mask_indices, dim=0)

        #create logits and apply temperature
        logits = torch.einsum('bdhw,bkd->bkhw', mask_features, t)
        logits = logits[batch_indices, mask_indices] #(torch.prod(target_sizes), H, W)
        logits /= self.temp

        #select target_masks
        m = target_mask[batch_indices, mask_indices] #(torch.prod(target_sizes), H, W)

        #flip so that there are HW examples for torch.prod(target_sizes) classes
        logits = rearrange(logits, 'k h w -> (h w) k')
        m = rearrange(m, 'k h w -> (h w) k')

        #eqn 17
        numerator = torch.logsumexp(m * logits, dim=-1) #(HW,)
        denominator = torch.logsumexp(logits, dim=-1) #(HW,)
        
        #log of quotient is difference of logs
        return (-numerator + denominator).mean()
    
    def forward(
        self,
        y_p:torch.Tensor,
        y_gt:torch.Tensor,
        m_p:torch.Tensor,
        m_gt:torch.Tensor,
        A_p:torch.Tensor,
        A_gt:torch.Tensor,
        semantic_mask:torch.Tensor,
        semantic_mask_gt:torch.Tensor,
        mask_features:torch.Tensor,
        gt_size:torch.Tensor,
        ):
        """
        Args:
            y_p: The predictions of classification scores. [B,N]
            y_gt: The gts of classification scores. [B,K]
            m_p: The predictions of masks [B,N,H,W]
            m_gt: The gts of masks [B,K,H,W]
            A_p: The predictions of affinity matrix [B,N,N]
            A_gt: The gts of affinity matrix [B,K,K]
            semantic_mask: [B,1,H,W]
            semantic_mask_gt: [B,H,W]
            mask_features: [B,H/4,W/4]
            gt_size: The number of gts [B,]
        """
        inp_pos_indices, gt_pos_indices, inp_neg_indices=self.matcher(y_p,y_gt,m_p,m_gt,gt_size)
        y_p_pos,y_gt_pos,y_p_neg,m_p_pos,m_gt_pos,m_p_neg=y_p[inp_pos_indices],y_gt[gt_pos_indices],y_p[inp_neg_indices],m_p[inp_pos_indices],m_gt[gt_pos_indices],m_p[inp_neg_indices]
        det_loss=self.cal_det_loss(y_p_pos,y_gt_pos,y_p_neg,m_p_pos,m_gt_pos)
        lay_loss=self.cal_lay_loss(inp_pos_indices, gt_pos_indices, inp_neg_indices,A_p,A_gt,gt_size)
        semantic_segmentation_loss=self.cal_semantic_segmentation_loss(semantic_mask,semantic_mask_gt)
        # semantic_segmentation_loss=0
        instance_dis_loss=self.cal_instance_dis_loss(mask_features,m_gt,gt_size)
        # instance_dis_loss=0
        # print(det_loss,lay_loss,semantic_segmentation_loss,instance_dis_loss)
        loss=self.lambda_1*det_loss+self.lambda_2*lay_loss+self.lambda_3*semantic_segmentation_loss+self.lambda_4*instance_dis_loss
        print(det_loss,lay_loss,semantic_segmentation_loss,instance_dis_loss)

        return loss





    