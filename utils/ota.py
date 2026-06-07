import torch.nn.functional as F
from utils.metrics import box_iou
from utils.torch_utils import de_parallel
from utils.general import xywh2xyxy

class ComputeLossOTA:
        # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device))
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEtheta = FocalLoss(BCEtheta, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.stride = det.stride
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEtheta = BCEtheta
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        tbox, tgaussian_theta = self.gettheta(p, targets)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        # pbt = []
        # for i in range(len(p)):
        #     pbt.append(p[i])
        # pbt[0] = pbt[0][:,:,:,:,0:55]
        # pbt[1] = pbt[1][:,:,:,:,0:55]
        # pbt[2] = pbt[2][:,:,:,:,0:55]
        # pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in pbt] 
    

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                if type(iou) is tuple:
                    lbox += (iou[1].detach() * (1 - iou[0])).mean()
                    iou = iou[0]
                else:
                    lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                class_index = 5 + self.nc
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:class_index], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:class_index], t)  # BCE

                # pstheta = p[i][b, a, gj, gi]
                t_theta = tgaussian_theta[i].type(ps.dtype) # target theta_gaussian_labels
                ltheta += self.BCEtheta(ps[:, class_index:], t_theta)
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + ltheta
        print(lbox.data)
        print(lobj.data)
        print(lcls.data)
        print(ltheta.data)
        return loss * bs, torch.cat((lbox, lobj, lcls, ltheta)).detach()

    def build_targets(self, pbt, targetsbt, imgs):
        indices, anch = self.find_3_positive(pbt, targetsbt)
        p = []
        for i in range(len(pbt)):
            p.append(pbt[i])
        p[0] = p[0][:,:,:,:,0:55]
        p[1] = p[1][:,:,:,:,0:55]
        p[2] = p[2][:,:,:,:,0:55]
        targets = targetsbt[:,0:7]
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # ttheta, tgaussian_theta = [], []
        tgaussian_theta = []
        # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets (tensor): (n_gt_all_batch, c) -> (na, n_gt_all_batch, c) -> (na, n_gt_all_batch, c+1)
        # targets (tensor): (na, n_gt_all_batch, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index]])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], # tensor: (5, 2)
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i] 
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0] # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # theta = t[:, 6]
            gaussian_theta_labels = t[:, 7:-1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            # tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            # tcls.append(c)  # class
            # ttheta.append(theta) # theta, θ∈[-pi/2, pi/2)
            # tgaussian_theta.append(gaussian_theta_labels)

        # return tcls, tbox, indices, anch

        return indices, anch
    
    def gettheta(self, p, targets):
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) pixel

        Return：non-normalized data
            tcls (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            tbox (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 4) featuremap pixel
            indices (list[P3_out,...]): len=self.na, tensor.size(4, n_filter2) [b, a, gj, gi]
            anch (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 2)
            tgaussian_theta (list[P3_out,...]): len=self.na, tensor.size(n_filter2, hyp['cls_theta'])
            # ttheta (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
        """
        # Build targets for compute_loss()
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # ttheta, tgaussian_theta = [], []
        tgaussian_theta = []
        # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets (tensor): (n_gt_all_batch, c) -> (na, n_gt_all_batch, c) -> (na, n_gt_all_batch, c+1)
        # targets (tensor): (na, n_gt_all_batch, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index]])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], # tensor: (5, 2)
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i] 
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0] # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # theta = t[:, 6]
            gaussian_theta_labels = t[:, 7:-1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # ttheta.append(theta) # theta, θ∈[-pi/2, pi/2)
            tgaussian_theta.append(gaussian_theta_labels)

        # return tcls, tbox, indices, anch
        return tbox, tgaussian_theta #, ttheta