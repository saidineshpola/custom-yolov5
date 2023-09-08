import math
import torch


def bbox_iou(box1,
             box2,
             xywh=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             SIoU=False,
             EIoU=False,
             WIoU=False,
             Focal=False,
             alpha=1,
             gamma=0.5,
             scale=False,
             monotonous=False,
             eps=1e-7):
    """
    Create args and returns for bbox_iou
    :param box1: (tensor) bounding boxes, shape: [n,4] or [1,4]
    :param box2: (tensor) bounding boxes, shape: [n,4] or [1,4]
    :param xywh: (bool) if True the boxes are in xywh format, else in xyxy
    :param GIoU: (bool) if True the Generalized IoU will be computed
    :param DIoU: (bool) if True the Distance IoU will be computed
    :param CIoU: (bool) if True the Complete IoU will be computed
    :param SIoU: (bool) if True the Shape IoU will be computed
    :param EIoU: (bool) if True the Extended IoU will be computed
    :param WIoU: (bool) if True the Wise IoU will be computed
    :param Focal: (bool) if True the Focal IoU will be computed
    :param alpha: (float) alpha parameter for Focal IoU
    :param gamma: (float) gamma parameter for Focal IoU
    :param scale: (bool) if True the Scale IoU will be computed
    :param monotonous: (bool) if True the Scale IoU will be computed
    :param eps: (float) epsilon to avoid zero division
    :return: (tensor) iou
    """
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        wise_scale = WIoU_Scale(1 - (inter / union), monotonous=monotonous)

    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter / (union + eps), alpha)  # alpha iou
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                        b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                if Focal:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter / (union + eps),
                                                                                                 gamma)  # Focal_CIoU
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Focal:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps), gamma)  # Focal_EIou
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw, s_ch = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps, (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1, sin_alpha_2 = torch.abs(s_cw) / sigma, torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x, rho_y = (s_cw / cw) ** 2, (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w, omiga_h = torch.abs(w1 - w2) / torch.max(w1, w2), torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(
                        inter / (union + eps), gamma)  # Focal_SIou
                return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)  # SIou
            elif WIoU: # https://arxiv.org/abs/2301.10051
                if scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(wise_scale), (1 - iou) * torch.exp((rho2 / c2)), iou  # WIoU v3 
                return iou, torch.exp((rho2 / c2))  # WIoU v1
            if Focal:
                return iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)  # Focal_DIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter / (union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        return iou, torch.pow(inter / (union + eps), gamma)  # Focal_IoU
    return iou  # IoU


class WIoU_Scale:
    """
    monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean
    """
    iou_mean = 1.
    _momentum = 1 - pow(0.5, exp=1 / 7000)
    _is_train = True

    def __init__(self, iou, monotonous=False):
        self.iou = iou
        self.monotonous = monotonous
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1