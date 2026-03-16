import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from abc import ABC, abstractmethod

class RegionPooling(nn.Module):
    def __init__(self, num_sample_point):
        super().__init__()
        self.num_sample_point = num_sample_point
        self.pooler = nn.AdaptiveAvgPool1d(output_size=1)

    def extract_region_feature(self, region_feature_map, region_masks, original_dtype, return_dtype):
        assert len(region_feature_map) == len(region_masks)
        all_points = []
        all_points_fea = []
        all_points_img_ids = []
        for img_id, (region_feature_map_i, region_masks_list_i) in enumerate(zip(region_feature_map, region_masks)):
            # print(f'type(region_feature_map_i): {type(region_feature_map_i)}, type(region_masks_list_i): {type(region_masks_list_i)}')
            # print(f'region_feature_map_i shape: {region_feature_map_i.shape}, num of region_masks_list_i: {region_masks_list_i.shape}')# torch.Size([256, 2048]) torch.Size([5, 448, 448])
            # [H*W, C]
            if len(region_masks_list_i) != 0:
                ori_image_wh = torch.tensor([region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]], device=region_masks_list_i[0].device)[None,]
                # print(f'ori_image_wh: {ori_image_wh}')
                # [num_sample_point, 2]
                # for m in region_masks_list_i:
                #     if m.nonzero().shape[0] <=0:
                #         print('error')
                #     else:
                #         print(f'true: m.nonzero().shape: {m.nonzero().shape}')

                cur_non_zero_pos = [rand_sample_repeat((m.nonzero() / ori_image_wh), self.num_sample_point) for m
                                    in
                                    region_masks_list_i] # torch.Size([5, 256, 2])
                # [num_mask, num_sample_point, 2]
                cur_non_zero_pos = torch.stack(cur_non_zero_pos) # torch.Size([5, 256, 2])

                h = w = int(math.sqrt(region_feature_map_i.shape[0])) # 16
                c = region_feature_map_i.shape[-1] # 2048

                dup_region_feature_map_i = region_feature_map_i.reshape(h, w, c).permute(2, 0, 1) # torch.Size([2048, 16, 16])
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(cur_non_zero_pos.shape[0], 1, 1,
                                                                                        1) # torch.Size([5, 2048, 16, 16])
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                '''
                region_feature_i function:
                input: [num_mask, C, H, W], [num_mask, num_sample_point, 2]
                output: [num_mask, C, num_sample_point]
                '''
                region_feature_i = _point_sample(dup_region_feature_map_i_ori_type,
                                                cur_non_zero_pos.flip(dims=(2,)).type(original_dtype),
                                                return_dtype,
                                                align_corners=True,
                                                ) # torch.Size([5, 2048, 256])
                # [num_mask, num_sample_point, C]
                region_feature_i = region_feature_i.transpose(-2, -1) # torch.Size([5, 256, 2048])

                cur_img_id = [img_id] * len(cur_non_zero_pos) # torch.Size([5])

                all_points.append(cur_non_zero_pos)
                all_points_fea.append(region_feature_i)
                all_points_img_ids.extend(cur_img_id)

        return all_points, all_points_fea, all_points_img_ids

    def forward(self, feature_map, region_masks, original_dtype, return_dtype):
        assert len(feature_map) == len(region_masks)
        batch_size = len(feature_map)
        all_points, all_points_fea, all_points_img_ids = self.extract_region_feature(feature_map, region_masks,
                                                                                     original_dtype, return_dtype)

        if len(all_points) == 0:
            return [None] * len(region_masks)

        all_points = torch.cat(all_points, dim=0).to(return_dtype)
        all_points_fea = torch.cat(all_points_fea, dim=0).to(return_dtype)
        all_points_img_ids = torch.tensor(all_points_img_ids, device=all_points_fea.device)

        region_feat = self.pooler(all_points_fea.transpose(-2, -1)).transpose(-2, -1)

        region_feature_list = []
        for bs in range(batch_size):
            index = all_points_img_ids == bs
            region_feature_list.append(region_feat[index])
        # print(f'Extract {len(all_points)} region features from {batch_size} images.')
        return region_feature_list

def rand_sample_repeat(x, max_len):
    # debug: 处理空张量情况
    if x.shape[0] == 0:
        # 创建一个全零张量作为替代
        # 假设每个点包含2维坐标 (x,y)
        return torch.zeros(max_len, x.shape[1] if len(x.shape) > 1 else 2, device=x.device, dtype=x.dtype)
    
    if x.shape[0] < max_len:
        if x.shape[0] == 0:
            # 如果 x.shape[0] 为 0，直接返回全零张量
            return torch.zeros(max_len, x.shape[1] if len(x.shape) > 1 else 2, device=x.device, dtype=x.dtype)
        indices = torch.randint(0, x.shape[0], (max_len - x.shape[0],))
        return torch.cat((x, x[indices]), dim=0)
    elif x.shape[0] == max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
        return x[rand_idx, :]

def _point_sample(input, point_coords, return_dtype, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    output = F.grid_sample(input.float(), (2.0 * point_coords - 1.0).float().to(input.device), **kwargs)
    output = output.to(return_dtype)
    if add_dim:
        output = output.squeeze(3)
    return output


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx
