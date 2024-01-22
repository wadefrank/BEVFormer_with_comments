# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,                     # list，大小为6，mlvl_feats[0].shape = torch.Size([1, 6, 256, 116, 200])
            bev_queries,                    # torch.Size([40000, 256])
            bev_h,                          # 200
            bev_w,                          # 200
            grid_length=[0.512, 0.512],     # (0.512, 0.512)
            bev_pos=None,                   # torch.Size([1, 256, 200, 200]) 位置编码
            prev_bev=None,                  # None
            **kwargs):
        """
        obtain bev features.
        """

        # bs = 1
        bs = mlvl_feats[0].size(0)

        # unsqueeze(n) 在第n维处增加一个维度
        # repeat(i,j,k) 在第1,2,3维处分别重复i,j,k次
        # bev_queries.shape = torch.Size([40000, 256])
        # bev_queries.unsqueeze(1).shape = torch.Size([40000, 1, 256])
        # bev_queries.unsqueeze(1).repeat(1, bs, 1) = torch.Size([40000, 1, 256])
        # bev_queries.shape = torch.Size([40000, 1, 256])
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)

        # flatten(n) 从第n维开始压平张量
        # bev_pos.shape = torch.Size([1, 256, 200, 200])
        # bev_pos.flatten(2).shape = torch.Size([1, 256, 40000])
        # bev_pos.flatten(2).permute(2, 0, 1) = torch.Size([40000, 1, 256])
        # bev_pos.shape = torch.Size([40000, 1, 256])
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        # kwargs['img_metas'][0]['can_bus'].shape = (18,)
        # can_bus 18维向量，分别表示pos(3)、orientation(4)、accel(3)、rotation_rate(3)、vel(3)和扩展位(2)

        # delta_x = array([0.]) (第一帧)
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])

        # delta_y = array([0.]) (第一帧)
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])

        # n帧本车坐标系在全局坐标系下的绝对角度，逆时针为正，顺时针为负 0~360
        # ego_angle = array([331.25864362])（第一帧）
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])

        # grid_length_y = 0.512 每个bev栅格代表y轴的实际长度，单位：米
        grid_length_y = grid_length[0]

        # grid_length_x = 0.512 每个bev栅格代表x轴的实际长度，单位：米
        grid_length_x = grid_length[1]

        # translation_length = array([0.]) (第一帧)
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)

        # translation_angle 表示两帧的平移向量在世界坐标系的角度
        # translation_angle = array([0.]) (第一帧)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180

        # bev_angle 表示两帧的平移向量在当前帧的角度
        bev_angle = ego_angle - translation_angle

        # shift_y表示在当前帧下，BEV特征图上的y轴的平移量（实际长度/单位长度/特征图大小）
        # shift_y = array([0.]) (第一帧)
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h

        # shift_x表示在当前帧下，BEV特征图上的x轴的平移量（实际长度/单位长度/特征图大小）
        # shift_x = array([0.]) (第一帧)
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w

        # self.use_shift = True
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift

        # shift.shape = torch.Size([1, 2])
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        # prev_bev = None (第一帧)
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        # can_bus.shape = torch.Size([1, 18])
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]

        # can_bus.shape = torch.Size([1, 1, 256])
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        # bev_queries.shape = torch.Size([40000, 1, 256])
        # can_bus.shape = torch.Size([1, 1, 256])
        # bev_queries = torch.Size([40000, 1, 256])
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            # lvl  feat.shape
            # 0    torch.Size([1, 6, 256, 116, 200])
            # 1    torch.Size([1, 6, 256, 58, 100])
            # 2    torch.Size([1, 6, 256, 29, 50])
            # 3    torch.Size([1, 6, 256, 15, 25])

            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)

            # lvl  feat.shape
            # 0    torch.Size([6, 1, 23200, 256])
            # 1    torch.Size([6, 1, 5800, 256])
            # 2    torch.Size([6, 1, 1450, 256])
            # 3    torch.Size([6, 1, 375, 256])
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                # self.cams_embeds.shape = torch.Size([6, 256])
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            # feat + self.level_embeds.shape = torch.Size([4, 256])
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)


        # feat_flatten.shape = torch.Size([6, 1, 30825, 256])
        feat_flatten = torch.cat(feat_flatten, 2)

        # spatial_shapes.shape = torch.Size([4, 2])
        # spatial_shapes = tensor([[116, 200],[ 58, 100],[ 29,  50],[ 15,  25]], device='cuda:0')
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)

        # level_start_index.shape = torch.Size([4])
        # level_start_index = tensor([    0, 23200, 29000, 30450], device='cuda:0')
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # feat_flatten = torch.Size([6, 30825, 1, 256])
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # encoder层
        bev_embed = self.encoder(
            bev_queries,                            # torch.Size([40000, 1, 256])
            feat_flatten,                           # torch.Size([6, 30825, 1, 256])
            feat_flatten,                           # torch.Size([6, 30825, 1, 256])
            bev_h=bev_h,                            # 200
            bev_w=bev_w,                            # 200
            bev_pos=bev_pos,                        # torch.Size([40000, 1, 256])
            spatial_shapes=spatial_shapes,          # torch.Size([4, 2])
            level_start_index=level_start_index,    # torch.Size([4])
            prev_bev=prev_bev,                      # None(第一帧)
            shift=shift,                            # torch.Size([1, 2]) 在当前帧的BEV特征图上的偏移量
            **kwargs                                # kwargs['img_metas']
        )

        return bev_embed

    # BEVFormer/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py
    # BEVFormerHead.forward()中 self.transformer() 调用该函数
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,                     # list，大小为6，mlvl_feats[0].shape = torch.Size([1, 6, 256, 116, 200])
                bev_queries,                    # torch.Size([40000, 256])
                object_query_embed,             # torch.Size([900, 512])
                bev_h,                          # 200
                bev_w,                          # 200
                grid_length=[0.512, 0.512],     # (0.512, 0.512)
                bev_pos=None,                   # torch.Size([1, 256, 200, 200]) 位置编码
                reg_branches=None,              # type(self.reg_branches) = <class 'torch.nn.modules.container.ModuleList'> 6层
                cls_branches=None,              # type(self.cls_branches) = None
                prev_bev=None,                  # prev_bev = None
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,                 # list，大小为6，mlvl_feats[0].shape = torch.Size([1, 6, 256, 116, 200])
            bev_queries,                # torch.Size([40000, 256])
            bev_h,                      # 200
            bev_w,                      # 200
            grid_length=grid_length,    # (0.512, 0.512)
            bev_pos=bev_pos,            # torch.Size([1, 256, 200, 200]) 位置编码
            prev_bev=prev_bev,          # None
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
