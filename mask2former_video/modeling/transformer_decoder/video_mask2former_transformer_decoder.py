# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# ------------------------------------------------------------------------------------------------
# Modified by Kaixuan Lu from https://github.com/facebookresearch/CutLER/tree/main/videocutler
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

from .position_encoding import PositionEmbeddingSine3D

from einops import rearrange
import math

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class MaskIoUFeatureExtractor(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(
            self,
            mask_the_feature=False,
            adaptive_pooling=False,
        ):
        super(MaskIoUFeatureExtractor, self).__init__()

        input_channels = 257

        self.maskiou_fcn1 = nn.Conv2d(input_channels, 256, 3, 1, 1)
        self.maskiou_fcn2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maskiou_fcn3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maskiou_fcn4 = nn.Conv2d(256, 256, 3, 2, 1)
        self.maskiou_fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)
        self.maskiou_fc3 = nn.Linear(1024, 1)

        for l in [
            self.maskiou_fcn1,
            self.maskiou_fcn2,
            self.maskiou_fcn3,
            self.maskiou_fcn4,
        ]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.maskiou_fc3.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou_fc3.bias, 0)

        self.mask_the_feature = mask_the_feature

        self.adaptive_pooling = adaptive_pooling
        if self.adaptive_pooling:
            self.adaptive_pool = nn.AdaptiveMaxPool2d((7, 7))

    def forward(self, x, mask):
        if self.mask_the_feature:
            b, q, t, _, _ = mask.shape
            mask = rearrange(mask, "b q t h w -> (b q t) () h w")
            mask = F.interpolate(
                mask,
                size=(28, 28),
                mode="bilinear",
                align_corners=False,
            )
        elif self.adaptive_pooling:
            b, q, t, _, _ = mask.shape
            _, _, h, w = x.shape
            mask = rearrange(mask, "b q t h w -> (b q t) () h w")
            mask = F.interpolate(
                mask,
                size=(2 * h, 2 * w),
                mode="bilinear",
                align_corners=False,
            )
            x = rearrange(x, "(b t) c h w -> b () t c h w", b=b, t=t)
            x = x.expand(b, q, t, -1, -1, -1)
            x = rearrange(x, "b q t c h w -> (b q t) c h w")
        else:
            b, q, t, _, _ = mask.shape
            mask = rearrange(mask, "b q t h w -> (b q t) () h w")
            mask = F.interpolate(
                mask,
                size=(28, 28),
                mode="bilinear",
                align_corners=False,
            )
            x = rearrange(x, "(b t) c h w -> b () t c h w", b=b, t=t)
            x = x.expand(b, q, t, -1, -1, -1)
            x = rearrange(x, "b q t c h w -> (b q t) c h w")
            x = F.interpolate(
                x,
                size=(14, 14),
                mode="bilinear",
                align_corners=False,
            )
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        if self.adaptive_pooling:
            x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        x = self.maskiou_fc3(x)
        x = rearrange(x, "(b q t) c -> b q t c", b=b, q=q, t=t)

        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)

class MaskIoUFeatureExtractorV2(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(
            self,
        ):
        super(MaskIoUFeatureExtractorV2, self).__init__()

        input_channels = 257

        self.maskiou_fcn1 = nn.Conv2d(input_channels, 256, 3, 1, 1)
        self.maskiou_fcn2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maskiou_fcn3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maskiou_fcn4 = nn.Conv2d(256, 256, 3, 2, 1)
        self.maskiou_fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)
        self.maskiou_fc3 = nn.Linear(1024, 1)

        for l in [
            self.maskiou_fcn1,
            self.maskiou_fcn2,
            self.maskiou_fcn3,
            self.maskiou_fcn4,
        ]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.maskiou_fc3.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou_fc3.bias, 0)

    def forward(self, x, mask):
        b, q, t, _, _ = mask.shape
        mask = rearrange(mask, "b q t h w -> (b q t) () h w")
        mask = F.interpolate(
            mask,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )
        x = rearrange(x, "b q t c h w -> (b q t) c h w")
        x = F.interpolate(
            x,
            size=(14, 14),
            mode="bilinear",
            align_corners=False,
        )
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        x = self.maskiou_fc3(x)
        x = rearrange(x, "(b q t) c -> b q t c", b=b, q=q, t=t)

        return x

class ViTMaskIoUHead(nn.Module):
    def __init__(
        self,
        score_threshold=0.8,
    ):
        super().__init__()
        self.score_threshold = score_threshold

        input_channels = 257

        self.input_proj = nn.Conv2d(input_channels, 256, kernel_size=1)
        self.pos3d = PositionEmbeddingSine3D(256 // 2, normalize=True)
        self.pos2d_encoder = PositionalEncoding(256, dropout=0.1, max_len=5000)
        transformer_decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_decoder_layer, num_layers=6
        )
        self.iou_head = nn.Linear(256, 1)

    def forward(
        self,
        img_feature,
        preds_class,
        queries,
        preds_mask,
    ):
        def get_batch_indices(preds_class, score_threshold=0.8):
            preds_score = F.softmax(preds_class, dim=-1)[:, :, :-1]
            preds_score = preds_score.flatten(1, 2)
            query_indices = preds_score >= score_threshold
            query_indices = query_indices.nonzero(as_tuple=True)
            index_dict = {}
            for id1, id2 in zip(*query_indices):
                id1 = id1.item()
                id2 = id2.item()
                if id1 not in index_dict:
                    index_dict[id1] = []
                index_dict[id1].append(id2)
            batch_indices = []
            for id1, indices in index_dict.items():
                batch_indices.append(([id1] * len(indices), indices))
            return batch_indices

        batch_indices = get_batch_indices(
            preds_class, score_threshold=self.score_threshold
        )

        b, q, t, _, _ = preds_mask.shape
        _, _, h, w = img_feature.shape

        preds_mask = rearrange(preds_mask, "b q t h w -> (b q t) () h w")
        preds_mask = F.interpolate(
            preds_mask,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )
        preds_mask = F.max_pool2d(preds_mask, kernel_size=2, stride=2)

        img_feature = F.interpolate(
            img_feature,
            size=(14, 14),
            mode="bilinear",
            align_corners=False,
        )

        img_feature = rearrange(img_feature, "(b t) c h w -> b () t c h w", b=b, t=t)
        img_feature = img_feature.expand(b, q, t, -1, -1, -1)
        img_feature = rearrange(img_feature, "b q t c h w -> (b q t) c h w")

        x = torch.cat((img_feature, preds_mask), 1)
        x = self.input_proj(x)
        x = rearrange(x, "(b q t) c h w -> (b t) q c h w", b=b, q=q, t=t)
        pos_x = self.pos3d(x)
        x = x + pos_x
        x = rearrange(x, "(b t) q c h w -> b q t c h w", b=b, q=q, t=t)
        queries = queries.unsqueeze(2)
        queries = queries.expand(-1, -1, t, -1)

        batch_ious = []
        for batch_index in batch_indices:
            batch_x = x[batch_index]
            batch_queries = queries[batch_index]
            batch_x = rearrange(batch_x, "q t c h w -> t (q h w) c")
            batch_x = self.pos2d_encoder(batch_x)
            batch_queries = rearrange(batch_queries, "q t c -> t q c")
            batch_queries = self.pos2d_encoder(batch_queries)
            batch_x = self.transformer_decoder(
                tgt=batch_x,
                memory=batch_queries,
            )
            # global average pooling
            batch_x = batch_x.mean(dim=1)
            tmp_ious = self.iou_head(batch_x)
            tmp_ious = tmp_ious.sigmoid()
            tmp_ious = tmp_ious.squeeze(1)
            batch_ious.append(tmp_ious)
        if len(batch_ious) == 0:
            return None
        batch_ious = torch.stack(batch_ious, dim=0)
        return batch_ious


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
        output_maskiou=False,
        output_maskiou_v2=False,
        mask_the_feature=False,
        adaptive_pooling=False,
        vit_maskious=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_frames = num_frames

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        if output_maskiou:
            self.maskiou_feature_extractor = MaskIoUFeatureExtractor(mask_the_feature, adaptive_pooling)
        if output_maskiou_v2:
            self.maskiou_feature_extractor = MaskIoUFeatureExtractorV2()
        self.output_maskiou = output_maskiou
        self.output_maskiou_v2 = output_maskiou_v2
        self.mask_the_feature = mask_the_feature
        self.adaptive_pooling = adaptive_pooling
        if adaptive_pooling and mask_the_feature:
            raise ValueError("Adaptive pooling and mask the feature cannot be used at the same time.")
        self.vit_maskious = vit_maskious
        if vit_maskious:
            self.vit_maskiou_head = ViTMaskIoUHead()
        if vit_maskious and output_maskiou:
            raise ValueError("vit_mask and output_maskiou cannot be used at the same time.")

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM

        if hasattr(cfg, "OUTPUT_MASKIOUS") and cfg.OUTPUT_MASKIOUS:
            output_maskiou = True
        else:
            output_maskiou = False
        if hasattr(cfg, "OUTPUT_MASKIOUS_V2") and cfg.OUTPUT_MASKIOUS_V2:
            output_maskiou_v2 = True
        else:
            output_maskiou_v2 = False
        if hasattr(cfg, "MASK_THE_FEATURE") and cfg.MASK_THE_FEATURE:
            mask_the_feature = True
        else:
            mask_the_feature = False
        if hasattr(cfg, "ADAPTIVE_POOLING") and cfg.ADAPTIVE_POOLING:
            adaptive_pooling = True
        else:
            adaptive_pooling = False
        if hasattr(cfg, "VIT_MASKIOUS") and cfg.VIT_MASKIOUS:
            vit_maskious = True
        else:
            vit_maskious = False
        ret["output_maskiou"] = output_maskiou
        ret["output_maskiou_v2"] = output_maskiou_v2
        ret["mask_the_feature"] = mask_the_feature
        ret["adaptive_pooling"] = adaptive_pooling
        ret["vit_maskious"] = vit_maskious

        return ret

    def forward(self, x, mask_features, mask = None):
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            _, c, hw = src[-1].shape
            pos[-1] = pos[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            src[-1] = src[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        queries = output.permute(1, 0, 2)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        def get_memory_feature(
            query: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            memory_pos: torch.Tensor,
            query_pos: torch.Tensor,
            cross_attn_layer: torch.nn.Module,
            t: int,
            h: int,
            w: int,
        ):
            """
            Get the memory feature
            :param query: The query feature
            :param memory: The memory feature
            :param memory_mask: The memory mask
            :param memory_pos: The memory position
            :param query_pos: The query position
            :return: The memory feature
            """
            attn_output, attn_output_weights = cross_attn_layer.multihead_attn(
                query=cross_attn_layer.with_pos_embed(query, query_pos),
                key=cross_attn_layer.with_pos_embed(memory, memory_pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=None,
                # average_attn_weights=False,
            )
            q = query.shape[0]
            memory_feature = rearrange(memory, "l b c -> b () l c")
            memory_feature = memory_feature.expand(-1, q, -1, -1)
            memory_feature = memory_feature * attn_output_weights.unsqueeze(-1)
            memory_feature = rearrange(memory_feature, "b q (t h w) c -> b q t c h w", t=t, h=h, w=w)
            return memory_feature
        
        memory_features = None

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            if self.output_maskiou_v2 and i == 0:
                memory_features = get_memory_feature(
                query=output,
                memory=src[level_index],
                memory_mask=attn_mask,
                memory_pos=pos[level_index],
                query_pos=query_embed,
                cross_attn_layer=self.transformer_cross_attention_layers[i],
                t=t,
                h=x[level_index].shape[2],
                w=x[level_index].shape[3],
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        predictions_iou = []
        predictions_frame_ious = []
        
        if self.output_maskiou and self.mask_the_feature:
            predictions_iou = [None] * len(predictions_mask)
            def mask_the_feature(x, mask):
                b, q, t, _, _ = mask.shape
                mask = rearrange(mask, "b q t h w -> (b q t) () h w")
                mask = mask.sigmoid()
                mask = mask > 0.5
                mask = mask.float().detach()
                mask = F.interpolate(
                    mask,
                    size=(28, 28),
                    mode="bilinear",
                    align_corners=False,
                )
                x = rearrange(x, "(b t) c h w -> b () t c h w", b=b, t=t)
                x = x.expand(b, q, t, -1, -1, -1)
                x = rearrange(x, "b q t c h w -> (b q t) c h w")
                x = F.interpolate(
                    x,
                    size=(28, 28),
                    mode="bilinear",
                    align_corners=False,
                )
                x = x * mask
                x = F.interpolate(
                    x,
                    size=(14, 14),
                    mode="bilinear",
                    align_corners=False,
                )
                return x
            mask_x = mask_the_feature(x[1], predictions_mask[1])
            predictions_iou[-1] = self.maskiou_feature_extractor(
                # mask_x.detach(), predictions_mask[-1].detach()
                mask_x, predictions_mask[-1]
            )
            aux_outputs = self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_iou
            )
        else:
            if self.output_maskiou:
                predictions_frame_ious = [None] * len(predictions_mask)
                if self.training:
                    for pred_mask in predictions_mask:
                        pred_maskiou = self.maskiou_feature_extractor(x[0], pred_mask)
                        predictions_iou.append(pred_maskiou)
                    aux_outputs = self._set_aux_loss(
                        predictions_class if self.mask_classification else None, predictions_mask, predictions_iou
                    )
                else:
                    predictions_iou.append(self.maskiou_feature_extractor(x[0], predictions_mask[-1]))
                    aux_outputs = None
            elif self.output_maskiou_v2:
                predictions_frame_ious = [None] * len(predictions_mask)
                if self.training:
                    for pred_mask in predictions_mask:
                        pred_maskiou = self.maskiou_feature_extractor(memory_features, pred_mask)
                        predictions_iou.append(pred_maskiou)
                    aux_outputs = self._set_aux_loss(
                        predictions_class if self.mask_classification else None, predictions_mask, predictions_iou
                    )
                else:
                    predictions_iou.append(self.maskiou_feature_extractor(memory_features, predictions_mask[-1]))
                    aux_outputs = None
            elif self.vit_maskious:
                for pred_mask, pred_class in zip(predictions_mask, predictions_class):
                    frame_ious = self.vit_maskiou_head(
                        img_feature=x[0],
                        preds_class=pred_class,
                        queries=queries,
                        preds_mask=pred_mask,
                    )
                    predictions_frame_ious.append(frame_ious)
                    predictions_iou = [None] * len(predictions_mask)
                aux_outputs = self._set_aux_loss(
                    predictions_class if self.mask_classification else None,
                    predictions_mask,
                    predictions_iou,
                    predictions_frame_ious,
                )
            else:
                predictions_iou = [None] * len(predictions_mask)
                predictions_frame_ious = [None] * len(predictions_mask)
                aux_outputs = self._set_aux_loss(
                    predictions_class if self.mask_classification else None,
                    predictions_mask,
                    predictions_iou,
                    predictions_frame_ious,
                )

   
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': aux_outputs,           
            'pred_maskiou': predictions_iou[-1],
            'pred_frame_ious': predictions_frame_ious[-1],
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
        b, q, t, _, _ = outputs_mask.shape

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class,
        outputs_seg_masks,
        outputs_iou,
        outputs_frame_ious=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_frame_ious is None:
            outputs_frame_ious = [None] * len(outputs_seg_masks)

        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_maskiou": c, "pred_frame_ious": d}
                for a, b, c, d in zip(
                    outputs_class[:-1],
                    outputs_seg_masks[:-1],
                    outputs_iou[:-1],
                    outputs_frame_ious[:-1]
                )
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
