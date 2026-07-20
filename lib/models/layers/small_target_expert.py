"""Independent lightweight RGB-event expert for small-target tracking."""

import torch
from torch import nn
from torch.nn import functional as F


class _DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=stride,
                padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, value):
        return self.block(value)


class _ModalityFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(channels * 3, 1, kernel_size=1)

    def forward(self, rgb, event):
        disagreement = (rgb - event).abs()
        rgb_weight = self.gate(
            torch.cat((rgb, event, disagreement), dim=1)).sigmoid()
        return rgb * rgb_weight + event * (1.0 - rgb_weight)


class _SpatialResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.body[3].weight)

    def forward(self, value):
        return value + self.body(value)


class _RGBEEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c2, c4, c8 = channels
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, c2),
            nn.GELU(),
        )
        self.event_stem = nn.Sequential(
            nn.Conv2d(3, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, c2),
            nn.GELU(),
        )
        self.rgb_s4 = _DepthwiseBlock(c2, c4, stride=2)
        self.event_s4 = _DepthwiseBlock(c2, c4, stride=2)
        self.rgb_s8 = _DepthwiseBlock(c4, c8, stride=2)
        self.event_s8 = _DepthwiseBlock(c4, c8, stride=2)
        self.rgb_s8_residual = _SpatialResidual(c8)
        self.event_s8_residual = _SpatialResidual(c8)
        self.fuse_s4 = _ModalityFusion(c4)
        self.fuse_s8 = _ModalityFusion(c8)

    def forward(self, rgb, event):
        rgb_s4 = self.rgb_s4(self.rgb_stem(rgb))
        event_s4 = self.event_s4(self.event_stem(event))
        rgb_s8 = self.rgb_s8_residual(self.rgb_s8(rgb_s4))
        event_s8 = self.event_s8_residual(self.event_s8(event_s4))
        return (
            self.fuse_s4(rgb_s4, event_s4),
            self.fuse_s8(rgb_s8, event_s8),
        )


class _TemplateTokenBoxRefiner(nn.Module):
    """Predict one template-conditioned correction to the final base box."""

    _TOKEN_GRID = 4
    _POOL_TEMPERATURE = 0.20

    def __init__(self):
        super().__init__()
        self.rgb_encoder = self._encoder()
        self.event_encoder = self._encoder()
        self.refine = nn.Sequential(
            nn.Conv2d(37, 16, kernel_size=1, bias=False),
            nn.GroupNorm(1, 16),
            nn.GELU(),
            nn.Conv2d(
                16, 16, kernel_size=5, padding=4, dilation=2,
                groups=16, bias=False),
            nn.GroupNorm(1, 16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.GroupNorm(1, 16),
            nn.GELU(),
        )
        self.output = nn.Conv2d(16, 4, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _encoder():
        return nn.Sequential(
            nn.Conv2d(
                3, 12, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.GroupNorm(1, 12),
            nn.GELU(),
            nn.Conv2d(
                12, 12, kernel_size=3, stride=2,
                padding=1, groups=12, bias=False),
            nn.Conv2d(12, 16, kernel_size=1, bias=False),
            nn.GroupNorm(1, 16),
            nn.GELU(),
        )

    @classmethod
    def _tokens(cls, feature):
        height, width = feature.shape[-2:]
        margin_y, margin_x = height // 4, width // 4
        target = feature[
            ..., margin_y:height - margin_y, margin_x:width - margin_x]
        if target.shape[-2] < 1 or target.shape[-1] < 1:
            raise ValueError("template target region must not be empty")
        return F.adaptive_avg_pool2d(
            target, (cls._TOKEN_GRID, cls._TOKEN_GRID)).flatten(2)

    @staticmethod
    def _validate_modalities(rgb, event):
        if rgb.ndim != 4 or event.ndim != 4:
            raise ValueError("box-refiner RGB and event inputs must be 4D")
        if rgb.shape != event.shape:
            raise ValueError("box-refiner RGB and event inputs must match")

    @staticmethod
    def _base_context(score_map, size_map, offset_map, spatial_size):
        if score_map.ndim != 4 or score_map.shape[1] != 1:
            raise ValueError("base score map must have shape [batch, 1, H, W]")
        expected = (score_map.shape[0], 2, *score_map.shape[-2:])
        if size_map.shape != expected or offset_map.shape != expected:
            raise ValueError("base size and offset maps must match the score grid")
        if score_map.shape[-2:] != spatial_size:
            raise ValueError("base prediction grid must match refiner evidence")
        logits = torch.logit(
            score_map.float().clamp(1e-4, 1.0 - 1e-4))
        return torch.cat((logits, size_map.float(), offset_map.float()), dim=1)

    def encode_template(self, rgb, event):
        self._validate_modalities(rgb, event)
        return (
            self._tokens(self.rgb_encoder(rgb)),
            self._tokens(self.event_encoder(event)),
        )

    @staticmethod
    def _correlate(tokens, search):
        tokens = F.normalize(tokens, dim=1)
        search = F.normalize(search, dim=1)
        return torch.einsum("bck,bchw->bkhw", tokens, search)

    def forward(
            self, template_tokens, search_rgb, search_event,
            base_score_map, base_size_map, base_offset_map):
        self._validate_modalities(search_rgb, search_event)
        if len(template_tokens) != 2:
            raise ValueError("box-refiner template cache must contain RGB and event")
        rgb_search = self.rgb_encoder(search_rgb)
        event_search = self.event_encoder(search_event)
        evidence = torch.cat((
            self._correlate(template_tokens[0], rgb_search),
            self._correlate(template_tokens[1], event_search),
        ), dim=1)
        base_context = self._base_context(
            base_score_map, base_size_map, base_offset_map,
            evidence.shape[-2:],
        ).detach().to(evidence)
        residual_map = self.output(
            self.refine(torch.cat((evidence, base_context), dim=1)))
        weights = F.softmax(
            base_context[:, :1].flatten(2) / self._POOL_TEMPERATURE,
            dim=-1,
        )
        return (residual_map.flatten(2) * weights).sum(dim=-1).unsqueeze(1)


class _SmallTargetHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.shared = _DepthwiseBlock(channels, channels)
        self.refine = _DepthwiseBlock(channels, channels)
        nn.init.zeros_(self.refine.block[4].weight)
        nn.init.zeros_(self.refine.block[4].bias)
        self.center = self._branch(channels, 1)
        self.size = self._branch(channels, 2)
        self.offset = self._branch(channels, 2)

    @staticmethod
    def _branch(channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, output_channels, kernel_size=1),
        )

    def forward(self, feature):
        feature = self.shared(feature)
        feature = feature + self.refine(feature)
        score_map = self.center(feature).sigmoid().clamp(1e-4, 1.0 - 1e-4)
        size_map = self.size(feature).sigmoid().clamp(1e-4, 1.0)
        offset_map = self.offset(feature).tanh() * 0.5
        return score_map, size_map, offset_map


class SmallTargetExpert(nn.Module):
    """Track small targets without invoking the shared ViT or Hopfield path."""

    _BOX_DECODE_TEMPERATURE = 0.20
    _BOX_RESIDUAL_SCALE = 0.125

    def __init__(self, search_size=256, channels=(32, 64, 96)):
        super().__init__()
        if search_size % 4:
            raise ValueError("search_size must be divisible by four")
        self.search_size = int(search_size)
        self.feat_sz = self.search_size // 4
        self.encoder = _RGBEEncoder(channels)
        c4, c8 = channels[1:]
        self.coarse_projection = nn.Conv2d(c8, c4, kernel_size=1, bias=False)
        self.match_refinement = _DepthwiseBlock(c4 * 2 + 2, c4)
        self.head = _SmallTargetHead(c4)
        self.box_refiner = _TemplateTokenBoxRefiner()

    @staticmethod
    def _flatten_frames(rgb, event):
        if rgb.ndim == 4 and event.ndim == 4:
            return rgb, event, rgb.shape[0], 1
        if rgb.ndim != 5 or event.ndim != 5:
            raise ValueError("RGB and event inputs must both be 4D or 5D")
        if rgb.shape[:2] != event.shape[:2]:
            raise ValueError("RGB and event frame dimensions must match")
        batch_size, frame_count = rgb.shape[:2]
        return (
            rgb.flatten(0, 1), event.flatten(0, 1),
            batch_size, frame_count,
        )

    def _encode_frames(self, rgb, event, average):
        rgb, event, batch_size, frame_count = self._flatten_frames(rgb, event)
        s4, s8 = self.encoder(rgb, event)
        s4 = s4.reshape(batch_size, frame_count, *s4.shape[1:])
        s8 = s8.reshape(batch_size, frame_count, *s8.shape[1:])
        if average:
            return s4.mean(dim=1), s8.mean(dim=1)
        return s4[:, -1], s8[:, -1]

    @staticmethod
    def _spatial_match(template, search):
        if template.ndim != 4 or search.ndim != 4:
            raise ValueError("template and search features must be 4D")
        if template.shape[:2] != search.shape[:2]:
            raise ValueError("template and search batch/channels must match")
        height, width = template.shape[-2:]
        margin_y, margin_x = height // 4, width // 4
        target = template[
            ..., margin_y:height - margin_y, margin_x:width - margin_x]
        kernel_height, kernel_width = target.shape[-2:]
        if kernel_height < 1 or kernel_width < 1:
            raise ValueError("template target region must not be empty")

        batch_size, channels = search.shape[:2]
        kernels = F.normalize(target.flatten(1), dim=1).reshape(
            batch_size, channels, kernel_height, kernel_width)
        pad_left = (kernel_width - 1) // 2
        pad_right = kernel_width // 2
        pad_top = (kernel_height - 1) // 2
        pad_bottom = kernel_height // 2
        padded = F.pad(
            search, (pad_left, pad_right, pad_top, pad_bottom))
        response = F.conv2d(
            padded.reshape(1, batch_size * channels, *padded.shape[-2:]),
            kernels,
            groups=batch_size,
        ).reshape(batch_size, 1, *search.shape[-2:])
        patch_energy = F.avg_pool2d(
            padded.square().sum(dim=1, keepdim=True),
            kernel_size=(kernel_height, kernel_width),
            stride=1,
        ) * float(kernel_height * kernel_width)
        response = response / patch_energy.clamp_min(1e-8).sqrt()
        centered = response - response.mean(dim=(-2, -1), keepdim=True)
        scale = centered.square().mean(
            dim=(-2, -1), keepdim=True).add(1e-4).rsqrt()
        return 2.0 * (centered * scale).sigmoid()

    def _decode_boxes(self, score_map, size_map, offset_map):
        logits = torch.logit(
            score_map.float().clamp(1e-4, 1.0 - 1e-4))
        weights = F.softmax(
            logits.flatten(1) / self._BOX_DECODE_TEMPERATURE, dim=1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.feat_sz, device=score_map.device),
            torch.arange(self.feat_sz, device=score_map.device),
            indexing="ij",
        )
        grid_x = grid_x.to(weights).flatten().unsqueeze(0)
        grid_y = grid_y.to(weights).flatten().unsqueeze(0)
        flat_size = size_map.float().flatten(2)
        flat_offset = offset_map.float().flatten(2)
        boxes = torch.stack((
            (weights * (grid_x + flat_offset[:, 0])).sum(dim=1)
            / self.feat_sz,
            (weights * (grid_y + flat_offset[:, 1])).sum(dim=1)
            / self.feat_sz,
            (weights * flat_size[:, 0]).sum(dim=1),
            (weights * flat_size[:, 1]).sum(dim=1),
        ), dim=1)
        return boxes.clamp(0.0, 1.0).unsqueeze(1)

    def _validate_search(self, search_rgb, search_event):
        if search_rgb.shape != search_event.shape:
            raise ValueError("RGB and event search tensors must have equal shapes")
        if search_rgb.shape[-2:] != (self.search_size, self.search_size):
            raise ValueError(
                "search tensor size must match configured search_size")
        if search_rgb.ndim == 5 and search_rgb.shape[1] != 1:
            raise ValueError("small-target expert expects exactly one search frame")

    def encode_template(self, template_rgb, template_event):
        template_s4, template_s8 = self._encode_frames(
            template_rgb, template_event, average=True)
        rgb, event, batch_size, frame_count = self._flatten_frames(
            template_rgb, template_event)
        box_tokens = tuple(
            value.reshape(batch_size, frame_count, *value.shape[1:]).mean(dim=1)
            for value in self.box_refiner.encode_template(rgb, event)
        )
        return template_s4, template_s8, box_tokens

    def track_with_template(
            self, template_features, search_rgb, search_event):
        self._validate_search(search_rgb, search_event)
        template_s4, template_s8, box_tokens = template_features
        search_s4, search_s8 = self._encode_frames(
            search_rgb, search_event, average=False)
        match_s4 = self._spatial_match(template_s4, search_s4)
        match_s8 = self._spatial_match(template_s8, search_s8)
        coarse = self.coarse_projection(search_s8)
        coarse = F.interpolate(
            coarse, size=search_s4.shape[-2:],
            mode="bilinear", align_corners=False)
        match_s8 = F.interpolate(
            match_s8, size=search_s4.shape[-2:],
            mode="bilinear", align_corners=False)
        feature = self.match_refinement(torch.cat((
            search_s4 * match_s4,
            coarse * match_s8,
            match_s4,
            match_s8,
        ), dim=1))
        base_score_map, size_map, offset_map = self.head(feature)
        flat_rgb, flat_event, _, _ = self._flatten_frames(
            search_rgb, search_event)
        raw_box_delta = self.box_refiner(
            box_tokens, flat_rgb, flat_event,
            base_score_map, size_map, offset_map)
        base_pred_boxes = self._decode_boxes(
            base_score_map, size_map, offset_map)
        box_delta = self._BOX_RESIDUAL_SCALE * raw_box_delta.tanh()
        pred_boxes = torch.cat((
            (base_pred_boxes[..., :2] + box_delta[..., :2]).clamp(0.0, 1.0),
            (base_pred_boxes[..., 2:] + box_delta[..., 2:]).clamp(1e-4, 1.0),
        ), dim=-1)
        return {
            "pred_boxes": pred_boxes,
            "score_map": base_score_map,
            "size_map": size_map,
            "offset_map": offset_map,
            "small_match_map": match_s4,
            "small_base_pred_boxes": base_pred_boxes,
            "small_box_delta": box_delta,
        }

    def forward(self, template_rgb, template_event, search_rgb, search_event):
        template_features = self.encode_template(
            template_rgb, template_event)
        return self.track_with_template(
            template_features, search_rgb, search_event)
