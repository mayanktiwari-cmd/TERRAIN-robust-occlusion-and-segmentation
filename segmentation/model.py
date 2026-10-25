"""
Multi-sensor road segmentation model: a Swin Transformer encoder + UNet decoder.

Uses timm's full Transformer `swin_tiny_patch4_window7_224` configured with a
4-channel input stem and `img_size=256` to match our processed tiles. Multi-scale
features (H/4, H/8, H/16, H/32) are extracted from the encoder stages and fused by
a light UNet-decoder with skip connections. Output is a 1-channel logit map at full
input resolution.

Optional Phase-C: `CONDITION_ON_GSD=True` prepends a small FiLM module that modulates
the bottleneck with log10(GSD), enabling resolution-conditioned decoding.
"""

import torch
import torch.nn as nn
import timm

import config


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample 2x, concat skip, 2x conv, halve channels."""

    def __init__(self, in_ch, cat_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBlock(in_ch + cat_ch, in_ch // 2),
            ConvBlock(in_ch // 2, in_ch // 2),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FiLM(nn.Module):
    """Phase-C conditioning: MLP(log10(GSD)) -> per-channel (gamma, beta) modulate."""

    def __init__(self, channels: int, cond_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.ReLU(inplace=True),
            nn.Linear(cond_dim, 2 * channels),
        )

    def forward(self, feature, gsd_log10):
        # feature: (B,C,H,W), gsd_log10: (B,)
        gb = self.mlp(gsd_log10.unsqueeze(-1))            # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.sigmoid() + 1.0                     # scale ~around 1
        beta = beta.tanh() * 0.5                          # shift ~around 0
        return feature * gamma[:, :, None, None] + beta[:, :, None, None]


class SwinUNet(nn.Module):
    def __init__(self, in_channels=None, encoder_weights=None,
                 condition_on_gsd=None):
        super().__init__()
        self.in_channels = in_channels or config.IN_CHANNELS
        self.condition_on_gsd = condition_on_gsd if condition_on_gsd is not None \
            else config.CONDITION_ON_GSD

        # Encoder: timm Swin with 4 input channels at our 256 input size.
        try:
            self.encoder = timm.create_model(
                config.ENCODER.removeprefix("tu-"),
                pretrained=bool(encoder_weights),
                img_size=config.IMG_SIZE,
                in_chans=self.in_channels,
            )
        except Exception:
            # offline / SSL-failure: fall back to random init
            self.encoder = timm.create_model(
                config.ENCODER.removeprefix("tu-"),
                pretrained=False,
                img_size=config.IMG_SIZE,
                in_chans=self.in_channels,
            )
        self.patch_embed = self.encoder.patch_embed
        self.encoder_stages = nn.ModuleList(self.encoder.layers)

        # stage channel widths (fine -> coarse)
        self._feat_chs = [96, 192, 384, 768]

        if self.condition_on_gsd:
            self.film = FiLM(self._feat_chs[-1])

        # Resolution-aware conditioning: a tiny additive embedding so the decoder
        # knows whether it sees 2 m Cartosat-3, 5.8 m LISS-4, or 10 m Sentinel-2.
        # Only +3*768 = 2304 params. gsd_map bucket ids: <=2 m -> 0, <=7 m -> 1, else 2.
        self.gsd_map = {"cartosat3": 0, "liss4": 1, "sentinel2": 2}
        self.gsd_embed = nn.Embedding(3, self._feat_chs[-1])

        # Decoder: 768@8 -> 384@16 -> 192@32 -> 96@64 -> 256 out
        self.d3 = DecoderBlock(self._feat_chs[3], self._feat_chs[2])
        self.d2 = DecoderBlock(self._feat_chs[2], self._feat_chs[1])
        self.d1 = DecoderBlock(self._feat_chs[1], self._feat_chs[0])
        self.up_out = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.head = nn.Sequential(
            ConvBlock(self._feat_chs[0], 64),
            nn.Conv2d(64, 1, 1),
        )

    def _encode(self, x):
        """Run the encoder; return list of (B,C,H,W) features fine->coarse."""
        cur = self.patch_embed(x)          # (B, H/4, W/4, C)
        outs = []
        for stage in self.encoder_stages:
            cur = stage(cur)
            cur = cur[0] if isinstance(cur, (tuple, list)) else cur
            outs.append(cur.permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, x, gsd_batch=None):
        e = self._encode(x)                # [96@64, 192@32, 384@16, 768@8]
        feat = e[3]
        if gsd_batch is not None:
            # Numeric GSD (m) -> resolution bucket id. If a sensor-name string was
            # passed instead, look it up in self.gsd_map.
            def _id(g):
                if isinstance(g, str):
                    return self.gsd_map[g.lower()]
                return 0 if g <= 2 else 1 if g <= 7 else 2
            idx = torch.tensor([_id(g) for g in gsd_batch],
                               dtype=torch.long, device=feat.device)
            feat = feat + self.gsd_embed(idx)[:, :, None, None]
        elif self.condition_on_gsd:
            g = torch.log10(torch.as_tensor(gsd_batch, device=feat.device)).float()
            feat = self.film(feat, g)
        feat = self.d3(feat, e[2])
        feat = self.d2(feat, e[1])
        feat = self.d1(feat, e[0])
        out = self.up_out(feat)
        return self.head(out)


if __name__ == "__main__":
    m = SwinUNet(encoder_weights=None)
    x = torch.randn(2, 4, 256, 256)
    y = m(x)
    print("SwinUNet output:", tuple(y.shape))
    y2 = m(x, gsd_batch=[2.0, 10.0])
    print("SwinUNet output (gsd):", tuple(y2.shape))
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.2f}M | condition_on_gsd={m.condition_on_gsd}")