from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
from aplf.blocks import SEBlock, ResBlock, SCSE, UpSample
import torch
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.block = nn.Sequential(
            ResBlock(
                in_ch=in_ch,
                out_ch=out_ch,
            ),
            SCSE(out_ch),
        )
        self.block = nn.Sequential(
            ResBlock(
                in_ch=in_ch,
                out_ch=out_ch,
            ),
            SCSE(out_ch),
        )

        self.down = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        out = self.block(x)
        conv = out
        down = self.down(conv)
        return down, conv


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, feature_size=64):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(in_ch, feature_size),
            DownSample(feature_size, feature_size * 2 ** 1),
            DownSample(feature_size * 2 ** 1, feature_size * 2 ** 2),
            DownSample(feature_size * 2 ** 2, out_ch),
        ])

    def forward(self, x):
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
        return x


class Fc(nn.Module):
    def __init__(self, in_ch, out_ch, feature_size=64, r=2):
        super().__init__()
        self.before_pool = ResBlock(
            in_ch=in_ch,
            out_ch=in_ch//r,
        ),
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//r//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//r//r, out_ch),
        )
        self.out_ch = out_ch

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_ch, 1, 1)
        return y


class UNet(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 feature_size=64,
                 depth=3,
                 ratio=2):
        super().__init__()
        self.down_layers = nn.ModuleList(
            [
                DownSample(
                    in_ch=in_ch,
                    out_ch=feature_size,
                ),
                *pipe(
                    range(depth),
                    map(lambda d: DownSample(
                        in_ch=int(feature_size*ratio**(d)),
                        out_ch=int(feature_size*ratio**(d + 1)),
                    )),
                    list,
                )
            ]
        )
        self.center = DownSample(
            in_ch=feature_size*ratio**depth,
            out_ch=feature_size*ratio**depth,
        )
        self.up_layers = nn.ModuleList([
            *pipe(
                range(depth),
                reversed,
                map(lambda l: UpSample(
                    in_ch=feature_size*ratio**(l+1) + feature_size*ratio**(l+1),
                    out_ch=feature_size*ratio**l,
                )),
                list,
            ),
            UpSample(
                in_ch=feature_size + feature_size,
                out_ch=out_ch,
            ),
        ])

    def forward(self, x):
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)
        d_outs = list(reversed(d_outs))
        x, _ = self.center(x)
        # up samples
        u_outs = []
        for d, layer in zip(d_outs, self.up_layers):
            x = layer(x, [d])
        return x


class MultiEncoder(nn.Module):
    def __init__(self,
                 feature_size=64,
                 resize=120,
                 pad=4,
                 ):
        super().__init__()
        self.resize = resize

        self.denoise = Encoder(
            in_ch=2,
            out_ch=feature_size * 2 ** 3,
            feature_size=feature_size,
        )

        self.rgb_enc = Encoder(
            in_ch=1,
            out_ch=3,
            feature_size=feature_size,
        )
        self.fusion_enc = Encoder(
            in_ch=1,
            out_ch=3,
            feature_size=feature_size,
        )

        self.out = Fc(
            in_ch=feature_size * 2 ** 3 + 6,
            out_ch=2
        )

        self.pad = nn.ZeroPad2d(pad)

    def forward(self, b_x, a_x):
        b_x = F.interpolate(b_x, mode='bilinear',
                            size=(self.resize, self.resize))
        a_x = F.interpolate(a_x, mode='bilinear',
                            size=(self.resize, self.resize))
        b_x = self.pad(b_x)
        a_x = self.pad(a_x)

        p_before = self.denoise_enc(x)
        p_after = self.denoise_enc(x)

        b_rgb = self.rgb_enc(b_x)
        a_rgb = self.rgb_enc(a_x)

        x = torch.cat([x, b_rgb, a_rgb, p_before], dim=1)
        self.out(x)
        x = self.out(x).view(-1, 2)
        b_rgb = F.interpolate(b_rgb, mode='bilinear', size=(4, 4))
        a_rgb = F.interpolate(a_rgb, mode='bilinear', size=(4, 4))
        return x, p_before, p_after, l_before, l_after


class AE(nn.Module):
    def __init__(self,
                 in_size=(2, 40, 40),
                 out_size=(2, 40, 40),
                 center_out_size=(6, 4, 4),
                 feature_size=64,
                 resize=120,
                 pad=4,
                 ):
        super().__init__()
        self.resize = resize
        self.in_size = in_size
        self.out_size = out_size
        self.center_out_size = center_out_size
        self.down_layers = nn.ModuleList([
            DownSample(in_size[0], feature_size),
            DownSample(feature_size, feature_size * 2 ** 1),
            DownSample(feature_size * 2 ** 1, feature_size * 2 ** 2),
            DownSample(feature_size * 2 ** 2, feature_size * 2 ** 3),
        ])

        self.center = DownSample(
            in_ch=feature_size * 2 ** 3,
            out_ch=feature_size * 2 ** 3,
        )

        self.center_out = nn.Conv2d(
            feature_size * 2 ** 3,
            center_out_size[0],
            kernel_size=3
        )

        self.up_layers = nn.ModuleList([
            UpSample(
                in_ch=feature_size * 16,
                out_ch=feature_size,
            ),
            UpSample(
                in_ch=feature_size * 13,
                out_ch=feature_size,
            ),
            UpSample(
                in_ch=feature_size * 7,
                out_ch=feature_size
            ),
            UpSample(
                in_ch=feature_size * 4,
                out_ch=feature_size
            ),
        ])
        self._output = nn.Conv2d(
            feature_size + center_out_size[0],
            out_size[0],
            kernel_size=3
        )
        self.pad = nn.ZeroPad2d(pad)

    def forward(self, x):
        x = F.interpolate(
            x,
            mode='bilinear',
            size=(self.resize, self.resize)
        )
        x = self.pad(x)

        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)

        _, x = self.center(x)
        center = self.center_out(x)
        center = F.interpolate(
            center,
            mode='bilinear',
            size=(self.center_out_size[1], self.center_out_size[2])
        )

        d_outs = list(reversed(d_outs))

        # up samples
        u_outs = []
        for i, layer in enumerate(self.up_layers):
            x = layer(x, d_outs[:i+1][-2:])

        x = torch.cat(
            [
                x,
                F.interpolate(
                    center,
                    size=x.size()[2:],
                    mode='bilinear',
                )
            ],
            dim=1
        )
        x = self._output(x)
        x = F.interpolate(
            x,
            mode='bilinear',
            size=(self.out_size[1], self.out_size[2]),
        )
        return x, center
