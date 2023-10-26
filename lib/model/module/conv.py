from torch import nn, Tensor
from typing import Optional, Tuple, Type
import torch
from tensordict import tensorclass


@tensorclass
class UNetEncoderOutput:
    latent: Tensor
    skips: Optional[Tuple[Tensor, ...]] = None


class ConvINReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = False,
        norm_type: Type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn = norm_type(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvINReLU(in_channels, out_channels, 3, stride, 1)
        self.conv2 = ConvINReLU(out_channels, out_channels, 3, 1, 1)
        self.rescale = (
            ConvINReLU(in_channels, out_channels, 1, stride)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.rescale is not None:
            identity = self.rescale(identity)
        x = x + identity
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.residual = Residual(in_channels, out_channels, stride)
        self.down = nn.Conv2d(out_channels, out_channels, 1, stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.residual(x)
        x = self.down(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 1, stride, bias=False)
        self.residual = Residual(out_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.residual(x)
        return x


class SkipUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels * 2, out_channels, 1, stride, bias=False
        )
        self.residual = Residual(in_channels, out_channels, stride)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.up(x)
        x = self.residual(x)
        return x


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        norm_type: Type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.residuals = nn.ModuleList(
            [
                Residual(in_channels, out_channels, stride)
                if i == 0
                else Residual(out_channels, out_channels)
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for residual in self.residuals:
            x = residual(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        depth: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.in_conv = nn.Conv2d(in_channels, base_channels, 1, bias=False)
        self.downsamples = nn.ModuleList(
            [
                Downsample(in_channels, base_channels),
                *[
                    Downsample(base_channels * 2**i, base_channels * 2 ** (i + 1))
                    for i in range(depth - 1)
                ],
            ]
        )
        self.out_channels = (
            base_channels * 2**depth if out_channels is None else out_channels
        )
        self.out_norm = nn.InstanceNorm2d(base_channels * 2**depth)
        self.out_conv = nn.Conv2d(
            base_channels * 2**depth, self.out_channels, 1, bias=False
        )

    def forward(self, x: Tensor, skip_connections: bool = False) -> Tensor:
        x = self.in_conv(x)
        skips = []
        for downsample in self.downsamples:
            skips.append(x)
            x = downsample(x)
        x = self.out_norm(x)
        x = self.out_conv(x)
        if skip_connections:
            return UNetEncoderOutput(latent=x, skips=tuple(skips))


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        depth: int,
        out_channels: Optional[int] = None,
        skip_connections: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.skip_connections = skip_connections
        self.depth = depth
        self.in_conv = nn.Conv2d(in_channels, base_channels * 2**depth, 1, bias=False)
        down_cls = SkipUpsample if skip_connections else Upsample
        self.upsamples = nn.ModuleList(
            [
                down_cls(base_channels * 2**i, base_channels * 2 ** (i - 1))
                for i in range(depth, 0, -1)
            ]
        )
        self.out_channels = base_channels if out_channels is None else out_channels
        self.out_norm = nn.InstanceNorm2d(base_channels)
        self.out_conv = nn.Conv2d(base_channels, self.out_channels, 1, bias=False)

    def forward(
        self,
        inputs: Tensor | UNetEncoderOutput,
        skips: Optional[Tuple[Tensor, ...]] = None,
    ) -> Tensor:
        x = None
        if isinstance(inputs, UNetEncoderOutput):
            x = inputs.latent
            skips = inputs.skips
        else:
            x = inputs
        x = self.in_conv(x)
        for upsample, skip in zip(self.upsamples, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
        x = self.out_norm(x)
        x = self.out_conv(x)
        return x
