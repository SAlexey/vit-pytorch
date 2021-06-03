import torch
from torch import nn
from torch.tensor import Tensor
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import einops as E


def conv3x3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3D(BasicBlock):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


class Bottleneck3D(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet18_3d(*, block=BasicBlock3D, norm_layer=nn.BatchNorm3d, **kwargs) -> ResNet:
    return ResNet3D(block, [2, 2, 2, 2], norm_layer=norm_layer, **kwargs)


def resnet50_3d(*, block=Bottleneck3D, norm_layer=nn.BatchNorm3d, **kwargs) -> ResNet:
    return ResNet3D(block, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)


class ResNet3D(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_bbox = MLP(2048, 2048, 12, 3)
        self.linear_class = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: BasicBlock3D,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x -> (BS, 1, 160, 300, 300)
        x = self.conv1(x)  # x -> (BS, 64, 80, 150, 150)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # x -> (BS, 64, 40, 75, 75)

        x = self.layer1(x)  # x -> (BS, 128, 20, 38, 38)
        x = self.layer2(x)  # x -> (BS, 256, 10, 19, 19)
        x = self.layer3(x)  # x -> (BS, )
        x = self.layer4(x)  # x -> (BS, 512, 5, 10, 10)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


CONFIG = {"resnet18_3d": (resnet18_3d, 512), "resnet50_3d": (resnet50_3d, 2048)}


class Net1(nn.Module):
    """
    Detects menisci and classifies them in a binary manner (healthy vs diseased)

    Outputs: a dictionary with keys
        - boxes: tensor[12] box coordinates in cxcywh format
        - labels: tebsor[2] probabilities of diseased menisci
    """

    def __init__(self, backbone: str, *args, **kwargs):
        assert backbone in CONFIG, f"unknown backbone: {backbone}"
        super().__init__()

        init, num_channels = CONFIG[backbone]
        self.backbone = init(*args, **kwargs)

        self.out_boxes = MLP(num_channels, 2048, 12, 3)
        self.out_labels = nn.Linear(num_channels, 2)

    def forward(self, x):
        feat = self.backbone(x)
        boxes = self.out_boxes(feat).sigmoid()
        labels = self.out_labels(feat)

        return {"boxes": boxes, "labels": labels}


class Net2(nn.Module):
    """
    Detects menisci and classifies them in a multilabel & multiclass manner (MOAKS)

    Outputs: a dictionary with keys
        - boxes: tensor[12] box coordinates in cxcywh format
    """

    def __init__(self, backbone: str, *args, **kwargs):
        assert backbone in CONFIG, f"unknown backbone: {backbone}"
        super().__init__()

        init, num_channels = CONFIG[backbone]
        self.backbone = init(*args, **kwargs)

        self._num_objects = 2  # number of menisci (med, lat)
        self._num_labels = 3  # number of parts permeniscus (A.Horn, Body, P.Horn)
        self._num_classes = 1  # number of classes per part (MOAKS)

        self.out_boxes = MLP(num_channels, 2048, 12, 3)
        self.out_labels = MLP(
            num_channels,
            2048,
            self._num_objects * self._num_labels,
            3,
        )

    def forward(self, x):
        x = self.backbone(x)

        labels = E.rearrange(
            self.out_labels(x),
            "bs (objs ls) ->bs objs ls",
            objs=self._num_objects,
            ls=self._num_labels,
        )

        boxes = E.rearrange(
            self.out_boxes(x).sigmoid(),
            "bs (boxes coords) -> bs boxes coords",
            boxes=self._num_objects,
        )

        return {"boxes": boxes, "labels": labels}


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == "__main__":

    model = resnet18_3d()

    print(model)

    input = torch.randn(1, 1, 160, 384, 384)

    output = model(input)
    print(output.shape)
