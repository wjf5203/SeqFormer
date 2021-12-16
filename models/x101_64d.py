from torchvision.models.resnet import _resnet, Bottleneck


def resnext101_64x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 64*4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet('resnext101_64x4d', Bottleneck, [3, 4, 23, 3],
                   False, progress, **kwargs)
