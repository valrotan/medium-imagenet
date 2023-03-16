from .convnext import ConvNext18, ConvNext26, ConvNext38
from .lenet import LeNet
from .resnet import ResNet18


def build_model(config):
    "Model builder."

    model_type = config.MODEL.NAME

    if model_type == 'lenet':
        model = LeNet(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'resnet18':
        model = ResNet18(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'convnext18':
        model = ConvNext18(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE)
    elif model_type == 'convnext26':
        model = ConvNext26(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE)
    elif model_type == 'convnext38':
        model = ConvNext38(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    
    return model
 