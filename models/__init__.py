from models.image_enhancement_model import ImageEnhancementModel

MODEL_REGISTRY = {
    'ImageEnhancementModel': ImageEnhancementModel,
}


def create_model(opt):
    model_type = opt['model_type']
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')
    return model_cls(opt)
