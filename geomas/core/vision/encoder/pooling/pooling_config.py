from geomas.core.vision.encoder.pooling.pooling import Pooling
from geomas.core.vision.encoder.pooling.mean_pooling import MeanPooling
from geomas.core.vision.encoder.pooling.cls_pooling import ClsPooling

_FACTORIES= {
    MeanPooling.code_name():MeanPooling,
    ClsPooling.code_name():ClsPooling
}


def create_pooling(name: str) -> Pooling:
    return _FACTORIES[name]()