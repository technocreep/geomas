from geomas.core.vision.encoder.auto_encoder import AutoModelEncoder
from geomas.core.vision.encoder.clip_encoder import ClipEncoder
from geomas.core.vision.encoder.sentencetransformer_encoder import SentencetransformerEncoder
from geomas.core.vision.encoder.api_encoder import ApiEncoder

_FACTORIES = {
    ApiEncoder.code_name(): ApiEncoder,
    AutoModelEncoder.code_name(): AutoModelEncoder,
    ClipEncoder.code_name(): ClipEncoder,
    SentencetransformerEncoder.code_name(): SentencetransformerEncoder,
}


def create_encode(encoder_type, model_name, **kwargs):
    encoder_class = _FACTORIES[encoder_type]
    encoder_instance = encoder_class(model_name, **kwargs)
    return encoder_instance
