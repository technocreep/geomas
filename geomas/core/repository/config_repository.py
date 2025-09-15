from dataclasses import dataclass
from enum import Enum
from functools import reduce
from inspect import signature, isclass
from typing import (
    get_origin, get_args,
    Any, Callable, Dict, Iterable, Literal, Optional, Union,
)
import logging

__all__ = [
    'ConfigTemplate',
    'PeftConfigTemplate',
    'ModelConfigTemplate',
    'UnslothTrainingTemplate',
    'get_nested',
    'LookUp',
    'LookUp',
]


def get_nested(root: object, k: str):
    """Func to allow subcrtiption like config['x.y.x']"""
    *path, last = k.split('.')
    return reduce(getattr, path, root), last


class MisconfigurationError(BaseException):
    def __init__(self, exs, *args):
        super().__init__(*args)
        self.exs = exs

    def __repr__(self):
        return '\n'.join([f'\t{str(x)}' for x in self.exs])

    def __str__(self):
        return self.__repr__()


@dataclass
class LookUp:
    """Wrapping telling to fill it from parental config if not specified"""
    value: Any


@dataclass
class ConfigTemplate:
    """Fixed Structure Config"""
    __slots__ = tuple()

    @classmethod
    def get_default_name(cls):
        name = cls.__name__.split('.')[-1]
        if name.endswith('Template'):
            name = name[:-8]
        return name

    @classmethod
    def get_annotation(cls, key):
        obj = cls if ConfigTemplate in cls.__bases__ else cls.__bases__[0]
        return signature(obj.__init__).parameters[key].annotation

    @classmethod
    def check(cls, key, val):
        # we don't check parental attr
        if key == '_parent':
            return

        def _check_primal(annotation, key, val):
            if isclass(annotation):
                if issubclass(annotation, Enum):
                    if val is not None and not hasattr(annotation, val):
                        return ValueError(
                            f'`{val}` not supported as {key} at config {cls.__name__}. Options: {annotation._member_names_}')
                elif not isinstance(val, annotation):
                    return TypeError(f'`Passed `{val}` at config: {cls.__name__}, field: {key}. Expected: {annotation}')
            elif annotation is Callable and not hasattr(val, '__call__'):
                return TypeError(f'`Passed `{val}` at config: {cls.__name__}, field: {key}, is not callable!')
            elif get_origin(annotation) is Literal and not val in get_args(annotation):
                return ValueError(f'Passed value `{val}` at config {cls.__name__}. Supported: {get_args(annotation)}')
            return False

        def _check(annotation, key, val):
            options = get_args(annotation) or (annotation,)
            exs = [_check_primal(option, key, val)
                   for option in options]
            if exs and all(exs):
                raise MisconfigurationError(exs)

        annotation = cls.get_annotation(key)
        _check(annotation, key, val)

    def __new__(cls, *args, **kwargs):
        """We don't need template instances themselves. Only normalized parameters"""
        allowed_parameters = _normalize_kwargs(cls.__init__, kwargs)
        for k in kwargs:
            if k not in allowed_parameters:
                raise KeyError(f'Unknown field `{k}` was passed into {cls.__name__}')
        sign_args = tuple(signature(cls.__init__).parameters)
        complemented_args = dict(zip(sign_args[1:],
                                     args))
        allowed_parameters.update(complemented_args)
        return cls, allowed_parameters

    def __repr__(self):
        params_str = '\n'.join(
            f'{k}: {getattr(self, k)}' for k in self.__slots__
        )
        return f'{self.get_default_name()}: \n{params_str}\n'

    def get_parent(self):
        return getattr(self, '_parent')

    def update(self, d: dict):
        for k, v in d.items():
            obj, attr = get_nested(self, k)
            obj.__setattr__(attr, v)

    def get(self, key, default=None):
        return getattr(*get_nested(self, key), default)

    def keys(self) -> Iterable:
        return tuple(slot for slot in self.__slots__ if slot != '_parent')

    def items(self) -> Iterable:
        return (
            (k, self[k]) for k in self.keys()
        )

    def to_dict(self) -> dict:
        ret = {}
        for k, v in self.items():
            if hasattr(v, 'to_dict'):
                v = v.to_dict()
            ret[k] = v
        return ret

    @property
    def config(self):
        return self


class ExtendableConfigTemplate(ConfigTemplate):
    """Allows to dynamically add attributes.
    Warning: check behaviour of keys with newly added ones"""

    @classmethod
    def check(cls, key, val):
        if not key in cls.__slots__:
            return
        super().check(key, val)

    def keys(self):
        return [*tuple(slot for slot in self.__slots__ if slot != '_parent'), *list(self.__dict__)]


@dataclass
class ModelConfigTemplate(ConfigTemplate):
    """Training device specification."""
    max_seq_length: int = 2048
    dtype: Literal['int'] = None
    load_in_4bit: bool = False


@dataclass
class UnslothTrainingTemplate(ConfigTemplate):
    """Unsloth learning process params"""
    per_device_train_batch_size: int = 10
    gradient_accumulation_steps: int = 10
    warmup_ratio: float = 0.1
    num_train_epochs: int = 10
    learning_rate: float = 0.0005  # 5e-5
    embedding_learning_rate: float = 0.00005  # 5e-6
    logging_steps: int = 1
    optim: Literal['adamw_8bit'] = "adamw_8bit"
    weight_decay: float = 0.0
    lr_scheduler_type: Literal['cosine'] = "cosine"
    seed: int = 3407
    save_strategy: Literal['epoch'] = 'epoch'


@dataclass
class PeftConfigTemplate(ConfigTemplate):
    """Params for finetune using LoRa"""
    r: int = 32
    target_modules: Literal["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: Literal["none"] = "none"
    use_gradient_checkpointing: Literal["unsloth"] = "unsloth"
    random_state: int = 3407
    use_rslora: bool = True
    loftq_config: Literal["none"] = None


@dataclass
class PeftMuiltimodalConfigTemplate(ConfigTemplate):
    """Params for finetune multimodal LLM using LoRa"""
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
