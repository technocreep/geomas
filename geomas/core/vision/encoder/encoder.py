import abc
from typing import List
import numpy


class Encoder(abc.ABC):
    @staticmethod
    def code_name() -> str:
        raise NotImplementedError("abstract")

    def encode(self, Samples: List[str]) -> numpy.ndarray:
        raise NotImplementedError("abstract")
