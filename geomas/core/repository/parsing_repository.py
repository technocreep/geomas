from dataclasses import dataclass
from geomas.core.repository.config_repository import ConfigTemplate


@dataclass
class DatabaseChunkingConfig(ConfigTemplate):
    """Params for finetune multimodal LLM using LoRa"""
    sum_chunk_num: bool = 15
    final_sum_chunk_num: bool = 3
    txt_chunk_num: bool = 15
    img_chunk_num: int = 2

class DataTypeLoaderConfig(ConfigTemplate):
    docx = 'docx'
    doc = 'doc'
    odt = 'odt'
    rtf = 'rtf'
    pdf = 'pdf'
    directory = 'directory'
    zip = 'zip'
    json = 'json'