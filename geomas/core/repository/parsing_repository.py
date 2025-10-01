from dataclasses import dataclass
from geomas.core.repository.config_repository import ConfigTemplate


class DatabaseChunkingConfig(ConfigTemplate):
    """Params for finetune multimodal LLM using LoRa"""
    sum_chunk_num: bool = 15
    final_sum_chunk_num: bool = 3
    txt_chunk_num: bool = 15
    img_chunk_num: int = 2


class ChunkingParamsConfig(ConfigTemplate):
    """"""
    max_chunk_size: int = 2500
    chunk_overlap: int = 200
    separators: list = ["\n\n", "\n", ". "]
    headers_to_split_on: list = [("h1", "Header 1"), ("h2", "Header 2")]
    elements_to_preserve: list = ["ul", "table", "ol"]
    preserve_images: bool = True


class DataTypeLoaderConfig(ConfigTemplate):
    docx = 'docx'
    doc = 'doc'
    odt = 'odt'
    rtf = 'rtf'
    pdf = 'pdf'
    directory = 'directory'
    zip = 'zip'
    json = 'json'


class ParsingPatternConfig(ConfigTemplate):
    image_pattern = r'!\[image:([^\]]+\.jpeg)\]\(([^)]+\.jpeg)\)'
    table_pattern = r'<table\b[^>]*>.*?</table>'
    ignored_topics = [
        "author information", "associated content", "acknowledgment", "acknowledgement", "acknowledgments",
        "acknowledgements", "references", "data availability", "declaration of competing interest",
        "credit authorship contribution statement", "funding", "ethical statements", "supplementary materials",
        "conflict of interest", "conflicts of interest", "author contributions", "data availability statement",
        "ethics approval", "supplementary information"
    ]
