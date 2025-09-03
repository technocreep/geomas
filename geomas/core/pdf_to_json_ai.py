import json
import os
import pathlib
import re

import google.generativeai as genai
import pdfplumber
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from geomas.core.logger import get_logger
from geomas.core.utils import PROJECT_PATH

logger = get_logger("AI_PDF")

load_dotenv(dotenv_path=PROJECT_PATH + "/geomas/.env")
API_KEY = os.getenv("LITELLM_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")

if not API_KEY:
    raise ValueError("LITELLM_API_KEY not set in .env")
genai.configure(api_key=API_KEY)

OUTPUT_DIR = PROJECT_PATH + "pdf_ai_results"  # change if needed
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CHUNK_SIZE_CHARS = 1800  # usually 1200-2200 characters is OK for RAG
CHUNK_OVERLAP_CHARS = 10  # overlap between adjacent chunks
MAX_OUTPUT_TOKENS = 8192  # model response limit (adjust if needed)


def read_pdf_raw_text(path: str) -> str:
    """
    Extract 'raw' text from the page + add page markers
    to preserve the origin of fragments.
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                txt = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            except Exception:
                txt = ""
            pages.append(f"\n\n<<<PAGE {i}>>>\n{txt.strip()}\n")
    return "".join(pages).strip()


def normalize_whitespace(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Splits text into chunks of limited size (chunk_size in characters),
    without breaking words. Overlap is specified in words.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    words = text.split()
    chunks = []
    current_words = []
    current_len = 0  # длина текущего чанка в символах

    i = 0
    while i < len(words):
        w = words[i]
        w_len = len(w) + 1  # +1 за пробел

        if current_len + w_len <= chunk_size or not current_words:
            # добавляем слово в текущий чанк
            current_words.append(w)
            current_len += w_len
            i += 1
        else:
            # фиксируем чанк
            chunk_str = " ".join(current_words)
            chunks.append({"id": f"chunk_{len(chunks):05d}", "text": chunk_str})

            # формируем overlap для следующего чанка
            if overlap > 0 and len(current_words) > overlap:
                current_words = current_words[-overlap:]
            else:
                current_words = current_words[:]

            current_len = sum(len(w) + 1 for w in current_words)

    # последний чанк
    if current_words:
        chunk_str = " ".join(current_words)
        chunks.append({"id": f"chunk_{len(chunks):05d}", "text": chunk_str})

    return chunks


def build_extraction_prompt():
    """
    Model instruction: EXTRACT ABSOLUTELY EVERYTHING and format it beautifully and structurally.
    """
    return (
        "Ты — строгий экстрактор знаний из PDF. "
        "Задача: извлечь АБСОЛЮТНО ВСЕ сведения из документа и выдать их в структурированном Markdown. "
        "Требования к ответу:\n"
        "1) Сохраняй ВЕСЬ смысл: названия, статусы, лицензии, коды, координаты, единицы измерения, диапазоны, формулы, параметры, проценты, даты, годы.\n"
        "2) Таблицы — рендери в формате Markdown-таблиц (| col | ... |) без потерь; если таблица шире — разбей на несколько.\n"
        "3) Формулы — сохрани в LaTeX между $$ ... $$ или $ ... $.\n"
        "4) Если тебе попадётся список, то его нужно переделать в обыкновенное предложение (не должно быть никаких списков, даже нумерованных).\n"
        "5) Сохраняй оригинальные термины (рус/eng/латиница), не переводить и не перефразировать значения.\n"
        "6) Не указывай никаких источников.\n"
        "7) Не добавляй знаний извне, только то, что есть в документе. Если чего-то нет — не выдумывай.\n"
        "8) Не используй символы, которые выделяют заголовки (диезы). \n"
        "9) Не используй нотации, которые отвечают за перенос строк (\\n). \n"
        "10) Если есть возможность написать текст более красиво с сохранением всей информации, указанной в пункте 1, то делай это. \n"
        "11) В конце добавь раздел «Полезные извлечённые сущности (для индексации)»:\n"
        "   - Ключевые параметры/поля: название, тип, локации, геология, минералогия, запасы/ресурсы, технологии, числа, диапазоны, единицы.\n"
        "   - Каждый пункт — в формате key: value.\n"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def gemini_extract_everything(pdf_path: str, raw_text: str) -> str:
    """
    Upload PDF as a file (so the model sees the structure), plus provide parsed 'raw' text.
    Use streaming generation to get a long response.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    # 1) Upload file to Gemini
    file_ref = genai.upload_file(pdf_path)  # returns File for multimodal context

    # 2) Build the query
    prompt = build_extraction_prompt()

    # We pass both PDF and raw text — so the model can stitch context if PDF has complex text layer
    parts = [
        file_ref,
        "\n---\nСырые извлечения из PDF (текстовые):\n",
        raw_text[
            :400000
        ],  # safety-ограничение: если очень длинно, обрежем контекст (на саму выжимку это не влияет)
        "\n---\nИнструкция по представлению результата:\n",
        prompt,
    ]

    # 3) Генерируем потоково
    resp = model.generate_content(
        parts,
        generation_config={
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.1,
        },
        safety_settings=None,
        stream=True,
    )

    collected = []
    for chunk in resp:
        if hasattr(chunk, "text") and chunk.text:
            collected.append(chunk.text)

    return "".join(collected).strip()


def save_markdown(md_text: str, source_pdf: str) -> str:
    base = pathlib.Path(source_pdf).with_suffix("").name
    out_md = f"{OUTPUT_DIR}/{base}__extracted.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    return out_md


def save_chunks_jsonl(chunks, source_pdf: str) -> str:
    base = pathlib.Path(source_pdf).with_suffix("").name
    out_jsonl = f"{OUTPUT_DIR}/{base}__chunks.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in chunks:
            rec = {
                "id": c["id"],
                "text": c["text"],
                "meta": {
                    "source": pathlib.Path(source_pdf).name,
                    "start": c["start"],
                    "end": c["end"],
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_jsonl


def save_chunks_json(chunks, source_pdf: str) -> str:
    base = pathlib.Path(source_pdf).with_suffix("").name
    out_json = f"{OUTPUT_DIR}/{base}__chunks.json"
    data = [{"text": c["text"]} for c in chunks]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_json


def pdf_to_json_ai(
    pdf_path, chunk_size_chars=CHUNK_SIZE_CHARS, chunk_overlap_chars=CHUNK_OVERLAP_CHARS
):
    if not pathlib.Path(pdf_path).exists():
        logger.info(
            "Please specify correct path to PDF in PDF_PATH variable and restart the cell."
        )
    else:
        logger.info("Reading PDF locally...")
        raw = read_pdf_raw_text(pdf_path)
        raw = normalize_whitespace(raw)

        logger.info(
            "Sending to Gemini for full extraction... (this may take some time for long documents)"
        )
        md = gemini_extract_everything(pdf_path, raw)

        # Small cosmetic changes
        title = pathlib.Path(pdf_path).stem
        header = f"# Complete structured extraction: {title}\n\n"
        md_final = header + md

        md_path = save_markdown(md_final, pdf_path)
        logger.info(f"Markdown saved: {md_path}")

        logger.info("Splitting into chunks for RAG...")
        chunks = chunk_text(md_final, chunk_size_chars, chunk_overlap_chars)
        jsonl_path = save_chunks_json(chunks, pdf_path)
        logger.info(f"Chunks saved: {jsonl_path}")
        logger.info(f"Total chunks: {len(chunks)}")
