import json
import os
import pathlib
import re
import time

import google.generativeai as genai
import pdfplumber
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

API_KEY = os.environ.get("GEMINI_API_KEY")
assert API_KEY and API_KEY.strip(), (
    "Please, put your GEMINI_API_KEY into environment variable."
)
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"
OUTPUT_DIR = ""  # если надо, меняйте
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CHUNK_SIZE_CHARS = 1800  # обычно 1200–2200 символов — ок для RAG
CHUNK_OVERLAP_CHARS = 10  # перекрытие между соседними чанками
MAX_OUTPUT_TOKENS = 8192  # ограничение ответа модели (адаптируйте при необходимости)


def read_pdf_raw_text(path: str) -> str:
    """
    Достаём «сырой» текст со страницы + проставляем маркеры страниц,
    чтобы не потерять происхождение фрагментов.
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
    Разрезает текст на чанки ограниченного размера (chunk_size в символах),
    не разрывая слова. Перекрытие задаётся в словах.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0")
    if overlap < 0:
        raise ValueError("overlap не может быть отрицательным")

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
    Инструкция модели: ВЫТАЩИТЬ АБСОЛЮТНО ВСЁ и оформить красиво и структурно.
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
    Загружаем PDF как файл (чтобы модель видела структуру), плюсом даём распарсенный «сырой» текст.
    Используем потоковую генерацию, чтобы забрать длинный ответ.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    # 1) Загружаем файл в Gemini
    file_ref = genai.upload_file(pdf_path)  # вернёт File для мультимодального контекста

    # 2) Собираем запрос
    prompt = build_extraction_prompt()

    # Мы передаём и PDF, и сырой текст — чтобы модель могла сшить контекст, если в PDF сложный слой текста
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
        print(
            "Укажите корректный путь к PDF в переменной PDF_PATH и перезапустите ячейку."
        )
    else:
        print("Читаю PDF локально…")
        raw = read_pdf_raw_text(pdf_path)
        raw = normalize_whitespace(raw)

        print(
            "Отправляю в Gemini на полную выжимку… (это может занять немного времени при длинном документе)"
        )
        md = gemini_extract_everything(pdf_path, raw)

        # Небольшая косметика
        title = pathlib.Path(pdf_path).stem
        header = f"# Полная структурированная выжимка: {title}\n\n"
        md_final = header + md

        md_path = save_markdown(md_final, pdf_path)
        print(f"Markdown сохранён: {md_path}")

        print("Режу на чанки для RAG…")
        chunks = chunk_text(md_final, chunk_size_chars, chunk_overlap_chars)
        jsonl_path = save_chunks_json(chunks, pdf_path)
        print(f"Чанки сохранены: {jsonl_path}")
        print(f"Всего чанков: {len(chunks)}")
