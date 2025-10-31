import json
import os
import re
import subprocess
import unicodedata

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from geomas.core.logger import get_logger

logger = get_logger()


def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return " ".join(text.split())


def djvu_to_text(djvu_path):
    txt_path = djvu_path + ".txt"
    subprocess.run(["djvutxt", djvu_path, txt_path], check=True)
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    os.remove(txt_path)
    return " ".join(text.split())


def html_to_text(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def txt_to_text(txt_path):
    with open(txt_path, 'r') as f:
        data = f.read()

    return data


def chunk_text_by_sentences(text, max_chunk_size=1200):
    sentences = text.split(".")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_with_dot = sentence + "."
        if len(current_chunk) + len(sentence_with_dot) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence_with_dot
        else:
            current_chunk += (
                " " + sentence_with_dot if current_chunk else sentence_with_dot
            )

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def clean_text(text: str) -> str:
    """Cleans and normalizes text after extraction"""

    # 1. Gluing word hyphenation
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # 2. Replace multiple spaces into single space
    text = re.sub(r"\s+", " ", text)

    # 3. Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # 4. Remove spaces after brackets
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # 5. Bring quotes to the same type
    text = text.replace("«", '"').replace("»", '"')

    # 6. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 7. Clean up extra spaces at the edges
    text = text.strip()

    return text


def save_to_json(chunks, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data = [{"text": chunk} for chunk in chunks if chunk]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def call_ollama(txt: str) -> dict:
    import requests
    system = """
    Ты — детерминированный «Text Cleaner». Задача — НИКАКИХ домыслов.

    1) Не добавляй и не удаляй фактическое содержание, только чисть формат.
    2) Восстанавливай разрывы слов и строк, удаляй колонтитулы/номера страниц/линии-разделители.
    3) Нормализуй юникод (NFKC), замены: “smart quotes”→", ‘ ’→', ligatures (ﬁ→fi), неразрывные пробелы→обычные.
    4) Убери переносы по дефису на границе строк: "микро-\nструктура"→"микроструктура".
    5) Сохрани списки и заголовки как Markdown (#, ##, -, 1.), таблицы — как pipe-таблицы без графики.
    6) Сноски/цитаты вида [1], (1) — убери из текста; колонтитулы/номера страниц/скан-метки — удалить.
    7) Математику и формулы не переписывать: оставь как есть, LaTeX не расширять.
    8) Язык источника сохраняй (рус/англ и т.п.).
    9) Собирать слова, разбитые на буквы воедино.
    10) Убирать то, что является подписью к рисунку.
    11) Убрать ссылки на рисунки, главы, разделы, авторов, литературные источники
    12) Удалить перенос строки в середине предложения.
    13) Удалить ссылки на ученых и их работы.
    14) Удалить текст заголовков.

    Примеры явных ошибок в тексте которые нужно исправить:
    *Бессвязная последовательность*
    – 2 10\"5 Li 1,5-2 10 Sr 8-Ю\"4 Be 610\"11 Y 3 10\”

    *Заголовок среди текста*
    – 30 Интерпретация геохимических данных Можно рассчитать
    – По этой причине при низкой Глава 1 • Базовые понятия и определения в геохимии 31 
    – осадочных пород. 1. 4. 3. Факторы, определяющие геохимическую специфику метаморфических пород Определяющую роль в геохимических особенностях


    *Ненужная последовательность чисел*
    – 3,060 0,071 0,042 0,081 1,830 0,012 1,090 0,230 0,023 0,170

    *Ссылки на литературные источники*
    – для андезитовых расплавов [220] Роговая обманка [144; 214]

    *Разделение слова на буквы*
    – pea к ци и гидрата ц и и /де гидрата ц и и

    *Подпись рисунка*
    – Изохимические изменения 1ит 2 | | Протолит | Рис. 1. 12. Диаграмма, иллюстрирующая основные процессы,

    *Лишние символы*
    – окончательном результате. * * * 44 Интерпретация

    *Ссылка на главу в книге*
    – данных Подводя итог всему изложенному в гл. 1,

    *Большие буквы*
    – ИСПОЛЬЗОВАНИЕ ГЕОХИМИЧЕСКИХ ДАННЫХ ПРИ ИЗУЧЕНИИ МАГМАТИЧЕСКИХ ПОРОД

    *ошибка в обозначении химического вещества*
    – (Na20 + К^О) - Si02

    *Ссылка на рисунок*
    – Линия на диаграмме (рис. 2. 4) разделяет породы

    *ссылка на автора*
    - континентальных окраинах (Таусон, 1977) 

    В качестве ответа верни ТОЛЬКО ИСПРАВЛЕННЫЙ ТЕКСТ 
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "pdf-cleaner"

    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": txt, "stream": False, "options": {"seed": 42}})
    r.raise_for_status()
    out = r.json()["response"].strip()
    return out.strip('`')


def clean_chunks(chunks):
    from tqdm import tqdm

    cleaned_chunks = []
    
    with tqdm(total=len(chunks), desc='Cleaning chunks') as pbar:
        for chunk in chunks :
            cleaned_chunk = call_ollama(chunk)
            if 'NOTHING' in cleaned_chunk:
                cleaned_chunk = ""
            cleaned_chunks.append(cleaned_chunk)
            pbar.update(1)
    return cleaned_chunks

def process_folder(folder_path, output_folder="output_json", max_chunk_size=1200):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            expected_json = os.path.join(
                output_folder,                                  # root 
                os.path.relpath(root, folder_path),             # subfolder
                file_path.split('/')[-1].split('.')[0]+'.json'  # filename.json
                )
            # check existance
            if os.path.exists(expected_json):
                # check content
                with open(expected_json, 'r') as f:
                    data = json.load(f)
                if len(data) > 0:
                    logger.info(f"❗ File {file_path} already JSONed. Pass.")
                    continue
                else:
                    logger.info(f"🥲 File {file_path} already JSONed but empty. Replacing.")

            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext == ".pdf":
                    text = pdf_to_text(file_path)
                elif ext == ".djvu":
                    text = djvu_to_text(file_path)
                elif ext in (".html", ".htm"):
                    text = html_to_text(file_path)
                elif ext == ".txt":
                    text = txt_to_text(file_path)
                else:
                    logger.info(f"Skipping {filename} (unsupported format)")
                    continue

                text = clean_text(text)
                chunks = chunk_text_by_sentences(text, max_chunk_size=max_chunk_size)

                cleaned_chunks = clean_chunks(chunks)

                if len(cleaned_chunks) == 0:
                    raise ValueError('Nothing extracted from source')

                # save JSON into the same folder structure
                rel_path = os.path.relpath(root, folder_path)
                out_dir = os.path.join(output_folder, rel_path)
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(out_dir, json_filename)

                save_to_json(cleaned_chunks, json_path)
                # save_to_json(chunks, json_path)
                logger.info(f"✅ {file_path} → {json_path} ({len(cleaned_chunks)} chunks)")
                # logger.info(f"✅ {file_path} → {json_path} ({len(chunks)} chunks)")

            except Exception as e:
                logger.info(f"❌ Processing error {file_path}: {e}")


if __name__ == "__main__":
    # input_folder = "/app/one_source"
    process_folder(folder_path='/app/geo_sources_stage_1_cutted',
                   output_folder='/app/geo_sources_stage_1_cutted_output'
                   )
