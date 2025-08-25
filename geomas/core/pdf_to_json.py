import os
import re
import unicodedata
import json
import subprocess
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
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
            current_chunk += " " + sentence_with_dot if current_chunk else sentence_with_dot

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
    text = text.replace("«", "\"").replace("»", "\"")

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


def process_folder(folder_path, output_folder="output_json", max_chunk_size=1200):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext == ".pdf":
                    text = pdf_to_text(file_path)
                elif ext == ".djvu":
                    text = djvu_to_text(file_path)
                elif ext in (".html", ".htm"):
                    text = html_to_text(file_path)
                else:
                    logger.info(f"Skipping {filename} (unsupported format)")
                    continue

                text = clean_text(text)
                chunks = chunk_text_by_sentences(text, max_chunk_size=max_chunk_size)

                # save JSON into the same folder structure
                rel_path = os.path.relpath(root, folder_path)
                out_dir = os.path.join(output_folder, rel_path)
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(out_dir, json_filename)

                save_to_json(chunks, json_path)
                logger.info(f"✅ {file_path} → {json_path} ({len(chunks)} chunks)")

            except Exception as e:
                logger.info(f"❌ Processing error {file_path}: {e}")


if __name__ == "__main__":
    input_folder = "geosources"
    process_folder(input_folder)
