import os
import re
import json
from collections import defaultdict
from geomas.core.vision.generate_final_description import generate_final_description
from geomas.core.logging.logger import get_logger
logger = get_logger("VISUAL_DATA_PROCESSOR")

class TextProcessor:
    """
    Класс для извлечения данных о рисунках:
    - извлекает подписи из файла *_det.mmd
    - находит соответствующие изображения и подписи в *.mmd
    - собирает контекст вокруг ссылок на рисунки
    - сохраняет результат в JSON
    """
    def __init__(self, path):
        self.root_path = path

    
    def extract_from_detmmd(self, file1_path):
        """
        Извлечение подписей из det.mmd
        """
        with open(file1_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pages = content.split('<--- Page Split --->')
        fig_to_caption_f1 = {}

        for page in pages:
            lines = [line.strip() for line in page.split('\n') if line.strip()]
            i = 0
            while i < len(lines):
                line = lines[i]
                if '<|ref|>image<|/ref|>' in line:
                    if i + 1 < len(lines) and '<|ref|>image_caption<|/ref|>' in lines[i + 1]:
                        if i + 2 < len(lines):
                            cap_text = lines[i + 2]
                            match = re.search(r'<center>\s*(.*?)\s*</center>', cap_text, re.IGNORECASE | re.DOTALL)
                            if match:
                                m = re.sub(r'\s+', ' ', match.group(1)).strip()
                            else:
                                m = cap_text
                            if cap_text:
                                full_cap = m
                                nm = re.search(r'[Рр]ис\.\s*(\d+(?:\.\d+)?)', full_cap)
                                if nm:
                                    fig_num = nm.group(1)
                                    fig_to_caption_f1[fig_num] = full_cap
                            i += 2
                i += 1
        return fig_to_caption_f1
    

    def extract_from_mmd(self, file2_path, fig_cap_f1):
        """
        Извлечение изображений/подписей из *.mmd
        """
        with open(file2_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pages = content.split('<--- Page Split --->')

        pending_images = []
        fig_data = {}
        all_texts = []

        img_pattern = re.compile(r'!\[[^\]]*\]\(images/([^)]*)\)')
        cap_pattern = re.compile(r'<center>\s*([Рр]ис\.\s*\d+(?:\.\d+)?[^\n<]*)</center>')

        for page in pages:
            lines = page.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 1) image
                img_match = img_pattern.search(line)
                if img_match:
                    pending_images.append("images/" + img_match.group(1))
                    continue

                # 2) caption
                cap_match = cap_pattern.search(line)
                if cap_match:
                    cap_text = cap_match.group(1)
                    m = re.search(r'[Рр]ис\.?\s*(\d+(?:\.\d+)?)', cap_text)
                    if not m:
                        continue
                    fig_num = m.group(1)

                    # FIFO
                    image_path = pending_images.pop() if pending_images else ""

                    fig_data[fig_num] = {
                        "image": image_path,
                        "caption": cap_text.strip(),
                        "context": []
                    }
            clean = re.sub(img_pattern, '', page)
            clean = re.sub(r'<center>.*?</center>', '', clean, flags=re.DOTALL)
            clean = clean.strip()
            if clean:
                all_texts.append(clean)

        # 3) найти context
        full_text = "\n".join(all_texts)
        fig_to_context = defaultdict(list)
        for idx,caption in fig_cap_f1.items():
            pattern = rf'[^.!?]*[Рр]ис\.?\s*{re.escape(str(idx))}(?![\d\.])[^.!?]*[.!?]'
            matches = re.findall(pattern, full_text)
            candidates = [
                m.strip()
                for m in matches
                if ''.join(caption.split()) not in ''.join(m.split())
            ]
            fig_to_context[idx] = candidates
        for k in fig_data:
            fig_data[k]["context"] = fig_to_context.get(k, "")

        return fig_data, fig_to_context


   
    def process_pair(self, file1_path, file2_path, output_path, image_meta_map, root):
        """
        Обрабатывает пару (det.mmd, mmd) и сохраняет JSON.
        """
        # Подписи из det.mmd
        fig_cap_f1 = self.extract_from_detmmd(file1_path)

        # Извлечение изображений и контекста
        fig_data, _ = self.extract_from_mmd(file2_path, fig_cap_f1)

        # Заполняем отсутствующие подписи
        for fig_num, data in fig_data.items():
            if not data['caption'].strip():
                data['caption'] = fig_cap_f1.get(fig_num, "")

        # Сортировка по номеру рисунка
        result = []
        for fig_num in sorted(fig_data.keys(), key=lambda x: tuple(map(int, x.split('.')))):
            item = fig_data[fig_num]
            image_key=f"{root}/{item['image']}"
            meta = image_meta_map.get(image_key, {})

            result.append({
                "image": image_key,
                "image_caption": item["caption"],
                "context": item["context"],
                "description": meta.get("description", ""),
                "metadata": meta,
            })

        # Запись JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Готово: {output_path} ({len(result)} записей)")
        return output_path


    def build_image_metadata_map(self, base_dir):
        """
        Constuct { (relative_folder_path, image_name): metadata } mapping
        """
        img_meta = {}
        base_dir = os.path.abspath(base_dir)

        for root, dirs, files in os.walk(base_dir):
            if os.path.basename(root) == "images":
                for file in files:
                    if file.endswith(".json"):
                        path = os.path.join(root, file)
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                image_path = data.get("metadata", {}).get("image_path",{})
                                img_meta[image_path] = data.get("metadata", {})
                        except Exception as e:
                            logger.info(f"⚠️ Read image metadata failed: {path}: {e}")
        return img_meta




    def process_all_folders(self):
        """
        Находит все *_det.mmd и соответствующие *.mmd,
        затем обрабатывает все пары файлов.
        """
        image_meta_map = self.build_image_metadata_map(self.root_path)
        for root, dirs, files in os.walk(self.root_path):
            det_files = [f for f in files if f.endswith("_det.mmd")]
            logger.info(root)
            for det_file in det_files:
                prefix = det_file[:-8]  # удаляем "_det.mmd"
                mmd_file = f"{prefix}.mmd"
                det_path = os.path.join(root, det_file)
                mmd_path = os.path.join(root, mmd_file)

                if os.path.exists(mmd_path):
                    out_path = os.path.join(root, f"{prefix}.json")
                    logger.info(f"🚀 Обработка: {prefix}")
                    try:
                        output_json=self.process_pair(det_path, mmd_path, out_path, image_meta_map, root)
                        generate_final_description(output_json)
                    except Exception as e:
                        logger.info(f"❌ Ошибка при обработке {prefix}: {e}")
                else:
                    logger.info(f"⚠️ Не найден файл {mmd_file} для {det_file}")
                    
    

if __name__ == "__main__":
    base_folder = "/home/hyl/rag/vlm/geo_sources_stage_1_ocr_result"
    extractor = TextProcessor(base_folder)
    extractor.process_all_folders()
