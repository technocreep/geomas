import os
import re
import json
from collections import defaultdict

# ---------------- 原有提取函数 ----------------
def extract_figures_from_file1(file1_path):
    """从文件1提取 {fig_num: caption}"""
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


def extract_from_file2(file2_path, fig_cap_f1):
    with open(file2_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pages = content.split('<--- Page Split --->')
    all_images = []
    all_captions = []
    all_texts = []

    for page in pages:
        images = re.findall(r'\(images/(.*?)\)', page)
        all_images.extend(images)
        caps = re.findall(r'<center>([Рр]ис\.\s*.*?\s*)</center>', page)
        all_captions.extend(caps)

        clean = re.sub(r'!\[\]\(images/[^)]+\.jpg\)', '', page)
        clean = re.sub(r'<center>.*?</center>', '', clean, flags=re.DOTALL)
        clean = clean.strip()
        if clean:
            all_texts.append(clean)

    full_text = ' '.join(all_texts)
    fig_to_context = {}
    for idx in fig_cap_f1:
        pattern = rf'[^.!?]*[Рр]ис\.?\s*{re.escape(str(idx))}[^.!?]*[.!?]'
        matches = re.findall(pattern, full_text)
        cleaned_matches = [m.strip() for m in matches if len(m.strip()) > 10]
        fig_to_context[idx] = cleaned_matches

    fig_data = {}
    for img, cap in zip(all_images, all_captions):
        m = re.search(r'[Рр]ис\.\s*(\d+(?:\.\d+)?)', cap, re.IGNORECASE)
        if m:
            fig_num = m.group(1)
            fig_data[fig_num] = {
                'image': f"images/{img}",
                'caption': cap,
                'context': fig_to_context.get(fig_num, "")
            }

    for cap in all_captions[len(all_images):]:
        m = re.search(r'Рис\.\s*(\d+\.\d+)', cap, re.IGNORECASE)
        if m:
            fig_num = m.group(1)
            if fig_num not in fig_data:
                fig_data[fig_num] = {
                    'image': "",
                    'caption': cap,
                    'context': fig_to_context.get(fig_num, "")
                }

    return fig_data, fig_to_context


def main(file1_path, file2_path, output_path):
    fig_cap_f1 = extract_figures_from_file1(file1_path)
    fig_data, _ = extract_from_file2(file2_path, fig_cap_f1)

    for fig_num, data in fig_data.items():
        if not data['caption'].strip():
            data['caption'] = fig_cap_f1.get(fig_num, "")

    result = []
    for fig_num in sorted(fig_data.keys(), key=lambda x: tuple(map(int, x.split('.')))):
        item = fig_data[fig_num]
        result.append({
            "image": item["image"],
            "image_caption": item["caption"],
            "context": item["context"]
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 处理完成: {output_path} ({len(result)} 条)")


# ---------------- 遍历所有子文件夹 ----------------
def process_all_folders(base_dir):
    for root, dirs, files in os.walk(base_dir):
        det_files = [f for f in files if f.endswith("_det.mmd")]
        for det_file in det_files:
            prefix = det_file[:-8]  # 去掉 "_det.mmd"
            mmd_file = f"{prefix}.mmd"
            det_path = os.path.join(root, det_file)
            mmd_path = os.path.join(root, mmd_file)

            if os.path.exists(mmd_path):
                out_path = os.path.join(root, f"{prefix}.json")
                print(f"🚀 正在处理: {prefix}")
                try:
                    main(det_path, mmd_path, out_path)
                except Exception as e:
                    print(f"❌ 处理失败 {prefix}: {e}")
            else:
                print(f"⚠️ 未找到匹配的 {mmd_file} 对应 {det_file}")


# ---------------- 入口 ----------------
if __name__ == "__main__":
    # base_folder = input("请输入主文件夹路径: ").strip()
    base_folder = "/home/hyl/vlm/geomas/geo_sources_stage_1_ocr_result"
    process_all_folders(base_folder)
