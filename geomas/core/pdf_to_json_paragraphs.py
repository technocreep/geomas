import fitz
import numpy as np
import re
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict

BULLET_RE = re.compile(r'^[\u2022\-\*\–\—\•\(\[]\s*|^\d+[\.\)]\s*')  # буллет/номера


def _para_bbox(para):
    return tuple(para["bbox"])


def _para_x0(para): return para["bbox"][0]
def _para_x1(para): return para["bbox"][2]
def _para_y0(para): return para["bbox"][1]
def _para_y1(para): return para["bbox"][3]
def _para_width(para): return para["bbox"][2] - para["bbox"][0]


def merge_two_paragraphs(prev, cur):
    """Сливает два абзаца: prev <- prev + cur. Возвращает новый объединённый объект."""
    # конкатенация текста (с пробелом)
    new_text = (prev.get("text","").rstrip() + " " + cur.get("text","").lstrip()).strip()
    x0 = min(prev["bbox"][0], cur["bbox"][0])
    y0 = min(prev["bbox"][1], cur["bbox"][1])
    x1 = max(prev["bbox"][2], cur["bbox"][2])
    y1 = max(prev["bbox"][3], cur["bbox"][3])
    # num_lines аггрегируем, если есть
    na = int(prev.get("num_lines", 1))
    nb = int(cur.get("num_lines", 1))
    num_lines = na + nb
    # avg_font: взвешиваем по длине текста (если нет: усредняем)
    fa = float(prev.get("avg_font", 0.0))
    fb = float(cur.get("avg_font", 0.0))
    la = max(1, len(prev.get("text","")))
    lb = max(1, len(cur.get("text","")))
    avg_font = float((fa * la + fb * lb) / (la + lb))
    # page dimensions and page index
    page = prev.get("page", cur.get("page"))
    page_w = prev.get("page_width", cur.get("page_width"))
    page_h = prev.get("page_height", cur.get("page_height"))
    # label -> после слияния ставим Text
    merged = {
        "page": page,
        "text": new_text,
        "bbox": (x0, y0, x1, y1),
        "avg_font": avg_font,
        "num_lines": num_lines,
        "page_width": page_w,
        "page_height": page_h,
        "label": "Text"
    }
    return merged


def _should_merge(prev, cur,
                  median_line_h,
                  x_align_tol=12.0,
                  gap_multiplier=1.25,
                  max_single_line_len=140,
                  merge_width_ratio=0.85,
                  allow_merge_into_title=False):
    """
    Здесь условия, при которых считаем, что `cur` — это последняя строка предыдущего абзаца,
    и её можно аккуратно слить в prev.
    Возвращает True/False.
    """
    # требования: cur — короткий single-line chunk, не многословный
    if cur.get("num_lines", 1) > 1:
        return False

    # не сливаемНА Title (обычно title отдельно), если не разрешено
    prev_label = prev.get("label", "Text")
    if prev_label == "Title" and not allow_merge_into_title:
        return False

    # вертикальный gap (может быть отрицательным, если bbox немного пересекаются)
    gap_y = cur["bbox"][1] - prev["bbox"][3]  # cur.y0 - prev.y1

    # если gap слишком большой — не сливаем
    if gap_y > max( max(1.0, median_line_h * gap_multiplier), 6.0):
        return False

    # если gap существенно отрицательный (слишком большое перекрытие) — осторожно: не сливаем
    if gap_y < - (0.5 * median_line_h):
        # но это маловероятно; оставляем False, чтобы не сливать случайные оверлеи
        return False

    # выравнивание по левому/правому краю
    left_aligned = abs(cur["bbox"][0] - prev["bbox"][0]) <= x_align_tol
    right_aligned = abs(cur["bbox"][2] - prev["bbox"][2]) <= x_align_tol

    # ширины: если cur почти такой же ширины как prev — это континуэйшн
    cur_w = _para_width(cur)
    prev_w = _para_width(prev)
    width_similar = (cur_w >= prev_w * merge_width_ratio) or (prev_w >= cur_w * merge_width_ratio)

    # длина текста — если очень короткая (корректируем порог при необходимости)
    short_text = len(cur.get("text","")) <= max_single_line_len

    # Bullets / numbering — если cur начинается с буллета, это скорее новый параграф
    import re
    if re.match(r'^[\u2022\-\*\–\—\•\(\[]\s*|^\d+[\.\)]\s*', cur.get("text","").lstrip()):
        return False

    # Если хотя бы одно из выравниваний и ширина/длина дают совпадение -> merge
    if short_text and ( (left_aligned and gap_y <= max(median_line_h * 0.6, 4.0)) or right_aligned or width_similar):
        return True

    return False


def postprocess_paragraphs(paras,
                           x_align_tol=12.0,
                           gap_multiplier=1.25,
                           max_single_line_len=140,
                           merge_width_ratio=0.85,
                           allow_merge_into_title=False):
    """
    Постобработка списка абзацев:
      - сливает отдельные короткие single-line абзацы назад в предыдущий, если это похоже на continuation
      - конвертирует ошибочные Subtitle -> Text и затем сливает по тем же правилам
    Возвращает новый список абзацев (с перекомпоновкой индексов).
    """
    # группируем по странице
    pages = {}
    for p in paras:
        pages.setdefault(p["page"], []).append(p)

    out = []
    for pnum, lst in pages.items():
        if not lst:
            continue
        # сортировка сверху вниз
        lst.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
        # медианная высота строки на странице (приближенно)
        per_line_heights = []
        for p in lst:
            nh = max(1, p.get("num_lines", 1))
            per_line_heights.append((p["bbox"][3] - p["bbox"][1]) / nh)
        median_line_h = float(np.median(per_line_heights)) if per_line_heights else 10.0

        i = 0
        # будем модифицировать список in-place (pop/replace)
        while i < len(lst):
            if i == 0:
                i += 1
                continue
            prev = lst[i-1]
            cur = lst[i]
            # Если текущий помечен Subtitle, но выглядит как продолжение — перевести в Text и/или merge
            # (вначале проверяем возможность слияния)
            if _should_merge(prev, cur, median_line_h,
                             x_align_tol=x_align_tol,
                             gap_multiplier=gap_multiplier,
                             max_single_line_len=max_single_line_len,
                             merge_width_ratio=merge_width_ratio,
                             allow_merge_into_title=allow_merge_into_title):
                # merge cur into prev
                merged = merge_two_paragraphs(prev, cur)
                # заменяем prev и удаляем cur
                lst[i-1] = merged
                lst.pop(i)
                # не инкрементируем i (потому что новое prev может далее сливаться с следующим cur)
                continue
            else:
                # если cur помечен Subtitle, но не сливается — всё равно можно перевести в Text,
                # если он явно находится под Text и выровнен (без слияния).
                # Это полезно, чтобы не иметь Subtitle внутри абзаца.
                if cur.get("label") == "Subtitle":
                    # консервативный перевод в Text, если он выровнен и gap маленький,
                    # но не соответствует условиям слияния.
                    gap_y = cur["bbox"][1] - prev["bbox"][3]
                    if gap_y <= max(median_line_h * 0.6, 4.0) and abs(cur["bbox"][0] - prev["bbox"][0]) <= x_align_tol:
                        # только смена label, без слияния
                        cur["label"] = "Text"
                        lst[i] = cur
                i += 1

        # после прохода добавляем в out
        out.extend(lst)

    # опционально — можно снова упорядочить out по (page, y)
    out.sort(key=lambda c: (c["page"], c["bbox"][1], c["bbox"][0]))
    return out


# ---------------------------
# 1) извлечение строк со шрифтами
# ---------------------------
def extract_lines_with_fonts(pdf_path: str) -> List[Dict]:
    """
    Возвращает список строк: каждую строку как dict с bbox, text, avg_font, page, page_width/height.
    """
    doc = fitz.open(pdf_path)
    all_lines = []
    for pnum in range(len(doc)):
        page = doc[pnum]
        page_w, page_h = page.rect.width, page.rect.height
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                texts, sizes, x0s, y0s, x1s, y1s, char_counts = [], [], [], [], [], [], []
                for sp in spans:
                    txt = sp.get("text", "")
                    if not txt.strip():
                        continue
                    texts.append(txt)
                    sizes.append(sp.get("size", 0.0))
                    bbox = sp.get("bbox", None)
                    if bbox:
                        x0s.append(bbox[0]); y0s.append(bbox[1]); x1s.append(bbox[2]); y1s.append(bbox[3])
                    char_counts.append(len(txt))
                if not texts or not (x0s and y0s and x1s and y1s):
                    continue
                text = " ".join(t.strip() for t in texts).strip()
                x0, y0, x1, y1 = min(x0s), min(y0s), max(x1s), max(y1s)
                sizes = np.array(sizes)
                char_counts = np.array(char_counts)
                if sizes.size == 1:
                    avg_font = float(sizes[0])
                else:
                    avg_font = float((sizes * char_counts).sum() / (char_counts.sum() + 1e-9))
                all_lines.append({
                    "page": pnum,
                    "text": text,
                    "bbox": (x0, y0, x1, y1),
                    "avg_font": avg_font,
                    "line_height": float(y1 - y0),
                    "page_width": float(page_w),
                    "page_height": float(page_h),
                })
    return all_lines


# ---------------------------
# 2) группировка строк в абзацы (улучшенная логика)
# ---------------------------
def group_lines_into_paragraphs(lines: List[Dict],
                                indent_tol: float = 12.0,
                                gap_multiplier: float = 1.4,
                                font_tol: float = 0.18,
                                min_gap_px: float = 3.0) -> List[Dict]:
    """
    Группирует строки в абзацы. Новая логика:
      - если вертикальный gap большой -> новый абзац
      - если поменялся отступ (x0) существенно -> новый абзац
      - если размер шрифта изменился сильно -> новый абзац
      - если строка начинается с буллета/номера -> новый абзац
      - короткий центрированный ряд -> новый абзац (вероятно заголовок/подпись)
    """
    paras = []
    pages = {}
    for L in lines:
        pages.setdefault(L["page"], []).append(L)

    for pnum, pls in pages.items():
        pls.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))  # сверху вниз, слева направо
        if not pls:
            continue
        line_heights = [c["line_height"] for c in pls if c["line_height"] > 0]
        median_h = float(np.median(line_heights)) if line_heights else 10.0
        page_w = pls[0]["page_width"]
        current_para = []
        for i, line in enumerate(pls):
            if not current_para:
                current_para.append(line)
                continue
            prev = current_para[-1]
            x0, y0, x1, y1 = line["bbox"]
            px0, py0, px1, py1 = prev["bbox"]
            gap_y = y0 - py1
            # conditions
            too_big_gap = gap_y > max(median_h * gap_multiplier, min_gap_px)
            indent_changed = abs(x0 - px0) > indent_tol and gap_y >= -1.0
            font_changed = (abs(line["avg_font"] - prev["avg_font"]) / max(prev["avg_font"], 1.0)) > font_tol
            is_bullet = bool(BULLET_RE.match(line["text"].lstrip()))
            # centered short line (likely separate chunk — title/subtitle/figure caption)
            center_x = (x0 + x1) / 2.0
            is_centered_short = (len(line["text"]) <= 50 and
                                 x0 > page_w * 0.12 and x1 < page_w * 0.88 and
                                 abs(center_x - page_w/2.0) < page_w * 0.12)
            # If any of these strongly indicate a new paragraph -> flush
            if too_big_gap or indent_changed or font_changed or is_bullet or is_centered_short:
                paras.append(merge_lines(current_para, pnum))
                current_para = [line]
            else:
                # otherwise continuation of paragraph
                current_para.append(line)
        if current_para:
            paras.append(merge_lines(current_para, pnum))
    return paras


def merge_lines(lines: List[Dict], page_num: int) -> Dict:
    """Сливаем строки в абзац, вычисляем агрегированные поля"""
    text = " ".join(l["text"] for l in lines).strip()
    x0 = min(l["bbox"][0] for l in lines)
    y0 = min(l["bbox"][1] for l in lines)
    x1 = max(l["bbox"][2] for l in lines)
    y1 = max(l["bbox"][3] for l in lines)
    # средний шрифт, взвешенный по длине строки
    lengths = np.array([len(l["text"]) for l in lines], dtype=float)
    fonts = np.array([l["avg_font"] for l in lines], dtype=float)
    if lengths.sum() > 0:
        avg_font = float((fonts * lengths).sum() / lengths.sum())
    else:
        avg_font = float(fonts.mean()) if len(fonts) else 0.0
    return {
        "page": page_num,
        "text": text,
        "bbox": (x0, y0, x1, y1),
        "avg_font": avg_font,
        "num_lines": len(lines),
        "page_width": lines[0]["page_width"],
        "page_height": lines[0]["page_height"],
    }


# ---------------------------
# 3) классификация абзацев
# ---------------------------
def classify_paragraphs(paras: List[Dict],
                        title_font_multiplier: float = 1.35,
                        absolute_title_size: float = 16.0,
                        subtitle_len: int = 120,
                        width_ratio_for_subtitle: float = 0.6) -> List[Dict]:
    """
    Классифицирует абзацы: Title / Subtitle / Text
    Правила:
      - Title: заметно больше медианного шрифта или явно большой абсолютный размер + вверху страницы
      - Subtitle: короткий текст, узкая ширина OR следует прямо после Title
      - Text: иначе
    """
    pages = {}
    for p in paras:
        pages.setdefault(p["page"], []).append(p)

    out = []
    for pnum, p_list in pages.items():
        p_list.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
        fonts = np.array([p["avg_font"] for p in p_list])
        median_font = float(np.median(fonts)) if fonts.size else 0.0
        for i, p in enumerate(p_list):
            text = p["text"].strip()
            x0, y0, x1, y1 = p["bbox"]
            w = x1 - x0
            page_w = p["page_width"]
            length = len(text)
            # candidate Title
            cond_title_font = (p["avg_font"] >= title_font_multiplier * median_font and length < 1000)
            cond_title_abs = (p["avg_font"] >= absolute_title_size)
            cond_top = (y0 <= 0.22 * p["page_height"])
            is_title = (cond_title_font or cond_title_abs) and cond_top
            # candidate subtitle
            is_narrow = (w < page_w * width_ratio_for_subtitle)
            is_short = (length <= subtitle_len)
            follows_title = (i > 0 and p_list[i - 1].get("label") == "Title")
            if is_title:
                label = "Title"
            elif follows_title and is_short:
                label = "Subtitle"
            elif is_short and is_narrow:
                label = "Subtitle"
            else:
                label = "Text"
            p_out = dict(p)
            p_out["label"] = label
            p_out["median_font_on_page"] = median_font
            out.append(p_out)
    return out


# ---------------------------
# 4) визуализация + экспорт JSON
# ---------------------------
def visualize_paragraphs(pdf_path: str,
                         paras: List[Dict],
                         page_num: int = 0,
                         show_labels: bool = True,
                         save_path: str = None,
                         linewidth: float = 1.6):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap()
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4-ish
    ax.imshow(img)
    ax.axis("off")
    color_map = {"Title": "magenta", "Subtitle": "blue", "Text": "cyan"}

    page_paras = [p for p in paras if p["page"] == page_num]
    page_paras.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    for p in page_paras:
        x0, y0, x1, y1 = p["bbox"]
        label = p.get("label", "Text")
        color = color_map.get(label, "yellow")
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=linewidth, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        if show_labels:
            s = p["text"].replace("\n", " ")
            if len(s) > 120:
                s = s[:117] + "..."
            ax.text(x1 + 6, y0, f"{label}: {s}", fontsize=7, verticalalignment="top",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved visualization to:", save_path)
    else:
        plt.show()


def export_paragraphs_to_json(paras: List[Dict], out_path: str):
    import json
    # приводим к сериализуемому виду
    serial = []
    for p in paras:
        serial.append({
            "page": int(p["page"]),
            "text": p["text"],
            "bbox": [float(x) for x in p["bbox"]],
            "label": p.get("label", "Text"),
            "avg_font": float(p.get("avg_font", 0.0)),
            "num_lines": int(p.get("num_lines", 1))
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2)
    print("Exported to", out_path)


def pdf_to_json_ai(pdf, indent_tol=12.0,
                        gap_multiplier=1.0,
                        font_tol=0.18,
                        min_gap_px=3.0,
                        title_font_multiplier=1.35,
                        absolute_title_size=16.0,
                        subtitle_len=120,
                        width_ratio_for_subtitle=0.6,
                        x_align_tol=12,
                        gap_multiplier_post=1.25,
                        max_single_line_len=140):
    lines = extract_lines_with_fonts(pdf)
    paras = group_lines_into_paragraphs(lines,
                                       indent_tol=indent_tol,
                                       gap_multiplier=gap_multiplier,
                                       font_tol=font_tol,
                                       min_gap_px=min_gap_px)
    labeled = classify_paragraphs(paras,
                                  title_font_multiplier=title_font_multiplier,
                                  absolute_title_size=absolute_title_size,
                                  subtitle_len=subtitle_len,
                                  width_ratio_for_subtitle=width_ratio_for_subtitle)
    refined = postprocess_paragraphs(labeled, x_align_tol=x_align_tol, gap_multiplier=gap_multiplier_post,
                                     max_single_line_len=max_single_line_len)
    visualize_paragraphs(pdf, refined, page_num=0, show_labels=False)
    export_paragraphs_to_json(refined, pdf[:-4] + ".json")
