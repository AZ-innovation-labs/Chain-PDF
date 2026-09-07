import re
from typing import List, Dict, Any

def clean_ocr_text(raw_text: str) -> str:
    """
    Sanitizes common OCR misclassifications, artifacts, and special quotation marks.
    """
    if not raw_text:
        return ""
    text = raw_text

    # Curly & smart quotes
    text = re.sub(r'[\u201C\u201D\u201E\u201F\u2033\u2036]', '"', text)
    text = re.sub(r'[\u2018\u2019\u201A\u201B\u2032\u2035]', "'", text)
    text = re.sub(r'[\u2013\u2014]', '-', text)

    # Pipe artifact fixing (| -> l)
    text = re.sub(r'([a-zA-Z0-9])\|', r'\1l', text)
    text = re.sub(r'\|([a-zA-Z0-9])', r'l\1', text)

    # Strip control codes while preserving standard characters
    text = re.sub(r'[^\x20-\x7E\u00A0-\u024F]', '', text)
    return text.strip()

def stitch_ocr_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reassembles words split by display font kerning or tight spacing.
    Each word dict has keys: 'text', 'bbox' with x0, y0, x1, y1.
    """
    if len(words) <= 1:
        return words

    sorted_words = sorted(words, key=lambda w: w['bbox']['x0'])
    stitched = []

    curr = {
        'text': clean_ocr_text(sorted_words[0]['text']),
        'bbox': dict(sorted_words[0]['bbox'])
    }

    for i in range(1, len(sorted_words)):
        next_w = sorted_words[i]
        next_text = clean_ocr_text(next_w['text'])
        if not next_text:
            continue

        curr_h = max(1, curr['bbox']['y1'] - curr['bbox']['y0'])
        next_h = max(1, next_w['bbox']['y1'] - next_w['bbox']['y0'])
        avg_h = (curr_h + next_h) / 2.0
        gap = next_w['bbox']['x0'] - curr['bbox']['x1']

        is_punct = bool(re.match(r'^[.,;:!?\')\]}"]', next_text))
        space_threshold = (avg_h * 0.35) if is_punct else (avg_h * 0.20)

        if gap < space_threshold:
            curr['text'] += next_text
            curr['bbox']['x1'] = max(curr['bbox']['x1'], next_w['bbox']['x1'])
            curr['bbox']['y0'] = min(curr['bbox']['y0'], next_w['bbox']['y0'])
            curr['bbox']['y1'] = max(curr['bbox']['y1'], next_w['bbox']['y1'])
        else:
            if curr['text']:
                stitched.append(curr)
            curr = {
                'text': next_text,
                'bbox': dict(next_w['bbox'])
            }

    if curr['text']:
        stitched.append(curr)

    return stitched

def build_line_object(words: List[Dict[str, Any]], page_height: int) -> Dict[str, Any]:
    """
    Constructs a structured line object with line bounding box and baseline calculation.
    """
    x0 = min(w['bbox']['x0'] for w in words)
    x1 = max(w['bbox']['x1'] for w in words)
    y0 = min(w['bbox']['y0'] for w in words)
    y1 = max(w['bbox']['y1'] for w in words)
    line_h = max(1, y1 - y0)
    baseline_y = (page_height - y1) + (line_h * 0.21)

    return {
        'words': words,
        'x0': x0,
        'x1': x1,
        'y0': y0,
        'y1': y1,
        'line_h': line_h,
        'baseline_y': baseline_y
    }

def separate_columns(lines: List[Dict[str, Any]], page_width: int) -> List[Dict[str, Any]]:
    """
    Uses histogram gutter binning to group multi-column layouts into natural reading order.
    """
    if len(lines) <= 1:
        return lines

    content_lines = []
    banners = []

    for l in lines:
        if (l['x1'] - l['x0']) > page_width * 0.65:
            banners.append(l)
        else:
            content_lines.append(l)

    if not content_lines:
        return sorted(lines, key=lambda l: l['y0'])

    bin_size = 4
    num_bins = max(1, (page_width + bin_size - 1) // bin_size)
    occupancy = [0] * num_bins

    for l in content_lines:
        start_bin = max(0, min(num_bins - 1, l['x0'] // bin_size))
        end_bin = max(0, min(num_bins - 1, l['x1'] // bin_size))
        for b in range(start_bin, end_bin + 1):
            occupancy[b] += 1

    min_x = min(l['x0'] for l in content_lines)
    max_x = max(l['x1'] for l in content_lines)
    start_bin = max(0, min_x // bin_size)
    end_bin = min(num_bins - 1, max_x // bin_size)

    gutters = []
    in_gutter = False
    gutter_start = 0

    for b in range(start_bin, end_bin + 1):
        if occupancy[b] == 0:
            if not in_gutter:
                in_gutter = True
                gutter_start = b
        else:
            if in_gutter:
                in_gutter = False
                gutter_w = (b - gutter_start) * bin_size
                if gutter_w >= 16:
                    gutters.append(((gutter_start + b) / 2) * bin_size)

    if not gutters:
        def line_sort_key(l):
            return l['y0']
        return sorted(lines, key=line_sort_key)

    split_x = sorted(gutters)
    columns = [[] for _ in range(len(split_x) + 1)]

    for l in content_lines:
        mid_x = (l['x0'] + l['x1']) / 2
        col_idx = len(split_x)
        for i, sx in enumerate(split_x):
            if mid_x < sx:
                col_idx = i
                break
        columns[col_idx].append(l)

    for col in columns:
        col.sort(key=lambda l: l['y0'])

    min_y = min(l['y0'] for l in content_lines)
    max_y = max(l['y1'] for l in content_lines)

    top_banners = sorted([b for b in banners if b['y1'] <= min_y], key=lambda b: b['y0'])
    bottom_banners = sorted([b for b in banners if b['y0'] >= max_y], key=lambda b: b['y0'])
    mid_banners = sorted([b for b in banners if b['y1'] > min_y and b['y0'] < max_y], key=lambda b: b['y0'])

    ordered_columns = []
    for col in columns:
        ordered_columns.extend(col)

    return top_banners + ordered_columns + mid_banners + bottom_banners
