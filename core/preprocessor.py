import numpy as np
from PIL import Image

def detect_border_crop(image: Image.Image) -> tuple[int, int, int, int]:
    """
    Detects scanner frames, black borders, and extraneous margins.
    Returns (x, y, w, h) bounding box of the cropped content.
    """
    orig_w, orig_h = image.size
    if orig_w < 80 or orig_h < 80:
        return (0, 0, orig_w, orig_h)

    rgb = np.array(image.convert("RGB"))
    h, w, _ = rgb.shape

    def get_edge_baseline(is_row: bool, index: int, max_len: int):
        step = max(1, max_len // 100)
        indices = np.arange(0, max_len, step)
        if is_row:
            pixels = rgb[index, indices, :].astype(np.float32)
        else:
            pixels = rgb[indices, index, :].astype(np.float32)

        avg = np.mean(pixels, axis=0)
        avg_r, avg_g, avg_b = avg[0], avg[1], avg[2]
        is_neutral = (abs(avg_r - avg_g) < 30 and abs(avg_g - avg_b) < 30 and abs(avg_r - avg_b) < 30)
        dev = np.mean(np.sum(np.abs(pixels - avg), axis=1))
        is_uniform = dev < 35.0
        return avg_r, avg_g, avg_b, bool(is_neutral and is_uniform)

    def is_corner_to_corner_line(is_row: bool, index: int, max_len: int, base_r: float, base_g: float, base_b: float) -> bool:
        step = 2 if max_len > 2500 else 1
        indices = np.arange(0, max_len, step)
        if is_row:
            pixels = rgb[index, indices, :].astype(np.float32)
        else:
            pixels = rgb[indices, index, :].astype(np.float32)

        base = np.array([base_r, base_g, base_b], dtype=np.float32)
        diff = np.sum(np.abs(pixels - base), axis=1)
        line_mask = diff > 45.0

        line_count = np.sum(line_mask)
        total = len(indices)
        if total == 0:
            return False

        coverage = line_count / total
        corner_margin_idx = int((max_len * 0.05) / step)
        start_found = np.any(line_mask[:max(1, corner_margin_idx)])
        end_found = np.any(line_mask[-max(1, corner_margin_idx):])

        # Check maximum gap between line pixels
        non_line_indices = np.where(~line_mask)[0]
        max_gap = 0
        if len(non_line_indices) > 0:
            # find consecutive gaps
            gaps = np.diff(np.where(line_mask)[0]) - 1 if np.any(line_mask) else np.array([total])
            if len(gaps) > 0:
                max_gap = np.max(gaps) * step
        max_allowed_gap = max(10, int(max_len * 0.05))

        return bool(coverage >= 0.92 and start_found and end_found and max_gap <= max_allowed_gap)

    max_scan_y = int(h * 0.25)
    max_scan_x = int(w * 0.25)
    top, bottom, left, right = 0, h, 0, w

    # Top boundary
    top_base_r, top_base_g, top_base_b, top_valid = get_edge_baseline(True, 0, w)
    if top_valid:
        for y in range(1, max_scan_y):
            if is_corner_to_corner_line(True, y, w, top_base_r, top_base_g, top_base_b):
                top = y + 1
                break

    # Bottom boundary
    bot_base_r, bot_base_g, bot_base_b, bot_valid = get_edge_baseline(True, h - 1, w)
    if bot_valid:
        for y in range(h - 2, h - max_scan_y, -1):
            if is_corner_to_corner_line(True, y, w, bot_base_r, bot_base_g, bot_base_b):
                bottom = y
                break

    # Left boundary
    left_base_r, left_base_g, left_base_b, left_valid = get_edge_baseline(False, 0, h)
    if left_valid:
        for x in range(1, max_scan_x):
            if is_corner_to_corner_line(False, x, h, left_base_r, left_base_g, left_base_b):
                left = x + 1
                break

    # Right boundary
    right_base_r, right_base_g, right_base_b, right_valid = get_edge_baseline(False, w - 1, h)
    if right_valid:
        for x in range(w - 2, w - max_scan_x, -1):
            if is_corner_to_corner_line(False, x, h, right_base_r, right_base_g, right_base_b):
                right = x
                break

    crop_w = max(50, right - left)
    crop_h = max(50, bottom - top)
    return (left, top, crop_w, crop_h)

def get_target_dimensions(orig_w: int, orig_h: int, target_res: str) -> tuple[int, int]:
    """
    Computes output dimensions matching the specified resolution preset.
    """
    if target_res == 'original':
        return (orig_w, orig_h)

    limits = {
        '720': {'max': 1280, 'min': 720},
        '1080': {'max': 1920, 'min': 1080},
        '1440': {'max': 2560, 'min': 1440},
        '4k': {'max': 3840, 'min': 2160},
        '8k': {'max': 7680, 'min': 4320}
    }
    config = limits.get(target_res, limits['1080'])
    is_landscape = orig_w >= orig_h
    bound_w = config['max'] if is_landscape else config['min']
    bound_h = config['min'] if is_landscape else config['max']
    scale = min(1.0, bound_w / orig_w, bound_h / orig_h)

    return (max(1, round(orig_w * scale)), max(1, round(orig_h * scale)))

def adaptive_ink_contrast(image: Image.Image) -> Image.Image:
    """
    Adaptive Ink-Contrast Isolation filter:
    Converts colored or faint text into solid high-contrast dark pixels while cleaning
    tinted, yellowed, or noisy scanned backgrounds.
    """
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    min_ch = np.min(rgb, axis=2)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    val = min_ch * 0.75 + luminance * 0.25

    # Adaptive contrast thresholds
    val = np.where(val > 195.0, 255.0, np.where(val < 165.0, val * 0.55, val))
    val = np.clip(val, 0, 255).astype(np.uint8)

    # Return RGB image with contrast applied to all channels
    stacked = np.stack([val, val, val], axis=2)
    return Image.fromarray(stacked, mode="RGB")
