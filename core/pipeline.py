import io
import base64
from typing import List, Dict, Any, Callable, Optional
from PIL import Image

from .preprocessor import detect_border_crop, get_target_dimensions, adaptive_ink_contrast
from .ocr_engine import OCREngine
from .layout_analyzer import (
    clean_ocr_text,
    stitch_ocr_words,
    separate_columns,
    build_line_object
)
from .pdf_builder import PDFBuilder

def ensure_rgb(image: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    """Converts any image mode (RGBA, LA, P, CMYK) safely to RGB with a clean white background."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, bg_color + (255,))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return image.convert("RGB")

def image_to_base64_thumbnail(image: Image.Image, max_dim: int = 400) -> str:
    """Helper to convert an image into a low-latency base64 thumbnail for UI live previews."""
    rgb_img = ensure_rgb(image)
    thumb = rgb_img.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

class DocumentPipeline:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.ocr_engine = OCREngine(project_root)

    def process_batch(
        self,
        images_data: List[Dict[str, Any]],  # [{'name': str, 'image': PIL.Image}]
        target_res: str = "1080",
        auto_crop: bool = True,
        lang: str = "eng",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> bytes:
        """
        Executes the end-to-end OCR processing pipeline for a batch of images.
        Returns final PDF bytes.
        """
        total_pages = len(images_data)
        if total_pages == 0:
            raise ValueError("No images provided for processing.")

        def emit(event_type: str, data: Dict[str, Any]):
            if event_callback:
                event_callback({"type": event_type, **data})

        emit("log", {"tag": "INIT", "message": f"Starting Python OCR Engine with {total_pages} page(s) [Lang: {lang}, Res: {target_res}]."})
        emit("progress", {"percentage": 2, "label": f"Engine Initialized ({lang}). Preparing document..."})

        pdf_builder = PDFBuilder()

        try:
            for idx, item in enumerate(images_data):
                page_num = idx + 1
                filename = item["name"]
                img: Image.Image = ensure_rgb(item["image"])

                page_frac = idx / total_pages
                emit("progress", {
                    "percentage": page_frac * 100,
                    "label": f"Processing Page {page_num} of {total_pages}: {filename}"
                })
                emit("log", {"tag": "PAGE", "message": f"Page {page_num}/{total_pages}: Loading {filename}..."})

                # 1. Border Detection / Crop
                orig_w, orig_h = img.size
                crop_x, crop_y, crop_w, crop_h = 0, 0, orig_w, orig_h
                crop_status = "Disabled"

                if auto_crop:
                    emit("telemetry", {
                        "page_num": page_num,
                        "total_pages": total_pages,
                        "filename": filename,
                        "crop_status": "Scanning edges..."
                    })
                    crop_x, crop_y, crop_w, crop_h = detect_border_crop(img)
                    was_cropped = (crop_x > 0 or crop_y > 0 or crop_w < orig_w or crop_h < orig_h)
                    if was_cropped:
                        crop_status = f"Cropped (-{orig_w - crop_w}×-{orig_h - crop_h})"
                        emit("log", {"tag": "CROP", "message": f"Auto-crop isolated content: {crop_w}×{crop_h} (offset: {crop_x}, {crop_y})"})
                    else:
                        crop_status = "Preserved (Clean)"
                
                cropped_img = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

                # 2. Resolution Scaling
                target_w, target_h = get_target_dimensions(crop_w, crop_h, target_res)
                if (target_w, target_h) != (crop_w, crop_h):
                    visual_img = cropped_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                else:
                    visual_img = cropped_img

                # 3. Emit Visual Preview & Dimensions to Cockpit HUD
                visual_thumb = image_to_base64_thumbnail(visual_img)
                emit("telemetry", {
                    "page_num": page_num,
                    "total_pages": total_pages,
                    "filename": filename,
                    "dims": f"{target_w} × {target_h} px",
                    "crop_status": crop_status,
                    "visual_preview": visual_thumb
                })

                # 4. Adaptive Ink-Contrast Isolation
                emit("progress", {
                    "percentage": (page_frac + (0.3 / total_pages)) * 100,
                    "label": f"Page {page_num}: Isolating ink contrast..."
                })
                # For OCR, downscale if extremely large to maintain high performance
                max_ocr_dim = 1400
                ocr_scale = 1.0
                if max(target_w, target_h) > max_ocr_dim:
                    ocr_scale = max_ocr_dim / max(target_w, target_h)
                    ocr_w = max(1, round(target_w * ocr_scale))
                    ocr_h = max(1, round(target_h * ocr_scale))
                    ocr_base_img = visual_img.resize((ocr_w, ocr_h), Image.Resampling.BILINEAR)
                else:
                    ocr_w = target_w
                    ocr_h = target_h
                    ocr_base_img = visual_img

                contrast_img = adaptive_ink_contrast(ocr_base_img)
                ocr_thumb = image_to_base64_thumbnail(contrast_img)
                emit("telemetry", {
                    "ocr_preview": ocr_thumb
                })

                # 5. Execute Tesseract Neural OCR with Laser Scanline Trigger
                emit("scanline", {"active": True})
                emit("progress", {
                    "percentage": (page_frac + (0.5 / total_pages)) * 100,
                    "label": f"Page {page_num}: Running neural OCR inference..."
                })
                emit("log", {"tag": "OCR", "message": f"Executing Tesseract neural OCR on page {page_num}..."})

                def ocr_cb(sub_pct, text):
                    emit("progress", {
                        "percentage": (page_frac + ((0.5 + sub_pct * 0.3) / total_pages)) * 100,
                        "label": f"Page {page_num}: {text}"
                    })

                raw_line_clusters, total_words = self.ocr_engine.extract_words_and_lines(
                    contrast_img,
                    lang=lang,
                    progress_callback=ocr_cb
                )

                emit("scanline", {"active": False})

                # Scale coordinates back up to full page space
                to_page_scale = 1.0 / ocr_scale
                structured_lines = []

                for raw_cluster in raw_line_clusters:
                    scaled_words = []
                    for w in raw_cluster:
                        clean = clean_ocr_text(w['text'])
                        if not clean:
                            continue
                        bbox = w['bbox']
                        scaled_words.append({
                            'text': w['text'],
                            'conf': w['conf'],
                            'bbox': {
                                'x0': round(bbox['x0'] * to_page_scale),
                                'y0': round(bbox['y0'] * to_page_scale),
                                'x1': round(bbox['x1'] * to_page_scale),
                                'y1': round(bbox['y1'] * to_page_scale),
                            }
                        })

                    if not scaled_words:
                        continue

                    stitched = stitch_ocr_words(scaled_words)
                    curr_cluster = [stitched[0]]

                    for wi in range(1, len(stitched)):
                        prev_w = stitched[wi - 1]
                        curr_w = stitched[wi]
                        gap = curr_w['bbox']['x0'] - prev_w['bbox']['x1']
                        word_h = max(prev_w['bbox']['y1'] - prev_w['bbox']['y0'], curr_w['bbox']['y1'] - curr_w['bbox']['y0'])
                        split_gap = max(18, word_h * 1.15)

                        if gap > split_gap:
                            structured_lines.append(build_line_object(curr_cluster, target_h))
                            curr_cluster = [curr_w]
                        else:
                            curr_cluster.append(curr_w)

                    if curr_cluster:
                        structured_lines.append(build_line_object(curr_cluster, target_h))

                emit("telemetry", {
                    "words_detected": f"{total_words} words"
                })
                emit("log", {"tag": "OCR", "message": f"Extracted {total_words} words across {len(structured_lines)} text segments."})

                # 6. Multi-Column Reading Order Alignment
                ordered_lines = separate_columns(structured_lines, target_w)

                # 7. Add page to PDF with high-res visual image and invisible text layer
                emit("progress", {
                    "percentage": (page_frac + (0.9 / total_pages)) * 100,
                    "label": f"Page {page_num}: Injecting calibrated invisible text layer..."
                })
                pdf_builder.add_searchable_page(
                    visual_image=visual_img,
                    ordered_lines=ordered_lines,
                    width=target_w,
                    height=target_h
                )

            # 8. Save compiled PDF
            emit("progress", {"percentage": 98, "label": "Compiling final searchable PDF..."})
            emit("log", {"tag": "PDF", "message": "Compiling PDF document stream..."})
            pdf_bytes = pdf_builder.save_to_bytes()

            emit("log", {"tag": "SUCCESS", "message": f"Conversion complete! PDF size: {(len(pdf_bytes) / (1024 * 1024)):.2f} MB."})
            emit("progress", {"percentage": 100, "label": "Searchable PDF Successfully Generated!"})
            emit("done", {"size_mb": round(len(pdf_bytes) / (1024 * 1024), 2)})

            return pdf_bytes

        finally:
            pdf_builder.close()
