import io
from typing import List, Dict, Any
from PIL import Image
import pymupdf as fitz

from .layout_analyzer import clean_ocr_text

class PDFBuilder:
    def __init__(self):
        self.doc = fitz.open()

    def add_searchable_page(
        self,
        visual_image: Image.Image,
        ordered_lines: List[Dict[str, Any]],
        width: int,
        height: int,
        jpeg_quality: int = 88
    ):
        """
        Adds a high-resolution visual page and overlays calibrated invisible searchable text.
        """
        page = self.doc.new_page(width=width, height=height)

        # 1. Embed visual background image as compressed JPEG
        if visual_image.mode != "RGB":
            if visual_image.mode in ("RGBA", "LA") or (visual_image.mode == "P" and "transparency" in visual_image.info):
                rgba = visual_image.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                bg.alpha_composite(rgba)
                rgb_img = bg.convert("RGB")
            else:
                rgb_img = visual_image.convert("RGB")
        else:
            rgb_img = visual_image

        img_buffer = io.BytesIO()
        rgb_img.save(img_buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        img_bytes = img_buffer.getvalue()
        rect = fitz.Rect(0, 0, width, height)
        page.insert_image(rect, stream=img_bytes)

        # 2. Inject calibrated invisible text layer (render_mode=3: Neither fill nor stroke text)
        fontname = "helv"  # Standard Helvetica built into PDF spec

        for line in ordered_lines:
            line_h = max(1, line.get('line_h', 12))
            nominal_size = max(4.0, line_h * 0.80)
            # Baseline from top in PyMuPDF coordinate space (0,0 is top-left)
            baseline_y = line['y0'] + (line_h * 0.79)

            for word in line.get('words', []):
                clean_text = clean_ocr_text(word.get('text', ''))
                if not clean_text:
                    continue

                bbox = word.get('bbox', {})
                x0 = bbox.get('x0', 0)
                x1 = bbox.get('x1', x0 + 10)
                y0 = bbox.get('y0', baseline_y - line_h)
                y1 = bbox.get('y1', baseline_y)

                actual_box_w = max(1, x1 - x0)
                actual_box_h = max(1, y1 - y0)
                target_w = actual_box_w + max(1.0, actual_box_h * 0.04)

                try:
                    unit_w = fitz.get_text_length(clean_text, fontname=fontname, fontsize=1.0)
                except Exception:
                    unit_w = 0

                if unit_w > 0:
                    font_size = target_w / unit_w
                else:
                    font_size = nominal_size

                font_size = min(max(font_size, nominal_size * 0.65), nominal_size * 1.60)
                point = fitz.Point(x0, baseline_y)

                # render_mode=3 is invisible text in PDF standard
                try:
                    page.insert_text(
                        point,
                        clean_text,
                        fontsize=font_size,
                        fontname=fontname,
                        render_mode=3
                    )
                except Exception:
                    # Fallback without render_mode if version-constrained
                    page.insert_text(
                        point,
                        clean_text,
                        fontsize=font_size,
                        fontname=fontname,
                        color=(1, 1, 1),
                        fill_opacity=0.0,
                        stroke_opacity=0.0
                    )

    def save_to_bytes(self) -> bytes:
        """
        Serializes and returns the complete PDF document as bytes.
        """
        return self.doc.tobytes(garbage=4, deflate=True)

    def save_to_file(self, file_path: str):
        """
        Saves the document to disk.
        """
        self.doc.save(file_path, garbage=4, deflate=True)

    def close(self):
        if self.doc:
            self.doc.close()
