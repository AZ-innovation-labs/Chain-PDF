import os
import sys
import shutil
import pytesseract
from PIL import Image
from typing import List, Dict, Any, Callable, Optional

class OCREngine:
    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self.tesseract_cmd = self._find_tesseract_binary()
        self.tessdata_dir = self._find_tessdata_dir()

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        if self.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = self.tessdata_dir

    def _find_tesseract_binary(self) -> Optional[str]:
        candidates = [
            os.path.join(self.project_root, "tesseract", "tesseract.exe"),
            os.path.join(self.project_root, "tesseract", "bin", "tesseract.exe"),
            os.path.join(self.project_root, "bin", "tesseract.exe"),
            shutil.which("tesseract")
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return None

    def _find_tessdata_dir(self) -> str:
        candidates = [
            os.path.join(self.project_root, "tessdata"),
            os.path.join(self.project_root, "tesseract", "tessdata"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return os.path.join(self.project_root, "tessdata")

    def is_available(self) -> bool:
        return bool(self.tesseract_cmd and os.path.isfile(self.tesseract_cmd))

    def get_available_languages(self) -> List[str]:
        langs = []
        if os.path.isdir(self.tessdata_dir):
            for f in os.listdir(self.tessdata_dir):
                if f.endswith(".traineddata") or f.endswith(".traineddata.gz"):
                    lang = f.split(".")[0]
                    if lang not in langs:
                        langs.append(lang)
        if not langs:
            langs = ["eng"]
        return sorted(langs)

    def extract_words_and_lines(
        self,
        image: Image.Image,
        lang: str = "eng",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[List[List[Dict[str, Any]]], int]:
        """
        Executes Tesseract OCR on the given image.
        Returns (raw_word_clusters, total_word_count).
        Each cluster represents a detected line containing words with bboxes.
        """
        if not self.is_available():
            raise RuntimeError(
                f"Tesseract binary not found. Please ensure portable Tesseract is installed in {os.path.join(self.project_root, 'tesseract')}"
            )

        if progress_callback:
            progress_callback(0.2, f"Executing Tesseract neural OCR [{lang}]...")

        # Ensure tessdata directory uses clean forward slashes
        clean_tessdata = self.tessdata_dir.replace("\\", "/")
        config = f'--tessdata-dir {clean_tessdata} --psm 3 -c preserve_interword_spaces=0 --dpi 300'
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)

        n_boxes = len(data['text'])
        lines_dict: Dict[tuple, List[Dict[str, Any]]] = {}
        total_words = 0

        for i in range(n_boxes):
            text = (data['text'][i] or '').strip()
            conf = int(data['conf'][i]) if 'conf' in data and data['conf'][i] != '-1' else 0

            # Filter out empty entries or ultra-low confidence noise
            if not text:
                continue

            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            if w <= 0 or h <= 0:
                continue

            block_num = data.get('block_num', [0] * n_boxes)[i]
            par_num = data.get('par_num', [0] * n_boxes)[i]
            line_num = data.get('line_num', [0] * n_boxes)[i]
            line_key = (block_num, par_num, line_num)

            word_info = {
                'text': text,
                'conf': conf,
                'bbox': {
                    'x0': x,
                    'y0': y,
                    'x1': x + w,
                    'y1': y + h
                }
            }

            if line_key not in lines_dict:
                lines_dict[line_key] = []
            lines_dict[line_key].append(word_info)
            total_words += 1

        raw_line_clusters = list(lines_dict.values())
        return raw_line_clusters, total_words
