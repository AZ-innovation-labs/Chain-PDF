import os
import sys
import time
import math
import urllib.request
from typing import Optional, Callable, Dict, Any, Tuple
from PIL import Image, ImageEnhance

# Register PyTorch DLL directories for Windows embedded Python 3.11
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    torch_lib = os.path.join(site_packages, "torch", "lib")
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "2x_Text2HD": {
        "filename": "2x_Text2HD_v.1-RealPLKSR.pth",
        "url": "https://huggingface.co/buckets/AZ-innovation-labs/upscalers/resolve/2x_Text2HD_v.1-RealPLKSR.pth?download=true",
        "scale": 2,
        "name": "2x Text2HD (RealPLKSR)",
        "desc": "Ultra-fast crisp text and document sharpening"
    },
    "8x_NMKD": {
        "filename": "8x_NMKD-Typescale_175k.pth",
        "url": "https://huggingface.co/buckets/AZ-innovation-labs/upscalers/resolve/8x_NMKD-Typescale_175k.pth?download=true",
        "scale": 8,
        "name": "8x NMKD Typescale (ESRGAN)",
        "desc": "Deep typography and character contour restoration"
    }
}

RES_LIMITS: Dict[str, Dict[str, int]] = {
    "720": {"max": 1280, "min": 720},
    "1080": {"max": 1920, "min": 1080},
    "1440": {"max": 2560, "min": 1440},
    "4k": {"max": 3840, "min": 2160},
    "8k": {"max": 7680, "min": 4320}
}
RES_STEPS = ["720", "1080", "1440", "4k", "8k"]

STEP_UP_MAP: Dict[str, str] = {
    "720": "1080",
    "1080": "1440",
    "1440": "4k",
    "4k": "8k"
}

class UpscaleEngine:
    """
    AI Deep Learning Upscaling & Enhancement Engine for ChainPDF.
    Supports RealPLKSR and ESRGAN models with seamless tiled inference
    and hardware acceleration (CUDA/CPU).
    """

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self.models_dir = os.path.join(self.project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self._cached_models: Dict[str, Any] = {}
        self._device = None
        self._device_name = "CPU"

    def is_available(self) -> bool:
        """Checks if PyTorch and Spandrel are installed."""
        try:
            import torch
            import spandrel
            return True
        except ImportError:
            return False

    def resolve_device(self, device_choice: str = "gpu") -> Tuple[Any, str]:
        """
        Resolves device choice ('gpu' or 'cpu') to a torch.device and descriptive name.
        Defaults to GPU if available; falls back cleanly to CPU if CUDA is unavailable.
        """
        import torch
        choice = (device_choice or "gpu").strip().lower()
        if choice in ("gpu", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda"), torch.cuda.get_device_name(0)
            else:
                return torch.device("cpu"), "CPU (CUDA Unavailable)"
        return torch.device("cpu"), "CPU Multi-Threading"

    def get_hardware_info(self) -> Dict[str, Any]:
        """Returns details about hardware acceleration status."""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Multi-Threading"
            return {
                "torch_version": torch.__version__,
                "cuda_available": has_cuda,
                "device_name": device_name,
                "available_devices": ["gpu", "cpu"] if has_cuda else ["cpu"],
                "default_device": "gpu" if has_cuda else "cpu",
                "models": self.get_available_models()
            }
        except ImportError:
            return {
                "torch_version": None,
                "cuda_available": False,
                "device_name": "Not Installed",
                "available_devices": ["cpu"],
                "default_device": "cpu",
                "models": self.get_available_models()
            }

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Returns local existence status for all supported models."""
        result = {}
        for key, info in MODEL_REGISTRY.items():
            path = os.path.join(self.models_dir, info["filename"])
            exists = os.path.isfile(path)
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 2) if exists else 0
            result[key] = {
                **info,
                "exists": exists,
                "size_mb": size_mb,
                "local_path": path if exists else None
            }
        return result

    def ensure_model_file(self, model_key: str, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Ensures the requested model file is downloaded and returns its path."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")

        info = MODEL_REGISTRY[model_key]
        path = os.path.join(self.models_dir, info["filename"])

        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            return path

        if progress_callback:
            progress_callback(f"Downloading model '{info['name']}' weights from Hugging Face...")

        url = info["url"]
        tmp_path = path + ".tmp"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ChainPDF AI Engine)"})

        try:
            with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                last_emit = 0
                chunk_size = 1024 * 256
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if progress_callback and (now - last_emit > 1.0 or downloaded == total):
                        last_emit = now
                        pct = round((downloaded / total) * 100) if total > 0 else 0
                        mb = round(downloaded / (1024 * 1024), 1)
                        total_mb = round(total / (1024 * 1024), 1) if total > 0 else "?"
                        progress_callback(f"Downloading {info['name']}: {pct}% ({mb}/{total_mb} MB)...")

            if os.path.isfile(tmp_path):
                if os.path.isfile(path):
                    os.remove(path)
                os.rename(tmp_path, path)
        except Exception as e:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Failed to download model '{info['name']}': {e}")

        return path

    def _get_device(self):
        """Initializes and returns the default computation device."""
        dev, _ = self.resolve_device("gpu")
        return dev

    def load_model(
        self,
        model_key: str,
        device: Optional[Any] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """Loads and caches the model on the target device."""
        import torch
        from spandrel import ModelLoader
        try:
            import spandrel_extra_arches
            spandrel_extra_arches.install()
        except Exception:
            pass

        target_device = device if device is not None else self.resolve_device("gpu")[0]
        dev_type = target_device.type
        cache_key = f"{model_key}_{dev_type}"

        if cache_key in self._cached_models:
            return self._cached_models[cache_key]

        model_path = self.ensure_model_file(model_key, progress_callback)
        dev_label = torch.cuda.get_device_name(0) if dev_type == "cuda" else "CPU"

        if progress_callback:
            progress_callback(f"Loading {MODEL_REGISTRY[model_key]['name']} into {dev_label}...")

        loader = ModelLoader(target_device)
        descriptor = loader.load_from_file(model_path)
        descriptor.eval()

        self._cached_models[cache_key] = descriptor
        return descriptor

    @staticmethod
    def _calculate_grid(h: int, w: int, target_tiles: int) -> Tuple[int, int]:
        """
        Calculates the optimal (rows, cols) grid for target_tiles such that
        individual tiles best approximate a square (1:1) aspect ratio on an image of size HxW.
        """
        if target_tiles <= 1:
            return 1, 1

        # Check exact factor pairs of target_tiles
        exact_pairs = []
        for r in range(1, target_tiles + 1):
            if target_tiles % r == 0:
                exact_pairs.append((r, target_tiles // r))

        if exact_pairs:
            best_grid = exact_pairs[0]
            best_ratio_diff = float("inf")
            for r, c in exact_pairs:
                th = h / r
                tw = w / c
                if th <= 0 or tw <= 0:
                    continue
                ratio_diff = max(tw / th, th / tw) - 1.0
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_grid = (r, c)
            return best_grid

        # Fallback for prime/irregular counts: check candidate pairs close to target_tiles
        candidates = []
        for total in range(max(2, target_tiles - 2), target_tiles + 3):
            for r in range(1, total + 1):
                if total % r == 0:
                    candidates.append((r, total // r))

        best_grid = (1, target_tiles) if w >= h else (target_tiles, 1)
        best_diff = float("inf")
        for r, c in candidates:
            th = h / r
            tw = w / c
            if th <= 0 or tw <= 0:
                continue
            ratio_diff = max(tw / th, th / tw) - 1.0
            diff = ratio_diff + abs((r * c) - target_tiles) * 0.5
            if diff < best_diff:
                best_diff = diff
                best_grid = (r, c)

        return best_grid

    @staticmethod
    def _get_tile_ranges(length: int, num_splits: int, pad: int) -> list[Tuple[int, int]]:
        """
        Computes (start, end) pixel slices along an axis for num_splits partitions,
        extending non-boundary edges by pad pixels to provide overlap for feathering.
        """
        if num_splits <= 1:
            return [(0, length)]

        ranges = []
        for i in range(num_splits):
            start_nom = round(i * length / num_splits)
            end_nom = round((i + 1) * length / num_splits)
            start = max(0, start_nom - pad) if i > 0 else 0
            end = min(length, end_nom + pad) if i < num_splits - 1 else length
            ranges.append((start, end))
        return ranges

    def _tiled_inference(
        self,
        model_descriptor,
        img_tensor,  # torch.Tensor (1, 3, H, W) in [0, 1] on device
        tile_mode: str = "multi",
        tile_count: Any = "auto",
        tile_size: int = 512,
        tile_pad: int = 24,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Any, str]:
        """
        Executes super-resolution inference in single-tile or multi-tile mode.
        Returns: (output_tensor, tile_summary_str)
        """
        import torch

        _, _, h, w = img_tensor.shape
        scale = getattr(model_descriptor, "scale", 1)

        # Single Tile Pass (or direct forward pass if small enough)
        is_single = (tile_mode == "single") or (tile_count in (1, "1"))
        if is_single or (h <= tile_size and w <= tile_size and tile_count == "auto"):
            if progress_callback:
                progress_callback(f"Running single-tile neural pass (full frame {w}×{h})...")
            with torch.no_grad():
                out = model_descriptor(img_tensor).clamp(0.0, 1.0)
            return out, "Single-Tile Pass"

        out_h = h * scale
        out_w = w * scale
        output = torch.zeros((1, 3, out_h, out_w), dtype=img_tensor.dtype, device=img_tensor.device)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=img_tensor.dtype, device=img_tensor.device)

        # Multi-Tile: Check if custom/preset tile count is requested
        parsed_count: Optional[int] = None
        if tile_count not in ("auto", None, ""):
            try:
                parsed_count = max(1, int(tile_count))
            except (ValueError, TypeError):
                parsed_count = None

        if parsed_count is not None and parsed_count > 1:
            # User-specified number of tiles
            rows, cols = self._calculate_grid(h, w, parsed_count)
            y_ranges = self._get_tile_ranges(h, rows, tile_pad)
            x_ranges = self._get_tile_ranges(w, cols, tile_pad)
            tile_ranges = [(y0, y1, x0, x1) for (y0, y1) in y_ranges for (x0, x1) in x_ranges]
            total_tiles = len(tile_ranges)
            tile_label = f"Multi-Tile ({total_tiles} tiles, {rows}×{cols} grid)"
        else:
            # Hardware-adaptive automatic tile sizing
            step = max(32, tile_size - (2 * tile_pad))
            y_steps = max(1, math.ceil((h - tile_size) / step) + 1)
            x_steps = max(1, math.ceil((w - tile_size) / step) + 1)

            y_indices = sorted(list(set([min(i * step, max(0, h - tile_size)) for i in range(y_steps)])))
            x_indices = sorted(list(set([min(j * step, max(0, w - tile_size)) for j in range(x_steps)])))

            tile_ranges = []
            for y0 in y_indices:
                y1 = min(y0 + tile_size, h)
                for x0 in x_indices:
                    x1 = min(x0 + tile_size, w)
                    tile_ranges.append((y0, y1, x0, x1))
            total_tiles = len(tile_ranges)
            tile_label = f"Multi-Tile ({total_tiles} tiles, auto)"

        tile_idx = 0
        for y0, y1, x0, x1 in tile_ranges:
            tile_idx += 1
            if progress_callback and total_tiles > 1:
                progress_callback(f"Processing neural tile {tile_idx}/{total_tiles}...")

            tile = img_tensor[:, :, y0:y1, x0:x1]
            th, tw = y1 - y0, x1 - x0

            with torch.no_grad():
                out_tile = model_descriptor(tile).clamp(0.0, 1.0)

            out_th = th * scale
            out_tw = tw * scale

            # 2D Linear feathering window
            pad_scaled = min(tile_pad * scale, max(1, out_th // 4), max(1, out_tw // 4))
            wy = torch.ones((out_th, 1), dtype=img_tensor.dtype, device=img_tensor.device)
            wx = torch.ones((1, out_tw), dtype=img_tensor.dtype, device=img_tensor.device)

            if y0 > 0 and pad_scaled > 0:
                wy[:pad_scaled, 0] = torch.linspace(0.01, 1.0, pad_scaled, device=img_tensor.device)
            if y1 < h and pad_scaled > 0:
                wy[-pad_scaled:, 0] = torch.linspace(1.0, 0.01, pad_scaled, device=img_tensor.device)

            if x0 > 0 and pad_scaled > 0:
                wx[0, :pad_scaled] = torch.linspace(0.01, 1.0, pad_scaled, device=img_tensor.device)
            if x1 < w and pad_scaled > 0:
                wx[0, -pad_scaled:] = torch.linspace(1.0, 0.01, pad_scaled, device=img_tensor.device)

            mask = (wy * wx).unsqueeze(0).unsqueeze(0)  # (1, 1, out_th, out_tw)

            out_y0 = y0 * scale
            out_x0 = x0 * scale
            output[:, :, out_y0:out_y0 + out_th, out_x0:out_x0 + out_tw] += out_tile * mask
            weights[:, :, out_y0:out_y0 + out_th, out_x0:out_x0 + out_tw] += mask

        output = output / weights.clamp(min=1e-5)
        return output.clamp(0.0, 1.0), tile_label

    @staticmethod
    def _heal_text_pinholes(tensor, kernel_size: int = 3):
        """
        Applies GPU-accelerated stroke hole-closing and anti-dithering.
        Seals 1-2px sub-pixel convolution / PixelShuffle pinholes inside character strokes
        before output conversion and downsampling, eliminating mottled dot artifacts.
        """
        import torch
        import torch.nn.functional as F

        # Compute luminance Y: (1, 1, H, W)
        Y = 0.299 * tensor[:, 0:1, :, :] + 0.587 * tensor[:, 1:2, :, :] + 0.114 * tensor[:, 2:3, :, :]
        inv_Y = 1.0 - Y
        pad = kernel_size // 2

        # Morphological closing (dilation followed by erosion) on inverted luminance
        dilated = F.max_pool2d(inv_Y, kernel_size=kernel_size, stride=1, padding=pad)
        closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)

        # 3x3 local smoothing to equalize sub-pixel dither inside solid ink
        avg_closed = F.avg_pool2d(closed, kernel_size=3, stride=1, padding=1)
        smoothed = 0.5 * closed + 0.5 * avg_closed

        # Only infill where pinholes or dither troughs occurred (where smoothed > inv_Y)
        healed_inv_Y = torch.max(inv_Y, smoothed)
        healed_Y = torch.clamp(1.0 - healed_inv_Y, 0.0, 1.0)

        # Scale RGB proportionally to preserve ink color tone
        scale = (healed_Y / Y.clamp(min=1e-4)).clamp(0.0, 1.0)
        return torch.clamp(tensor * scale, 0.0, 1.0)

    def enhance(
        self,
        image: Image.Image,
        upscale_mode: str = "auto",
        upscale_model: str = "2x_Text2HD",
        upscale_device: str = "gpu",
        anti_dither: bool = True,
        tile_mode: str = "multi",
        tile_count: Any = "auto",
        target_res: str = "1080",
        sharpness: float = 2.0,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Image.Image, bool, str]:
        """
        Enhances an image using the selected AI model or algorithmic scaling + sharpness filter.
        Supported upscale modes:
          - 'auto': Intelligently triggers AI upscale if image < 80% of target resolution, else regular rescale
          - 'disabled': Standard OCR pass-through
          - 'regular': Pure algorithmic Lanczos scaling + sharpness
          - 'always': Always neural upscale
          - 'downsample_neural': Downscales to lower tier (720p, 1080p, 1440p, 4k) then neural upscales to target
          - 'neural_downsample': Neural upscales to a step above target resolution (up to 4K) then downsamples
        Returns: (resulting_image, was_upscaled, summary_message)
        """
        # Alias legacy mode names
        if upscale_mode == "auto_1080":
            upscale_mode = "auto"
        elif upscale_mode == "enhanced_1080":
            upscale_mode = "neural_downsample"

        if upscale_mode == "disabled" or target_res == "original":
            return image, False, "Upscale disabled (Pass-through)"

        orig_w, orig_h = image.size
        is_landscape = orig_w >= orig_h

        # Resolve target resolution bounds
        res_config = RES_LIMITS.get(target_res, RES_LIMITS["1080"])
        bound_w = res_config["max"] if is_landscape else res_config["min"]
        bound_h = res_config["min"] if is_landscape else res_config["max"]
        scale_to_target = min(bound_w / orig_w, bound_h / orig_h)
        target_w = max(1, round(orig_w * scale_to_target))
        target_h = max(1, round(orig_h * scale_to_target))

        # Size ratio of original image relative to fitted target resolution (1.0 = 100% of target)
        size_ratio = (1.0 / scale_to_target) if scale_to_target > 0 else 1.0

        # Resolve step-up resolution bounds (one step above target_res)
        step_up_tier = STEP_UP_MAP.get(target_res, "1440")
        step_up_cfg = RES_LIMITS.get(step_up_tier, RES_LIMITS["1440"])
        step_up_bound_w = step_up_cfg["max"] if is_landscape else step_up_cfg["min"]
        step_up_bound_h = step_up_cfg["min"] if is_landscape else step_up_cfg["max"]
        scale_to_step_up = min(step_up_bound_w / orig_w, step_up_bound_h / orig_h)
        step_up_w = max(1, round(orig_w * scale_to_step_up))
        step_up_h = max(1, round(orig_h * scale_to_step_up))

        # Handle 'neural_downsample' mode native resolution check:
        # If the original image is ALREADY at or higher than the one-step-above tier
        # (e.g. A4 scan 2480x3508 when target is 1080p and step-up is 1440p):
        # The image is already supersampled! Neural upscaling an already-high-res image (or downscaling it beforehand)
        # destroys character strokes and makes text gibberish. We preserve native resolution and
        # downsample cleanly with Lanczos directly to target_res.
        if upscale_mode == "neural_downsample" and orig_w >= step_up_w and orig_h >= step_up_h:
            start_time = time.time()
            if progress_callback:
                progress_callback(
                    f"Native image ({orig_w}×{orig_h}) already exceeds {step_up_tier}p ({step_up_w}×{step_up_h}). "
                    f"Directly supersampling down to {target_res}p ({target_w}×{target_h})..."
                )
            if (target_w, target_h) != (orig_w, orig_h):
                final_img = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                final_img = image.copy()

            elapsed = time.time() - start_time
            summary = (
                f"Neural Super-Sample (Intelligent Native): Native resolution ({orig_w}×{orig_h} >= {step_up_tier}p) "
                f"downsampled cleanly with Lanczos to {target_w}×{target_h} ({target_res}p) ({elapsed:.2f}s)"
            )
            return final_img, False, summary

        # Handle 'regular' upscale mode: pure algorithmic scaling + post-scale sharpness filter
        if upscale_mode == "regular":
            start_time = time.time()
            if progress_callback:
                progress_callback("Running regular algorithmic scaling (Lanczos)...")

            if (target_w, target_h) != (orig_w, orig_h):
                scaled_img = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                scaled_img = image.copy()

            sharpness_val = max(0.0, sharpness)
            has_sharpness = (sharpness_val > 0.0 and abs(sharpness_val - 1.0) >= 0.01)
            if has_sharpness:
                if progress_callback:
                    progress_callback(f"Applying post-scale sharpness filter ({sharpness_val:.1f}×)...")
                enhancer = ImageEnhance.Sharpness(scaled_img)
                sharpened_img = enhancer.enhance(sharpness_val)
            else:
                sharpened_img = scaled_img

            elapsed = time.time() - start_time
            sharp_note = f" + Sharpness ({sharpness_val:.1f}×)" if has_sharpness else ""
            summary = (
                f"Regular Scaled (Lanczos {orig_w}×{orig_h} -> {target_w}×{target_h}){sharp_note} ({elapsed:.2f}s)"
            )
            return sharpened_img, True, summary

        # Handle 'auto' upscale mode:
        # Trigger AI upscale only if original image is less than 80% (< 0.80) of target resolution.
        # If original image is >= 80% of target resolution, perform regular image rescale.
        if upscale_mode == "auto" and size_ratio >= 0.80:
            start_time = time.time()
            if progress_callback:
                progress_callback(
                    f"Auto Upscale: Page size ({orig_w}×{orig_h}) is {size_ratio*100:.1f}% of target resolution (>= 80%). "
                    f"Performing regular algorithmic rescale..."
                )

            if (target_w, target_h) != (orig_w, orig_h):
                scaled_img = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                scaled_img = image.copy()

            sharpness_val = max(0.0, sharpness)
            has_sharpness = (sharpness_val > 0.0 and abs(sharpness_val - 1.0) >= 0.01)
            if has_sharpness:
                if progress_callback:
                    progress_callback(f"Applying post-scale sharpness filter ({sharpness_val:.1f}×)...")
                enhancer = ImageEnhance.Sharpness(scaled_img)
                sharpened_img = enhancer.enhance(sharpness_val)
            else:
                sharpened_img = scaled_img

            elapsed = time.time() - start_time
            sharp_note = f" + Sharpness ({sharpness_val:.1f}×)" if has_sharpness else ""
            summary = (
                f"Auto Scaled: Regular Lanczos ({orig_w}×{orig_h} -> {target_w}×{target_h}, {size_ratio*100:.1f}% >= 80%){sharp_note} ({elapsed:.2f}s)"
            )
            return sharpened_img, False, summary

        # Load AI neural model
        target_device, dev_name = self.resolve_device(upscale_device)
        model_key = upscale_model if upscale_model in MODEL_REGISTRY else "2x_Text2HD"

        # Intelligent model selection for 'neural_downsample':
        # If 8x model is requested, but the upscale factor needed to reach step-up resolution is <= 3.0x
        # (or 2x is already enough to reach target resolution), intelligently select 2x_Text2HD.
        # This avoids generating a massive 8K (8192px) tensor that takes 30s+ and memory bloat,
        # while delivering identical or superior crisp typography in ~4-5s.
        if upscale_mode == "neural_downsample" and model_key == "8x_NMKD":
            req_scale = max(step_up_w / orig_w, step_up_h / orig_h)
            if (req_scale <= 3.0 or orig_w * 2 >= target_w or orig_h * 2 >= target_h) and "2x_Text2HD" in MODEL_REGISTRY:
                model_key = "2x_Text2HD"
                if progress_callback:
                    progress_callback(
                        f"Intelligently selected 2x Text2HD: {req_scale:.1f}× upscale needed for {step_up_tier}p — 2x is optimal without 8x over-scaling..."
                    )

        descriptor = self.load_model(model_key, device=target_device, progress_callback=progress_callback)
        model_info = MODEL_REGISTRY[model_key]
        scale = getattr(descriptor, "scale", model_info.get("scale", 2))

        work_img = image
        down_summary = ""

        # Handle 'downsample_neural' mode:
        # Downscale to lower resolution tier:
        # - Downscale to 720p if image size > 720p and < 1080p
        # - Downscale to 1080p if image size > 1080p and < 1440p
        # - Downscale to 1440p if image size > 1440p and < 4k
        # - Downscale to 4k if image size > 4k
        if upscale_mode == "downsample_neural":
            b4k_w = RES_LIMITS["4k"]["max"] if is_landscape else RES_LIMITS["4k"]["min"]
            b4k_h = RES_LIMITS["4k"]["min"] if is_landscape else RES_LIMITS["4k"]["max"]
            b1440_w = RES_LIMITS["1440"]["max"] if is_landscape else RES_LIMITS["1440"]["min"]
            b1440_h = RES_LIMITS["1440"]["min"] if is_landscape else RES_LIMITS["1440"]["max"]
            b1080_w = RES_LIMITS["1080"]["max"] if is_landscape else RES_LIMITS["1080"]["min"]
            b1080_h = RES_LIMITS["1080"]["min"] if is_landscape else RES_LIMITS["1080"]["max"]
            b720_w = RES_LIMITS["720"]["max"] if is_landscape else RES_LIMITS["720"]["min"]
            b720_h = RES_LIMITS["720"]["min"] if is_landscape else RES_LIMITS["720"]["max"]

            down_tier = None
            if orig_w > b4k_w or orig_h > b4k_h:
                down_tier = "4k"
            elif orig_w > b1440_w or orig_h > b1440_h:
                down_tier = "1440"
            elif orig_w > b1080_w or orig_h > b1080_h:
                down_tier = "1080"
            elif orig_w > b720_w or orig_h > b720_h:
                down_tier = "720"

            if down_tier:
                d_cfg = RES_LIMITS[down_tier]
                dw_bound = d_cfg["max"] if is_landscape else d_cfg["min"]
                dh_bound = d_cfg["min"] if is_landscape else d_cfg["max"]
                down_scale = min(dw_bound / orig_w, dh_bound / orig_h)
                dw = max(1, round(orig_w * down_scale))
                dh = max(1, round(orig_h * down_scale))
                if (dw, dh) != (orig_w, orig_h):
                    if progress_callback:
                        progress_callback(f"Pre-downscaling image from {orig_w}×{orig_h} to {down_tier}p ({dw}×{dh})...")
                    work_img = image.resize((dw, dh), Image.Resampling.LANCZOS)
                    down_summary = f"Pre-downscaled {orig_w}×{orig_h} -> {dw}×{dh} ({down_tier}p)"

        start_time = time.time()
        if progress_callback:
            progress_callback(f"Running neural inference with {model_info['name']} on {dev_name}...")

        import torch
        import numpy as np

        # Convert PIL RGB to Torch Tensor (1, 3, H, W) normalized to [0, 1]
        rgb_np = np.array(work_img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(target_device)

        # Determine tile size: 512 for CUDA, 256 for CPU
        is_cuda = target_device.type == "cuda"
        tile_size = 512 if is_cuda else 256
        tile_pad = 24 if is_cuda else 16

        # Execute tiled or single-pass inference with graceful CUDA OOM fallback
        try:
            out_tensor, tile_summary = self._tiled_inference(
                descriptor,
                tensor,
                tile_mode=tile_mode,
                tile_count=tile_count,
                tile_size=tile_size,
                tile_pad=tile_pad,
                progress_callback=progress_callback
            )
        except (getattr(torch.cuda, "OutOfMemoryError", RuntimeError), RuntimeError) as oom_err:
            err_msg = str(oom_err).lower()
            if "out of memory" in err_msg or "cuda" in err_msg:
                if progress_callback:
                    progress_callback("GPU VRAM exhausted on single-tile pass. Falling back to multi-tile inference...")
                if is_cuda:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                out_tensor, tile_summary = self._tiled_inference(
                    descriptor,
                    tensor,
                    tile_mode="multi",
                    tile_count=tile_count if tile_count not in (1, "1") else "auto",
                    tile_size=tile_size,
                    tile_pad=tile_pad,
                    progress_callback=progress_callback
                )
                tile_summary += " (OOM Fallback)"
            else:
                raise

        # Apply ink hole healing for 2x_Text2HD (or if anti_dither explicitly requested)
        healed_note = ""
        if anti_dither and model_key == "2x_Text2HD":
            out_tensor = self._heal_text_pinholes(out_tensor, kernel_size=3)
            healed_note = " + Ink Hole Healing"

        # Convert back to PIL
        out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        upscaled_img = Image.fromarray(out_np, mode="RGB")
        elapsed = time.time() - start_time

        if is_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        up_w, up_h = upscaled_img.size

        # Mode 6: Neural Upscale then Downsample - downsample to target_res boundary
        if upscale_mode == "neural_downsample":
            # Ensure output is at or higher than the one-step-above tier
            if up_w < step_up_w or up_h < step_up_h:
                step_up_img = upscaled_img.resize((step_up_w, step_up_h), Image.Resampling.LANCZOS)
            else:
                step_up_img = upscaled_img

            final_img = step_up_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            summary = (
                f"Neural Upscale then Downsample: AI {model_info['name']} [{tile_summary}]{healed_note} "
                f"({orig_w}×{orig_h} -> {up_w}×{up_h}, >= {step_up_tier}p) + Lanczos downsampled to {target_w}×{target_h} ({target_res}p) "
                f"({elapsed:.2f}s, {dev_name})"
            )
            return final_img, True, summary

        # Mode 5: Downsample then Neural Upscale - scale neural output to target_res boundary
        if upscale_mode == "downsample_neural":
            if (target_w, target_h) != (up_w, up_h):
                final_img = upscaled_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                final_img = upscaled_img
            pre_note = f"{down_summary} -> " if down_summary else ""
            summary = (
                f"Downsample then Neural Upscale: {pre_note}AI {model_info['name']} [{tile_summary}]{healed_note} "
                f"({up_w}×{up_h}) -> Lanczos {target_w}×{target_h} ({elapsed:.2f}s, {dev_name})"
            )
            return final_img, True, summary

        # Mode 1: Auto Neural Upscale (when image was < 80% of target resolution)
        if upscale_mode == "auto":
            if (target_w, target_h) != (up_w, up_h):
                final_img = upscaled_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                final_img = upscaled_img
            summary = (
                f"Auto Neural Upscale: {orig_w}×{orig_h} ({size_ratio*100:.1f}% of target < 80%) -> "
                f"AI {model_info['name']} [{tile_summary}]{healed_note} ({up_w}×{up_h}) -> Lanczos {target_w}×{target_h} "
                f"({elapsed:.2f}s, {dev_name})"
            )
            return final_img, True, summary

        # Mode 4: Always Neural Upscale
        if (target_w, target_h) != (up_w, up_h):
            final_img = upscaled_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            final_img = upscaled_img
        summary = (
            f"Always Neural Upscale: AI {model_info['name']} [{tile_summary}]{healed_note}: {orig_w}×{orig_h} -> "
            f"{up_w}×{up_h} -> Lanczos {target_w}×{target_h} ({elapsed:.2f}s, {dev_name})"
        )
        return final_img, True, summary
