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
        upscale_mode: str = "auto_1080",
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
        Returns: (resulting_image, was_upscaled, summary_message)
        """
        if upscale_mode == "disabled" or target_res == "original":
            return image, False, "Upscale disabled (Pass-through)"

        orig_w, orig_h = image.size
        is_landscape = orig_w >= orig_h

        # Handle 'regular' upscale mode: pure algorithmic scaling + post-scale sharpness filter
        if upscale_mode == "regular":
            start_time = time.time()
            if progress_callback:
                progress_callback("Running regular algorithmic scaling (Lanczos)...")

            if target_res == "original":
                target_w, target_h = orig_w * 2, orig_h * 2
            else:
                limits = {
                    "720": {"max": 1280, "min": 720},
                    "1080": {"max": 1920, "min": 1080},
                    "4k": {"max": 3840, "min": 2160}
                }
                config = limits.get(target_res, limits["1080"])
                bound_w = config["max"] if is_landscape else config["min"]
                bound_h = config["min"] if is_landscape else config["max"]
                scale = min(bound_w / orig_w, bound_h / orig_h)
                target_w = max(1, round(orig_w * scale))
                target_h = max(1, round(orig_h * scale))

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

        # Evaluate conditional logic (< 1080p only)
        if upscale_mode == "auto_1080":
            # 1080p standard boundary: 1920x1080 (landscape) or 1080x1920 (portrait)
            max_bound = 1920
            min_bound = 1080
            bound_w = max_bound if is_landscape else min_bound
            bound_h = min_bound if is_landscape else max_bound

            # If dimensions already meet or exceed 1080p boundary, skip upscaling
            if (orig_w >= bound_w and orig_h >= bound_h) or (orig_w >= max_bound or orig_h >= max_bound):
                return image, False, f"Page resolution ({orig_w}×{orig_h}) meets/exceeds 1080p threshold. Upscale bypassed."

        target_device, dev_name = self.resolve_device(upscale_device)
        model_key = upscale_model if upscale_model in MODEL_REGISTRY else "2x_Text2HD"
        descriptor = self.load_model(model_key, device=target_device, progress_callback=progress_callback)
        model_info = MODEL_REGISTRY[model_key]
        scale = getattr(descriptor, "scale", model_info.get("scale", 2))

        # Handle 'enhanced_1080' mode:
        # Pre-calibrate input size so that upscaling hits ~4K maximum bounds (3840x2160)
        # then downsamples cleanly with Lanczos to 1080p.
        work_img = image
        if upscale_mode == "enhanced_1080":
            target_4k_max = 3840
            target_4k_min = 2160
            bound_4k_w = target_4k_max if is_landscape else target_4k_min
            bound_4k_h = target_4k_min if is_landscape else target_4k_max

            # If image * scale would far exceed 4K, scale input down beforehand
            if scale is not None:
                if orig_w * scale > bound_4k_w or orig_h * scale > bound_4k_h:
                    pre_scale = min(bound_4k_w / (orig_w * scale), bound_4k_h / (orig_h * scale))
                    pre_w = max(64, round(orig_w * pre_scale))
                    pre_h = max(64, round(orig_h * pre_scale))
                    work_img = image.resize((pre_w, pre_h), Image.Resampling.BILINEAR)

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

        # For 'enhanced_1080', downscale to 1080p boundary
        if upscale_mode == "enhanced_1080":
            target_1080_max = 1920
            target_1080_min = 1080
            bound_1080_w = target_1080_max if is_landscape else target_1080_min
            bound_1080_h = target_1080_min if is_landscape else target_1080_max

            scale_down = min(bound_1080_w / up_w, bound_1080_h / up_h)
            final_w = max(1, round(up_w * scale_down))
            final_h = max(1, round(up_h * scale_down))

            final_img = upscaled_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
            summary = (
                f"Enhanced 1080p: Neural 4K upscale ({orig_w}×{orig_h} -> {up_w}×{up_h}) "
                f"[{tile_summary}]{healed_note} + Lanczos supersampled to {final_w}×{final_h} ({elapsed:.2f}s, {dev_name})"
            )
            return final_img, True, summary

        summary = (
            f"AI Upscaled with {model_info['name']} [{tile_summary}]{healed_note}: {orig_w}×{orig_h} -> {up_w}×{up_h} "
            f"({elapsed:.2f}s, {dev_name})"
        )
        return upscaled_img, True, summary
