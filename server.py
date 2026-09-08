import os
import sys
import json
import uuid
import queue
import threading
import mimetypes
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import email
from PIL import Image
import io
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.pipeline import DocumentPipeline
from core.ocr_engine import OCREngine
from core.upscaler import UpscaleEngine
TASKS: dict[str, dict] = {}

class ChainPDFHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def log_message(self, format, *args):
        # Suppress noisy SSE logging to keep terminal clean
        try:
            msg = format % args
            if "/api/events" in msg or "favicon.ico" in msg:
                return
            sys.stdout.write(f"[{self.log_date_time_string()}] {msg}\n")
        except Exception:
            pass

    def end_headers(self):
        # Enable CORS for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/chainpdf.html"
        return super().do_HEAD()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.path = "/chainpdf.html"
            return super().do_GET()

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path == "/api/status":
            self.handle_api_status()
            return

        if path == "/api/events":
            task_id = query.get("task_id", [None])[0]
            self.handle_api_events(task_id)
            return

        if path == "/api/download":
            task_id = query.get("task_id", [None])[0]
            self.handle_api_download(task_id)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/process":
            self.handle_api_process()
            return

        self.send_error(404, "Endpoint not found")

    def handle_api_status(self):
        ocr = OCREngine(PROJECT_ROOT)
        upscaler = UpscaleEngine(PROJECT_ROOT)
        hw = upscaler.get_hardware_info()
        data = {
            "status": "ready",
            "python_version": sys.version,
            "tesseract_available": ocr.is_available(),
            "tesseract_path": ocr.tesseract_cmd,
            "tessdata_dir": ocr.tessdata_dir,
            "languages": ocr.get_available_languages(),
            "upscaler_available": upscaler.is_available(),
            "cuda_available": hw.get("cuda_available", False),
            "gpu_device_name": hw.get("device_name", "CPU"),
            "available_devices": hw.get("available_devices", ["gpu", "cpu"] if hw.get("cuda_available") else ["cpu"]),
            "default_device": hw.get("default_device", "gpu" if hw.get("cuda_available") else "cpu"),
            "torch_version": hw.get("torch_version"),
            "models": hw.get("models", {})
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def handle_api_process(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length <= 0:
            self.send_error(400, "Empty request body")
            return

        raw_body = self.rfile.read(content_length)

        # Parse multipart form data
        msg_bytes = b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + raw_body
        msg = email.message_from_bytes(msg_bytes)

        files = []
        options = {
            "target_res": "1080",
            "auto_crop": True,
            "lang": "eng",
            "upscale_mode": "disabled",
            "upscale_model": "2x_Text2HD",
            "upscale_device": "gpu",
            "anti_dither": True
        }

        if msg.is_multipart():
            payload = msg.get_payload()
            if isinstance(payload, list):
                for part in payload:
                    if not isinstance(part, email.message.Message):
                        continue
                    cd = part.get("Content-Disposition", "")
                    name = part.get_param("name", header="content-disposition")
                    filename = part.get_filename()

                    if filename:
                        file_bytes = part.get_payload(decode=True)
                        if isinstance(file_bytes, (bytes, bytearray)):
                            try:
                                img = Image.open(io.BytesIO(file_bytes))
                                img.load()  # Ensure image is decoded in RAM
                                files.append({"name": filename, "image": img})
                            except Exception as e:
                                print(f"[WARN] Failed to decode image {filename}: {e}")
                    elif name in options:
                        raw_val = part.get_payload(decode=True)
                        val = (
                            raw_val.decode("utf-8").strip()
                            if isinstance(raw_val, (bytes, bytearray))
                            else str(raw_val).strip()
                        )
                        if name in ("auto_crop", "anti_dither"):
                            options[name] = (val.lower() in ("true", "1", "enabled", "on"))
                        else:
                            options[name] = val

        if not files:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "No valid images received"}).encode("utf-8"))
            return

        # Sort files naturally
        def natural_key(item):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', item["name"])]

        files.sort(key=natural_key)

        task_id = str(uuid.uuid4())
        event_q = queue.Queue()
        TASKS[task_id] = {
            "queue": event_q,
            "status": "processing",
            "pdf_bytes": None,
            "error": None
        }

        # Background processing thread
        def run_task():
            pipeline = DocumentPipeline(PROJECT_ROOT)
            try:
                def cb(evt):
                    event_q.put(evt)

                pdf_bytes = pipeline.process_batch(
                    images_data=files,
                    target_res=options.get("target_res", "1080"),
                    auto_crop=options.get("auto_crop", True),
                    lang=options.get("lang", "eng"),
                    upscale_mode=options.get("upscale_mode", "disabled"),
                    upscale_model=options.get("upscale_model", "2x_Text2HD"),
                    upscale_device=options.get("upscale_device", "gpu"),
                    anti_dither=options.get("anti_dither", True),
                    event_callback=cb
                )
                TASKS[task_id]["pdf_bytes"] = pdf_bytes
                TASKS[task_id]["status"] = "completed"
                event_q.put({"type": "finished", "task_id": task_id})
            except Exception as e:
                import traceback
                traceback.print_exc()
                TASKS[task_id]["status"] = "error"
                TASKS[task_id]["error"] = str(e)
                event_q.put({"type": "error", "message": str(e)})

        t = threading.Thread(target=run_task, daemon=True)
        t.start()

        # Return task_id to client immediately
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "task_id": task_id,
            "files_count": len(files),
            "status": "processing"
        }).encode("utf-8"))

    def handle_api_events(self, task_id: str | None):
        if not task_id or task_id not in TASKS:
            self.send_error(404, "Task not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        task = TASKS[task_id]
        q = task["queue"]

        while True:
            try:
                evt = q.get(timeout=25.0)
                data_str = json.dumps(evt)
                msg = f"data: {data_str}\n\n".encode("utf-8")
                self.wfile.write(msg)
                self.wfile.flush()

                if evt.get("type") in ("finished", "error"):
                    break
            except queue.Empty:
                # Keep-alive heartbeat comment
                try:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                except Exception:
                    break
            except Exception:
                break

    def handle_api_download(self, task_id: str | None):
        if not task_id or task_id not in TASKS or TASKS[task_id].get("pdf_bytes") is None:
            self.send_error(404, "PDF not ready or task not found")
            return

        pdf_bytes = TASKS[task_id]["pdf_bytes"]
        filename = f"ChainPDF_Searchable_{task_id[:8]}.pdf"

        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(pdf_bytes)))
        self.end_headers()
        self.wfile.write(pdf_bytes)

def run_server(port=8000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, ChainPDFHandler)
    print(f"============================================================")
    print(f"  ChainPDF Local Python Server Active")
    print(f"  URL: http://localhost:{port}/")
    print(f"  Embedded Engine: {sys.executable}")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port)
