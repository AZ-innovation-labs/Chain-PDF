# ChainPDF — Local & Offline Searchable OCR Studio

[![Privacy: 100% Air-Gapped](https://img.shields.io/badge/Privacy-100%25%20Air--Gapped-00f0ff?style=flat-square&logo=shield)](file:///e:/PROJECTS/HTML/Chain-PDF/chainpdf.html)
[![Engine: SIMD LSTM WebAssembly](https://img.shields.io/badge/Engine-SIMD%20LSTM%20WASM-7928ca?style=flat-square)](file:///e:/PROJECTS/HTML/Chain-PDF/chainpdf.html)
[![Zero Cloud Dependency](https://img.shields.io/badge/Cloud%20Dependency-Zero%20(Client--Side)-10b981?style=flat-square)](file:///e:/PROJECTS/HTML/Chain-PDF/chainpdf.html)

**ChainPDF** is a browser-based, client-side document processing suite that converts batches of raw scanned images into fully searchable, selectable, and archivable PDF documents.

Running completely inside the client browser via **WebAssembly (WASM)** and **Web Workers**, ChainPDF processes sensitive records, books, legal documents, and receipts with zero server communication — guaranteeing **100% data privacy and offline air-gapped security**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Processing Pipeline](#architecture--processing-pipeline)
- [Modules & Dependencies](#modules--dependencies)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1-Click Launch (Recommended)](#1-click-launch-recommended)
  - [Offline Asset Download](#offline-asset-download)
  - [Multi-Language Setup](#multi-language-setup)
- [Technical Capabilities](#technical-capabilities)
  - [Adaptive Ink-Contrast Extraction](#adaptive-ink-contrast-extraction)
  - [Border Auto-Cropping](#border-auto-cropping)
  - [Column & Flow Separation](#column--flow-separation)
  - [Calibrated Invisible Text Overlay](#calibrated-invisible-text-overlay)
- [Browser Compatibility](#browser-compatibility)
- [License](#license)

---

## Overview

Traditional OCR utilities typically rely on external cloud APIs (Google Cloud Vision, AWS Textract, Azure Document Intelligence), exposing confidential papers to third-party servers and network latency.

**ChainPDF** brings high-performance optical character recognition directly to the browser. Utilizing hardware-accelerated **SIMD WebAssembly**, multi-threaded Web Workers, and computer vision canvas filters, it pairs the raw visual fidelity of scanned pages with an invisibly embedded, perfectly aligned text layer. The generated PDFs allow instant full-text searching, copying, and indexing across Adobe Acrobat, Chrome, Firefox, Preview, and document search systems.

---

## Key Features

- 100% Air-Gapped & Local Execution**: Zero API keys, zero network telemetry, and no file uploads. All processing takes place inside client RAM.
- SIMD LSTM Neural Engine**: Accelerated WebAssembly LSTM (Long Short-Term Memory) neural network OCR for high-speed offline inference.
- **Smart Batch Ingestion**:
  - Drag-and-drop entire folders or multiple image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
  - Recursive folder scanning via HTML5 DataTransfer API.
  - **Natural Alphanumeric Sorting** (e.g., `Page_1.jpg`, `Page_2.jpg` ... `Page_10.jpg` in correct sequential order).
- **Automatic Border Detection & Crop**: Scans edge baseline uniformity to detect and trim scanner frames, black borders, and extraneous margins.
- **Adaptive Ink-Contrast Isolation**: Pre-OCR computer vision canvas filter enhances faded or colored inks while cleaning tinted, yellowed, or noisy scan backgrounds.
- **Multi-Column Reading Order Alignment**: Histogram gutter analysis separates multi-column page layouts (e.g. newspapers, magazines, academic papers), ensuring text selection reads naturally down column 1 before continuing to column 2.
- **Exact Invisible Text Layer Injection**:
  - Dynamically matches bounding boxes, baseline offsets, and glyph widths using `Helvetica` typography.
  - Renders transparent selectable text (`opacity: 0`) directly over the visual bitmap.
- **Cyberpunk Cockpit HUD & Dual-Mode Monitor**:
  - Live inspection viewport toggling between the **Visual Frame** and the processed **OCR Ink Map**..
  - Built-in Engine Execution Terminal for live event logs.
- **Resolution Presets**: Target output scaling for 720p (Compact), 1080p (Recommended), 4K (Maximum Detail), or Original unscaled dimensions.

---

## Architecture & Processing Pipeline

```
[ Image Ingestion ]
  │  Drag-and-drop folder or files (JPG, PNG, WEBP, BMP)
  ▼
[ Natural Alphanumeric Sort ]
  │  Pages ordered numerically (1, 2, ... 9, 10, 11)
  ▼
[ Computer Vision Edge Detection ]
  │  Automatic border & crop margin isolation
  ▼
[ Canvas Dimension Normalization ]
  │  Resized to selected profile (720p / 1080p / 4K / Original)
  ▼
[ Pre-OCR Ink-Contrast Filter ]
  │  Luminance weighting & channel thresholding for high OCR accuracy
  ▼
[ Tesseract SIMD LSTM Worker ]
  │  WASM neural OCR extracts bounding boxes, lines, and words
  ▼
[ Post-OCR Geometric Alignment ]
  │  ├── Character sanitization (quotes, hyphens, OCR artifacts)
  │  ├── Word re-stitching across display font kerning gaps
  │  └── Multi-column gutter binning & reading-order sorting
  ▼
[ PDF-Lib Document Compilation ]
  │  ├── Embeds compressed visual image layer
  │  └── Injects calibrated invisible font overlay (opacity: 0)
  ▼
[ Searchable PDF Auto-Download ]
```

---

## Modules & Dependencies

ChainPDF is built using vanilla web technologies and standalone local libraries with no external build step or npm bundler required:

| Module / Component | Version / File | Role & Purpose |
|-------------------|----------------|----------------|
| **[pdf-lib](https://pdf-lib.js.org/)** | `pdf-lib.min.js` (v1.17.1) | Pure JavaScript library for creating PDF documents client-side. Embeds image canvases as JPEG streams and injects the calibrated invisible font text layers. |
| **[Tesseract.js](https://tesseract.projectnaptha.com/)** | `tesseract.min.js` (v5.1.1) | JavaScript OCR bridge that creates, communicates with, and controls asynchronous background recognition workers. |
| **Tesseract Worker** | `worker.min.js` (v5.1.1) | Dedicated Web Worker runtime executing OCR jobs in a separate browser thread to prevent UI freezing. |
| **Tesseract Core SIMD WASM** | `tesseract-core-simd-lstm.wasm.js` & `tesseract-core-simd-lstm.wasm` (v5.1.1) | Low-level WebAssembly binary compiled with SIMD vectorization and LSTM neural network inference. |
| **Tessdata Language Models** | `tessdata/eng.traineddata.gz` (Naptha 4.0.0 Fast) | Trained neural network character models and dictionary weights for English text extraction. |
| **Modern Typography** | `Outfit`, `Inter`, `Roboto Mono` | Google Fonts for clean interface presentation, telemetry readouts, and monospace engine terminal logs (with local fallback). |

---

## Project Structure

```
Chain-PDF/
├── chainpdf.html                     # Complete single-page studio application (UI, CSS, Engine)
├── pdf-lib.min.js                    # Local standalone PDF-Lib build (v1.17.1)
├── tesseract.min.js                  # Local Tesseract.js client bridge (v5.1.1)
├── worker.min.js                     # Local Tesseract Web Worker (v5.1.1)
├── tesseract-core-simd-lstm.wasm.js  # Tesseract WASM JavaScript glue code
├── tesseract-core-simd-lstm.wasm     # High-speed SIMD WebAssembly neural core
├── tessdata/                         # Language models directory
│   └── eng.traineddata.gz            # English OCR trained neural data
├── download_assets.bat               # Automation script to fetch or update offline assets
├── launch.bat                        # One-click launcher with Python HTTP server fallback
├── script.js                         # Minimal reference script demonstrating offline OCR
└── README.md                         # Documentation
```

---

## Getting Started

### Prerequisites

- Any modern web browser supporting **WebAssembly** and **Web Workers** (Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, Safari).
- *(Recommended)* **Python 3.x** installed to run the local HTTP server.

> [!NOTE]
> Because Web Workers and WebAssembly files are subject to browser security restrictions (CORS and origin isolation) when opened via the `file://` protocol, running via a local HTTP server is strongly recommended.

### 1-Click Launch (Recommended)

Simply double-click:
```cmd
launch.bat
```
This batch script will:
1. Check for a local Python installation.
2. If Python is available, spin up a lightweight local server on `http://localhost:8000/chainpdf.html` and launch your default browser.
3. If Python is not installed, it falls back to launching the file directly.

### Alternative Manual Launch

If you prefer starting your own local server:

**Using Python:**
```bash
python -m http.server 8000
# Open http://localhost:8000/chainpdf.html in your browser
```

**Using Node.js (npx):**
```bash
npx serve .
# Or: npx http-server . -p 8000
```

### Offline Asset Download

If you ever need to refresh or download the offline vendor files and language models, execute:
```cmd
download_assets.bat
```
This script downloads `pdf-lib.min.js`, `tesseract.min.js`, `worker.min.js`, `tesseract-core-simd-lstm.wasm`, and `eng.traineddata.gz` directly into your workspace.

### Multi-Language Setup

By default, ChainPDF includes English (`eng`). To enable additional languages (e.g. Spanish, French, German):

1. Download the corresponding `*.traineddata.gz` file (Naptha fast 4.0.0 format):
   - Spanish: `https://github.com/naptha/tessdata/raw/gh-pages/4.0.0_fast/spa.traineddata.gz`
   - French: `https://github.com/naptha/tessdata/raw/gh-pages/4.0.0_fast/fra.traineddata.gz`
   - German: `https://github.com/naptha/tessdata/raw/gh-pages/4.0.0_fast/deu.traineddata.gz`
2. Place the file inside the `tessdata/` directory (e.g., `tessdata/spa.traineddata.gz`).
3. Select the desired language in the **OCR Language** dropdown in the ChainPDF cockpit.

---

## Technical Capabilities

### Adaptive Ink-Contrast Extraction
Scanned pages often suffer from faint pencil marks, colored ink on tinted parchment, or dark scanner shadows. ChainPDF processes every frame prior to OCR with a specialized pixel shader:
$$\text{Luminance} = 0.299R + 0.587G + 0.114B$$
$$\text{Val} = 0.75 \times \min(R, G, B) + 0.25 \times \text{Luminance}$$
Pixels above the background threshold are washed out to pure white ($255$), while ink pixels below the shadow threshold are intensified, yielding clear, high-contrast letterforms for the neural network.

### Border Auto-Cropping
Detects uniform border margins (black scanner edges, white margins) by calculating standard deviation across edge scanlines. If a continuous corner-to-corner border transition line is detected within $25\%$ of the image perimeter, the content box is trimmed inward to isolate the document text.

### Column & Flow Separation
Standard OCR tools read multi-column documents horizontally across the entire page width, corrupting the paragraph sequence. ChainPDF partitions the page horizontally into discrete $4\text{px}$ bins to detect empty gutters. Content bounded by gutters is grouped into independent columns and sorted top-to-bottom, keeping banners and side-by-side columns in natural reading sequence.

### Calibrated Invisible Text Overlay
To ensure selecting text in a PDF viewer matches the visual ink on screen:
1. Bounding box coordinates $[x_0, y_0, x_1, y_1]$ are scaled to the target PDF canvas.
2. The standard font unit width is measured via `helveticaFont.widthOfTextAtSize(text, 1)`.
3. Font size is computed proportionally:
   $$\text{FontSize} = \frac{\text{TargetBoxWidth}}{\text{UnitWidth}}$$
4. The text baseline is compensated:
   $$Y_{\text{baseline}} = (\text{PageHeight} - y_1) + (\text{LineHeight} \times 0.21)$$
5. Characters are placed with `opacity: 0`, creating an invisible text mask over the underlying image.

---

## Browser Compatibility

| Browser | Supported | Minimum Recommended Version |
|---------|:---------:|-----------------------------|
| **Google Chrome** | ✅ | 84+ (SIMD WASM support) |
| **Microsoft Edge** | ✅ | 84+ (Chromium) |
| **Mozilla Firefox** | ✅ | 89+ (SIMD WASM support) |
| **Brave** | ✅ | Latest |
| **Apple Safari** | ✅ | 16.4+ (WebAssembly SIMD) |

---

## License

This project is licensed under the [MIT License](LICENSE).
Third-party libraries (`pdf-lib`, `tesseract.js`, and `tesseract.js-core`) are distributed under their respective open-source licenses (Apache-2.0 / MIT).
