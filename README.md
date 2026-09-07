# ChainPDF — Local & Offline Searchable OCR Studio

[![Privacy: 100% Air-Gapped](https://img.shields.io/badge/Privacy-100%25%20Air--Gapped-00f0ff?style=flat-square&logo=shield)](file:///e:/PROJECTS/HTML/Chain-PDF/chainpdf.html)
[![Engine: Portable Python + Tesseract](https://img.shields.io/badge/Engine-Portable%20Python%20%2B%20Tesseract-7928ca?style=flat-square)](file:///e:/PROJECTS/HTML/Chain-PDF/chainpdf.html)
[![Zero Cloud Dependency](https://img.shields.io/badge/Cloud%20Dependency-Zero%20(100%25%20Local)-10b981?style=flat-square)](file:///e:/PROJECTS/HTML/Chain-PDF/chainpdf.html)

**ChainPDF** is an air-gapped, fully portable document processing studio that converts batches of raw scanned images into fully searchable, selectable, and archivable PDF documents.

Running on a **self-contained embedded Python runtime** with a **native portable Tesseract OCR engine**, ChainPDF processes sensitive records, books, legal documents, and receipts with zero server telemetry or third-party cloud communication — guaranteeing **100% data privacy and offline security**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Processing Pipeline](#architecture--processing-pipeline)
- [Modules & Dependencies](#modules--dependencies)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [1-Click Launch (Recommended)](#1-click-launch-recommended)
  - [Automated Portable Environment Setup](#automated-portable-environment-setup)
- [Technical Capabilities](#technical-capabilities)
  - [Adaptive Ink-Contrast Extraction](#adaptive-ink-contrast-extraction)
  - [Border Auto-Cropping](#border-auto-cropping)
  - [Column & Flow Separation](#column--flow-separation)
  - [Calibrated Invisible Text Overlay](#calibrated-invisible-text-overlay)
- [License](#license)

---

## Overview

Traditional OCR utilities typically rely on external cloud APIs (Google Cloud Vision, AWS Textract, Azure Document Intelligence), exposing confidential papers to third-party servers and network latency.

**ChainPDF** executes all processing locally inside an embedded, portable Python environment:
- **No Node.js** or external system package managers required.
- **Portable Python 3.11 Runtime** bundled right within the project directory.
- **Native Tesseract OCR 5.4.0 binary** accelerated by AVX2/AVX vector instruction sets.
- **Modern Cyberpunk Cockpit UI** powered by pure Vanilla HTML5/CSS/JavaScript with real-time Server-Sent Events (SSE) telemetry, laser scanlines, and live dual-monitor preview frames.

---

## Key Features

- **100% Air-Gapped & Local Execution**: Zero API keys, zero cloud telemetry, and zero third-party file uploads. All processing takes place locally on your machine.
- **Embedded Portable Python Runtime**: Runs from a self-contained Python distribution without requiring global Python installation.
- **AVX2-Accelerated Tesseract OCR**: Native C++ Tesseract engine provides fast character recognition without browser memory limits.
- **Smart Batch Ingestion**:
  - Drag-and-drop entire folders or multiple image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
  - Recursive folder scanning via HTML5 DataTransfer API.
  - **Natural Alphanumeric Sorting** (e.g., `Page_1.jpg`, `Page_2.jpg` ... `Page_10.jpg` in sequential order).
- **Automatic Border Detection & Crop**: Vectorized edge baseline uniformity inspection detects and trims scanner frames and black borders.
- **Adaptive Ink-Contrast Isolation**: Pre-OCR computer vision filter enhances faded or colored inks while cleaning tinted, yellowed, or noisy scan backgrounds.
- **Multi-Column Reading Order Alignment**: Histogram gutter analysis separates multi-column page layouts (e.g. newspapers, magazines, academic papers), ensuring text selection reads naturally down column 1 before continuing to column 2.
- **Calibrated Invisible Text Layer Injection**:
  - Automatically matches bounding boxes, baseline offsets, and glyph widths using standard Helvetica typography.
  - Injects native invisible text (`render_mode=3`) directly over the visual bitmap using PyMuPDF.
- **Cyberpunk Cockpit HUD & Dual-Mode Monitor**:
  - Live inspection viewport toggling between the **Visual Frame** and the processed **OCR Ink Map**.
  - Built-in Engine Execution Terminal for live event logs streamed in real time via Server-Sent Events (SSE).
- **Resolution Presets**: Target output scaling for 720p (Compact), 1080p (Recommended), 4K (Maximum Detail), or Original unscaled dimensions.

---

## Architecture & Processing Pipeline

```mermaid
flowchart TD
    subgraph Client ["1. User Interface & Cockpit Layer (Vanilla JS)"]
        A["User Input (Folder / Multi-File Drag & Drop)"]
        B["Parameter Settings (Crop, Resolution, Language)"]
        HUD["Live Dual Monitor (Visual Canvas / OCR Ink Map)"]
        TERM["Terminal HUD (Real-time Telemetry & Logs)"]
        EVT["EventSource / SSE Listener"]
    end

    subgraph Server ["2. Local Server Controller (server.py)"]
        SRV["Threading HTTP & SSE Server (localhost:8000)"]
        API["Endpoints: /api/process, /api/events, /api/download"]
    end

    subgraph Pipeline ["3. Python Document Processing Engine (core/)"]
        PRE["Image Preprocessor (core/preprocessor.py)"]
        OCR["Native Tesseract OCR (core/ocr_engine.py)"]
        LAY["Spatial Layout & Column Engine (core/layout_analyzer.py)"]
        PDF["Searchable PDF Builder (core/pdf_builder.py)"]
    end

    subgraph Runtime ["4. Portable Environment"]
        EMB["python/ (Windows 64-bit Embeddable Python 3.11)"]
        TESS["tesseract/ (AVX2-Accelerated Tesseract 5.4.0 Binary)"]
        DATA["tessdata/ (Trained Neural Language Models)"]
    end

    A --> API
    B --> API
    API --> SRV
    SRV --> PRE
    PRE --> OCR
    OCR --> LAY
    LAY --> PDF
    SRV -->|Real-time Telemetry & Preview Stream| EVT
    EVT --> HUD
    EVT --> TERM
    PDF -->|Searchable PDF File| API

    EMB -->|Executes| SRV
    TESS -->|Invoked by| OCR
    DATA -->|Loaded by| OCR
```

---

## Modules & Dependencies

| Module / Component | Version / Location | Role & Purpose |
|-------------------|--------------------|----------------|
| **Embedded Python** | Python 3.11.9 (`python/`) | Portable Windows 64-bit Python runtime providing complete self-contained execution. |
| **Portable Tesseract** | Tesseract 5.4.0 (`tesseract/`) | High-speed C++ OCR engine with AVX2/AVX hardware optimization. |
| **PyMuPDF (`fitz`)** | v1.28.2 | C++ accelerated PDF compiler for embedding visual images and calibrated invisible text layers. |
| **Pillow (PIL)** | v12.3.0 | Image decoding, format conversions, and Lanczos resolution scaling. |
| **NumPy** | v2.2.3 | Vectorized array operations for scanner edge detection, auto-cropping, and adaptive ink-contrast filtering. |
| **PyTesseract** | v0.3.13 | Python bridge interface to the local Tesseract binary. |
| **Cockpit HUD** | `chainpdf.html` | Pure Vanilla HTML5/CSS/JavaScript interface (no Node.js, no bundlers, no npm). |

---

## Project Structure

```
Chain-PDF/
├── chainpdf.html           # Single-page cyberpunk cockpit UI (HTML, CSS, JS)
├── server.py               # Multi-threaded Python HTTP and SSE server
├── requirements.txt        # Python dependencies specification
├── launch.bat              # One-click studio launcher
├── setup_portable_env.bat  # Automated portable environment setup script
├── core/                   # Core Python document processing package
│   ├── __init__.py
│   ├── preprocessor.py     # Border auto-crop and adaptive ink-contrast filter
│   ├── ocr_engine.py       # Tesseract OCR engine wrapper
│   ├── layout_analyzer.py  # Word stitching and multi-column separation
│   ├── pdf_builder.py      # PyMuPDF searchable PDF builder
│   └── pipeline.py         # End-to-end processing pipeline orchestrator
├── python/                 # Embedded Python 3.11 runtime (created by setup)
├── tesseract/              # Portable Tesseract 5.4.0 binary (created by setup)
├── tessdata/               # OCR trained language models (eng.traineddata)
└── README.md               # Documentation
```

---

## Getting Started

### 1-Click Launch (Recommended)

Simply double-click:
```cmd
launch.bat
```
This batch script will:
1. Detect if the embedded Python runtime is ready. If not, it automatically runs `setup_portable_env.bat`.
2. Start the local Python server on `http://localhost:8000/`.
3. Open your default web browser directly to the ChainPDF cockpit.

### Automated Portable Environment Setup

To manually download or refresh the embedded Python environment and portable Tesseract binary, double-click:
```cmd
setup_portable_env.bat
```
The script downloads and configures the embedded Python package, library wheels, and Tesseract binary directly into your project folder.

---

## Technical Capabilities

### Adaptive Ink-Contrast Extraction
Scanned pages often suffer from faint pencil marks, colored ink on tinted parchment, or dark scanner shadows. ChainPDF processes every frame prior to OCR with a vectorized NumPy pixel transformation:
$$\text{Luminance} = 0.299R + 0.587G + 0.114B$$
$$\text{Val} = 0.75 \times \min(R, G, B) + 0.25 \times \text{Luminance}$$
Pixels above the background threshold ($>195$) are washed out to pure white ($255$), while ink pixels below the shadow threshold ($<165$) are intensified ($\times 0.55$), producing ultra-clean letterforms for Tesseract.

### Border Auto-Cropping
Detects uniform border margins by scanning edge baseline uniformity. If a continuous corner-to-corner border transition line is detected within $25\%$ of the image perimeter, the content box is trimmed inward to isolate the document text.

### Column & Flow Separation
Standard OCR tools read multi-column documents horizontally across the entire page width, corrupting the paragraph sequence. ChainPDF partitions the page horizontally into discrete $4\text{px}$ bins to detect empty gutters. Content bounded by gutters is grouped into independent columns and sorted top-to-bottom, keeping banners and side-by-side columns in natural reading sequence.

### Calibrated Invisible Text Overlay
To ensure selecting text in a PDF viewer matches the visual ink on screen:
1. Bounding box coordinates $[x_0, y_0, x_1, y_1]$ are scaled to the target PDF canvas.
2. Character widths are matched using standard Helvetica typography metrics.
3. The text baseline is compensated for font ascent:
   $$Y_{\text{baseline}} = y_0 + (\text{LineHeight} \times 0.79)$$
4. Characters are placed with `render_mode=3` (invisible text in PDF specification), creating a 100% searchable and selectable text layer over the visual bitmap.

---

## License

This project is licensed under the [MIT License](LICENSE).
Tesseract OCR is licensed under the Apache-2.0 License.
PyMuPDF is licensed under the GNU AGPL/Commercial license.
