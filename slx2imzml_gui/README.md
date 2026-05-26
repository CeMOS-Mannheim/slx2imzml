# SCiLS Exporter — Flask/HTML5 GUI

A clean, single-page web application that replaces the original PyShiny GUI.  
Built with **Flask** (Python backend) + **HTML5 / Vanilla JS / Plotly.js** (frontend).

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3 |
| Template engine | Jinja2 (bundled with Flask) |
| Frontend | Vanilla JS (ES2022), no framework |
| Charts | Plotly.js (CDN, slim build) |
| Styling | Hand-crafted CSS design system (no Bootstrap) |

## Features

* **Browse** — opens a native Windows file dialog to pick an `.slx` file.  
* **Region Tree** — colour-coded table with multi-select (click + Shift-click), Select All / Clear.  
* **Region Map** — interactive Plotly map; selected regions are highlighted with fill + thicker border.  
* **Feature Lists** — selectable table; selecting a list loads per-feature details automatically.  
* **Feature Details** — merged, de-duplicated table with computed `m/z center`, `ppm width`, and optional CCS columns.  
* **Normalizations** — informational sidebar list.  
* **Advanced Options** — slice thickness (µm) and *Include CCS Values* toggle.  
* **Start Export** — writes a JSON config and calls `python -m slx2imzml` in a subprocess.  
* Toast notifications for all async operations.

## Installation

```bash
# From the project root:
pip install flask matplotlib Pillow numpy
# (scilslab and slx2imzml must already be available)
```

## Running

```bash
python slx2imzml_gui/launcher.py
```

Or directly:

```bash
python slx2imzml_gui/app.py
```

The app opens automatically at `http://127.0.0.1:5001`.
