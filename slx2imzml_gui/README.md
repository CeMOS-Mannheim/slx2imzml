# SCiLS Exporter PyShiny GUI

This directory contains the Python-based graphical user interface for the SCiLS-2-ImzML exporter.

## Installation and Setup

To ensure the GUI functions correctly, the following files must be present in your target project:

1.  **`slx2imzml_gui/app.py`**: The main PyShiny application.
2.  **`slx2imzml/__main__.py`**: (CRITICAL!) This file must be in the `slx2imzml` package folder to enable the `python -m slx2imzml` command used by the GUI.

### Install Dependencies

Run the following command from your project root:

```bash
pip install -r slx2imzml_gui/requirements.txt
```

Ensure the `slx2imzml` package is installed in "Editable Mode" (the app will attempt to do this automatically on startup):

```bash
pip install -e .
```

## Running the App

Execute the following command in your terminal:

```bash
python -m shiny run slx2imzml_gui/app.py
```

## Features

1.  **Browse**: Select a `.slx` file using a native Windows dialog.
2.  **Tables**: Mark the desired regions and feature lists for export.
3.  **Start**: Generates a configuration JSON and triggers the exporter in the background.
