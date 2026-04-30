import os
import json
import pathlib
import datetime
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import faicons as fa
import scilslab as sl

import sys
import subprocess
import importlib.util

# Add the parent directory to sys.path to import local modules
root_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_dir))

def ensure_package_installed():
    """Ensure the slx2imzml package is installed in editable mode."""
    if importlib.util.find_spec("slx2imzml") is None:
        print("slx2imzml package not found in current environment.")
        print(f"Installing in editable mode from: {root_dir}")
        try:
            # Use sys.executable to ensure we use the Correct pip (from the .venv)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(root_dir)])
            print("Successfully installed slx2imzml.")
        except Exception as e:
            print(f"FAILED to install slx2imzml automatically: {e}")
            print("Please run 'pip install -e .' manually in the project root.")

# Run check at startup
ensure_package_installed()

from slx2imzml.slxFileHelper import slxFileHelper

def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(filetypes=[("SCiLS Lab files", "*.slx")])
    root.destroy()
    return file_path

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.card(
            ui.card_header("Choose .slx file"),
            ui.input_action_button("btn_browse", "Browse...", icon=fa.icon_svg("folder-open")),
            ui.output_text("txt_path"),
        ),
        ui.card(
            ui.card_header("Start Export"),
            ui.input_action_button("btn_process", "Start", icon=fa.icon_svg("file-export"), class_="btn-primary", disabled=True),
            ui.output_text_verbatim("txt_selection_status"),
            ui.p("Click the button to start the export process. Ensure you have selected regions and feature lists."),
        ),
        title="SCiLS Exporter Controls",
    ),
    ui.layout_column_wrap(
        ui.card(
            ui.card_header("Region Tree"),
            ui.output_data_frame("region_table"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Feature Lists"),
            ui.output_data_frame("feature_table"),
            full_screen=True,
        ),
        width=1/2,
    ),
    title="SCiLS Exporter",
    fillable=True,
)

def server(input: Inputs, output: Outputs, session: Session):
    slx_path = reactive.Value(None)
    slx_regions = reactive.Value(pd.DataFrame())
    slx_feature_lists = reactive.Value(pd.DataFrame())
    
    # New: Cache selections to avoid SilentException in event handlers
    selected_regions_idx = reactive.Value(tuple())
    selected_features_idx = reactive.Value(tuple())
    
    @reactive.effect
    @reactive.event(input.btn_browse)
    def _():
        print("Browse button clicked")
        path = select_file()
        if path:
            print(f"Selected file: {path}")
            slx_path.set(path)
            
    @output
    @render.text
    def txt_path():
        p = slx_path()
        return p if p else "No file selected"

    @reactive.effect
    def _():
        path = slx_path()
        if not path or not os.path.exists(path):
            return

        ui.notification_show("Trying to access SCiLS file...", type="message")
        try:
            with sl.LocalSession(path) as slx_file:
                dataset = slx_file.dataset_proxy
                
                # Load Feature Lists
                fl = dataset.feature_table.get_feature_lists()
                # Rename columns to match R app style
                fl_display = fl.rename(columns={
                    "num_features": "nFeat",
                    "has_mz_features": "mzFeat",
                    "has_mobility_intervals": "mobilityIntervals",
                    "has_ccs_features": "ccsFeatures"
                })
                # Select only relevant columns
                if "has_external_features" in fl_display.columns:
                    fl_display = fl_display.drop(columns=["has_external_features"])
                slx_feature_lists.set(fl_display)
                
                # Load Regions
                region_tree = dataset.get_region_tree()
                all_regions = region_tree.get_all_regions()
                print(f"Found {len(all_regions)} regions in total")
                
                regions_data = []
                for r in all_regions:
                    # Skip regions with subregions (only leaf nodes)
                    if len(r.subregions) == 0:
                        # Get spot count
                        spots = dataset.get_region_spots(r.id)
                        regions_data.append({
                            "name": r.name,
                            "nPx": len(spots),
                            "subRegions": 0 # R app logic: length(x$polygons) - simplified here as leaf
                        })
                
                print(f"Loaded {len(regions_data)} leaf regions")
                slx_regions.set(pd.DataFrame(regions_data))
                
            ui.notification_show("SCiLS file access successful. Please choose regions and feature lists to export.", type="message")
        except Exception as e:
            ui.notification_show(f"Error accessing SCiLS file: {str(e)}", type="error")

    @output
    @render.data_frame
    def region_table():
        df = slx_regions()
        if df.empty:
            return None
        return render.DataGrid(df, selection_mode="rows")

    @output
    @render.data_frame
    def feature_table():
        df = slx_feature_lists()
        if df.empty:
            return None
        return render.DataGrid(df, selection_mode="rows")

    def get_selection(base_id):
        """Retrieve selection indices from the confirmed _selected_rows attribute."""
        try:
            # Shiny 1.6.0 DataGrid selection name
            val = getattr(input, f"{base_id}_selected_rows", None)
            if val is not None:
                return val()
        except:
            pass
        return tuple()

    @output
    @render.text
    def txt_selection_status():
        try:
            r = get_selection("region_table")
            f = get_selection("feature_table")
            return f"Selected: Regs={r}, Feats={f}"
        except:
            return "Selection: (Waiting...)"

    @reactive.effect
    def _():
        # Enable start button only if file and regions are selected
        try:
            reg_sel = get_selection("region_table")
            feat_sel = get_selection("feature_table")
            
            # Cache the values
            selected_regions_idx.set(reg_sel)
            selected_features_idx.set(feat_sel)

            ready = (slx_path() is not None and 
                     len(reg_sel) > 0 and 
                     len(feat_sel) > 0)
            
            ui.update_action_button("btn_process", disabled=not ready)
        except:
            ui.update_action_button("btn_process", disabled=True)
            return

    @reactive.effect
    @reactive.event(input.btn_process)
    def _():
        path = slx_path()
        # Read from CACHED values
        selected_region_indices = selected_regions_idx()
        selected_feature_indices = selected_features_idx()
        
        if not path or not selected_region_indices or len(selected_region_indices) == 0:
            ui.notification_show("Please select at least one region.", type="warning")
            return
            
        ui.notification_show("Starting export. This might take a while. Please wait...", type="message")
        
        # Get selected names
        try:
            reg_df = slx_regions()
            feat_df = slx_feature_lists()
            
            sel_regions = reg_df.iloc[list(selected_region_indices)]["name"].tolist()
            sel_features = feat_df.iloc[list(selected_feature_indices)]["name"].tolist()
        except Exception as e:
            ui.notification_show(f"Data error: {e}", type="error")
            return

        # Prepare JSON context (similar to R app)
        data = {
            "description": "SCiLs-2-ImzML::@::Cemos",
            "version": "0.1",
            "date": str(datetime.datetime.now()),
            "slice_thickness": 10,
            "data_exports": [
                {
                    "filename": path,
                    "outputpath": None,
                    "spot_images": None,
                    "optical_images": None,
                    "featurelists": sel_features,
                    "regions": sel_regions,
                    "regions_as_labels": None,
                    "labels": None
                }
            ]
        }
        
        # Write JSON file
        dir_name = os.path.dirname(path)
        base_name = pathlib.Path(path).stem
        json_file = os.path.join(dir_name, f"{base_name}.json")
        
        print(f"Writing JSON to: {json_file}")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        # Call the export tool
        try:
            # Use subprocess.run for better argument handling on Windows
            json_abs_path = os.path.abspath(json_file)
            print(f"Executing command for: {json_abs_path}")
            
            # Non-blocking or at least cleaner argument list
            process = subprocess.run(
                [sys.executable, "-m", "slx2imzml", json_abs_path],
                check=False
            )
            
            if process.returncode == 0:
                ui.notification_show("Export completed successfully!", type="message", duration=10)
            else:
                print(f"Export tool failed with code: {process.returncode}")
                ui.notification_show(f"Export tool failed (Code: {process.returncode}). Check console for details.", type="error", duration=15)
        except Exception as e:
            ui.notification_show(f"Execution failed: {str(e)}", type="error")

app = App(app_ui, server)
