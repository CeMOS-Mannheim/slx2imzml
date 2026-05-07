import os
import json
import pathlib
import datetime
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_widget
import faicons as fa
import scilslab as sl
from PIL import Image
import io
import base64

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

# --- Helpers ---
def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(filetypes=[("SCiLS Lab files", "*.slx")])
    root.destroy()
    return file_path

def get_selection(input: Inputs, base_id: str):
    """Retrieve selection indices from the confirmed _selected_rows attribute."""
    try:
        val = getattr(input, f"{base_id}_selected_rows", None)
        if val is not None:
            return val()
    except Exception:
        pass
    return tuple()

def process_optical_image(dataset):
    """Extracts and processes the optical image from the SCiLS dataset for display."""
    try:
        df_ids = dataset.optical_images.get_ids()
        if df_ids.empty:
            return None
            
        # Prefer the 'Overview Image' which is always present in standard exports
        overview_ids = df_ids[df_ids['name'] == 'Overview Image']
        img_id = overview_ids.iloc[0]['id'] if not overview_ids.empty else df_ids.iloc[0]['id']
        opt_img = dataset.optical_images.get_image(img_id)
        
        pil_img = Image.open(io.BytesIO(opt_img.data)).convert("RGBA")
        
        # Downsample for faster rendering
        max_dim = 1500
        if pil_img.width > max_dim or pil_img.height > max_dim:
            ratio = min(max_dim/pil_img.width, max_dim/pil_img.height)
            new_size = (int(pil_img.width*ratio), int(pil_img.height*ratio))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            W, H = pil_img.width / ratio, pil_img.height / ratio
        else:
            W, H = pil_img.width, pil_img.height
            
        T = np.array(opt_img.px2world)
        x0, y0 = T[0, 3], T[1, 3]
        sizex_raw, sizey_raw = T[0, 0] * W, T[1, 1] * H
        
        # Handle negative transforms by flipping the image
        if sizex_raw < 0: pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        if sizey_raw < 0: pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            
        # Encode the oriented image
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return dict(
            source="data:image/png;base64," + img_str,
            xref="x", yref="y",
            x=min(x0, x0 + sizex_raw),
            y=min(y0, y0 + sizey_raw),
            xanchor="left", yanchor="top",
            sizex=abs(sizex_raw), sizey=abs(sizey_raw),
            sizing="stretch", opacity=1.0, layer="below"
        )
    except (AttributeError, KeyError, IndexError) as e:
        print(f"Warning: Dataset structure does not support optical images: {e}")
    except Exception as e:
        print(f"Unexpected error loading optical image: {e}")
    return None

# --- UI Definition ---
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
            ui.output_ui("ui_selection_status"),
            ui.p("Click the button to start the export process. Ensure you have selected regions and feature lists."),
        ),
        ui.card(
            ui.card_header("Normalizations - For Info Only"),
            ui.output_data_frame("spot_image_table"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Advanced Options"),
            ui.input_numeric("slice_thickness", "Slice Thickness (µm)", value=10, min=1),
        ),
        title="SCiLS Exporter Controls",
        width="20%",
    ),
    ui.layout_columns(
        ui.div(
            ui.card(
                ui.card_header("Region Tree"),
                ui.div(
                    ui.input_action_button("btn_regions_all", "Select All", class_="btn-sm"),
                    ui.input_action_button("btn_regions_none", "Clear", class_="btn-sm"),
                    class_="d-flex gap-2 mb-2"
                ),
                ui.output_data_frame("region_table"),
                full_screen=True,
                class_="flex-fill"
            ),
            ui.card(
                output_widget("region_plot"),
                full_screen=True,
                class_="flex-fill"
            ),
            class_="d-flex flex-column h-100 gap-3"
        ),
        ui.div(
            ui.card(
                ui.card_header("Feature Lists"),
                ui.div(
                    ui.input_action_button("btn_features_all", "Select All", class_="btn-sm"),
                    ui.input_action_button("btn_features_none", "Clear", class_="btn-sm"),
                    class_="d-flex gap-2 mb-2"
                ),
                ui.output_data_frame("feature_table"),
                full_screen=True,
                class_="flex-fill"
            ),
            ui.card(
                ui.card_header("Feature Details"),
                ui.output_data_frame("feature_details_table"),
                full_screen=True,
                class_="flex-fill"
            ),
            class_="d-flex flex-column h-100 gap-3"
        ),
        col_widths=[6, 6]
    ),
    title="SCiLS Exporter",
    fillable=True,
)

# --- Server Logic ---
def server(input: Inputs, output: Outputs, session: Session):
    # --- State Definitions ---
    slx_path = reactive.Value(None)
    slx_regions = reactive.Value(pd.DataFrame())
    slx_regions_styles = reactive.Value([])
    slx_feature_lists = reactive.Value(pd.DataFrame())
    slx_spot_images = reactive.Value(pd.DataFrame())
    slx_feature_details = reactive.Value(pd.DataFrame())
    slx_optical_image = reactive.Value(None)
    
    fig = go.FigureWidget()
    fig.update_layout(
        yaxis=dict(autorange='reversed', title='Y Coordinates', showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        xaxis=dict(title='X Coordinates', showgrid=False, zeroline=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode='pan', hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # --- Loading Logic ---
    @reactive.effect
    @reactive.event(input.btn_browse)
    def _():
        print("Browse button clicked")
        path = select_file()
        if path:
            print(f"Selected file: {path}")
            slx_path.set(path)
            
    @reactive.effect
    def _load_dataset():
        path = slx_path()
        if not path or not os.path.exists(path):
            return

        ui.notification_show("Trying to access SCiLS file...", type="message")
        try:
            with sl.LocalSession(path) as slx_file:
                dataset = slx_file.dataset_proxy
                
                # 1. Load Feature Lists
                fl = dataset.feature_table.get_feature_lists()
                fl_display = fl.rename(columns={
                    "num_features": "nFeat", "has_mz_features": "mzFeat",
                    "has_mobility_intervals": "mobilityIntervals", "has_ccs_features": "ccsFeatures"
                })
                if "has_external_features" in fl_display.columns:
                    fl_display = fl_display.drop(columns=["has_external_features"])
                slx_feature_lists.set(fl_display)
                
                # 2. Load Normalizations
                normalizations = dataset.get_normalizations()
                slx_spot_images.set(pd.DataFrame([{"name": name} for uid, name in normalizations.items()]))
                
                # 3. Load Optical Image
                slx_optical_image.set(process_optical_image(dataset))
                
                # 4. Load Regions
                all_regions = dataset.get_region_tree().get_all_regions()
                print(f"Found {len(all_regions)} regions in total")
                
                cmap = plt.get_cmap('tab20')
                regions_data, trace_data, styles = [], [], []
                
                idx = 0
                for r in all_regions:
                    if len(r.subregions) == 0: # Leaf nodes only
                        spots = dataset.get_region_spots(r.id)
                        num_spots = len(spots.get('spot_id', [])) if isinstance(spots, dict) else 0
                        color = cmap(idx % 20)
                        hex_color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                        
                        regions_data.append({"Color": "", "name": r.name, "nPx": num_spots, "subRegions": 0})
                        styles.append({"rows": [idx], "cols": [0], "style": {"background-color": hex_color}})
                        
                        if hasattr(r, 'polygons'):
                            xs, ys = [], []
                            for poly in r.polygons:
                                if len(poly) > 0:
                                    poly_xs = [p.x for p in poly]
                                    poly_ys = [p.y for p in poly]
                                    # Close the polygon by repeating the first point
                                    poly_xs.append(poly_xs[0])
                                    poly_ys.append(poly_ys[0])
                                    xs.extend(poly_xs); xs.append(None)
                                    ys.extend(poly_ys); ys.append(None)
                            
                            if xs:
                                trace_data.append(dict(
                                    x=xs, y=ys, fill='toself', mode='lines',
                                    line=dict(color=hex_color, width=3),
                                    fillcolor=f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.0)',
                                    name=r.name, hoverinfo='name', hoverlabel=dict(namelength=-1)
                                ))
                        idx += 1
                
                print(f"Loaded {len(regions_data)} leaf regions")
                slx_regions.set(pd.DataFrame(regions_data))
                slx_regions_styles.set(styles)
                
                with fig.batch_update():
                    fig.data = []
                    for td in trace_data: fig.add_trace(go.Scatter(**td))
                
            ui.notification_show("SCiLS file access successful. Please choose regions and feature lists to export.", type="message")
        except FileNotFoundError:
            ui.notification_show(f"File not found: {path}", type="error")
        except PermissionError:
            ui.notification_show(f"Permission denied accessing: {path}. Is it open in SCiLS Lab?", type="error")
        except Exception as e:
            ui.notification_show(f"Error accessing SCiLS file: {str(e)}", type="error")
            print(f"Detailed loading error: {e}")

    # --- UI Logic & Selection ---
    @reactive.effect
    def _update_ui_state():
        # Enable start button only if file and regions/features are selected
        try:
            reg_sel = get_selection(input, "region_table")
            feat_sel = get_selection(input, "feature_table")
            ready = (slx_path() is not None and len(reg_sel) > 0 and len(feat_sel) > 0)
            ui.update_action_button("btn_process", disabled=not ready)
        except Exception:
            ui.update_action_button("btn_process", disabled=True)

    @reactive.effect
    def _update_feature_details():
        selected_indices = get_selection(input, "feature_table")
        path = slx_path()
        if not path or not selected_indices:
            slx_feature_details.set(pd.DataFrame())
            return

        fl_df = slx_feature_lists()
        selected_ids = fl_df.iloc[list(selected_indices)]["id"].tolist()
        
        try:
            with sl.LocalSession(path) as slx_file:
                dataset = slx_file.dataset_proxy
                all_features = []
                for list_id in selected_ids:
                    features = dataset.feature_table.get_features(list_id)
                    all_features.append(features)
                
                if all_features:
                    df = pd.concat(all_features, ignore_index=True)
                    # Add mz_center/centroid for convenience
                    df["mz_center"] = (df["mz_low"] + df["mz_high"]) / 2
                    # Add mz width in ppm
                    # ppm = (delta_mz / mz_center) * 10^6
                    df["mz width"] = ((df["mz_high"] - df["mz_low"]) / df["mz_center"]) * 1e6
                    
                    # Sort by mz_center
                    df = df.sort_values("mz_center")

                    # Round values for display: 4 digits for mz, 1 digit for ppm
                    for col in ["mz_center", "mz_low", "mz_high"]:
                        if col in df.columns:
                            df[col] = df[col].round(4)
                    if "mz width" in df.columns:
                        df["mz width"] = df["mz width"].round(1)
                    
                    # Order columns as requested: id, name, mz_center, mz width, mz_low, mz_high
                    available_cols = [c for c in ["id", "name", "mz_center", "mz width", "mz_low", "mz_high"] if c in df.columns]
                    slx_feature_details.set(df[available_cols])
                else:
                    slx_feature_details.set(pd.DataFrame())
        except Exception as e:
            print(f"Error fetching feature details: {e}")
            slx_feature_details.set(pd.DataFrame())

    @reactive.effect
    def _update_plot_selection():
        selected = get_selection(input, "region_table")
        with fig.batch_update():
            for i, trace in enumerate(fig.data):
                c = trace.line.color
                r_val, g_val, b_val = (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)) if c.startswith('#') else (128,128,128)
                if i in selected:
                    trace.line.width = 5
                    trace.fillcolor = f"rgba({r_val}, {g_val}, {b_val}, 0.5)"
                else:
                    trace.line.width = 3
                    trace.fillcolor = f"rgba({r_val}, {g_val}, {b_val}, 0.0)"

    # --- Export Execution ---
    @reactive.effect
    @reactive.event(input.btn_process)
    def _run_export():
        path = slx_path()
        selected_region_indices = get_selection(input, "region_table")
        selected_feature_indices = get_selection(input, "feature_table")
        
        if not path or not selected_region_indices:
            ui.notification_show("Please select at least one region.", type="warning")
            return
            
        ui.notification_show("Starting export. This might take a while. Please wait...", type="message")
        
        try:
            reg_df, feat_df = slx_regions(), slx_feature_lists()
            sel_regions = reg_df.iloc[list(selected_region_indices)]["name"].tolist()
            sel_features = feat_df.iloc[list(selected_feature_indices)]["name"].tolist()
        except (KeyError, IndexError) as e:
            ui.notification_show(f"Selection data error: Could not find metadata. ({e})", type="error")
            return
        except Exception as e:
            ui.notification_show(f"Data processing error: {e}", type="error")
            return

        # Prepare JSON context
        data = {
            "description": "SCiLs-2-ImzML::@::Cemos", "version": "0.1", "date": str(datetime.datetime.now()),
            "data_exports": [{
                "filename": path, "outputpath": None, "slice_thickness": input.slice_thickness(),
                "spot_images": None, "optical_images": None, "featurelists": sel_features,
                "regions": sel_regions, "regions_as_labels": None, "labels": None
            }]
        }
        
        json_file = os.path.join(os.path.dirname(path), f"{pathlib.Path(path).stem}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        try:
            json_abs_path = os.path.abspath(json_file)
            print(f"Executing export tool for: {json_abs_path}")
            process = subprocess.run([sys.executable, "-m", "slx2imzml", json_abs_path], check=False)
            
            if process.returncode == 0:
                out_dir = os.path.dirname(path)
                ui.notification_show(f"Export completed successfully! Files saved to: {out_dir}", type="message", duration=15)
            else:
                ui.notification_show(f"Export tool failed (Code: {process.returncode}).", type="error", duration=15)
        except FileNotFoundError:
            ui.notification_show("Python executable or slx2imzml module not found.", type="error")
        except subprocess.SubprocessError as e:
            ui.notification_show(f"Export process failed to start: {str(e)}", type="error")
        except Exception as e:
            ui.notification_show(f"Unexpected execution failure: {str(e)}", type="error")

    # --- Renderers ---
    @output
    @render.text
    def txt_path():
        return slx_path() or "No file selected"

    @output
    @render.data_frame
    def region_table():
        df = slx_regions()
        return render.DataGrid(df, selection_mode="rows", styles=slx_regions_styles(), height="100%") if not df.empty else None
        
    @output
    @render_widget
    def region_plot():
        if slx_regions().empty: return None
        fig.update_layout(images=[slx_optical_image()] if slx_optical_image() else [])
        return fig

    @output
    @render.data_frame
    def feature_table():
        df = slx_feature_lists()
        return render.DataGrid(df, selection_mode="rows", height="100%") if not df.empty else None

    @output
    @render.data_frame
    def spot_image_table():
        df = slx_spot_images()
        return render.DataGrid(df, selection_mode="none", height="200px") if not df.empty else None

    @output
    @render.data_frame
    def feature_details_table():
        df = slx_feature_details()
        return render.DataGrid(df, selection_mode="none", height="100%") if not df.empty else None

    @output
    @render.ui
    def ui_selection_status():
        try:
            r, f = get_selection(input, "region_table"), get_selection(input, "feature_table")
            reg_df, feat_df = slx_regions(), slx_feature_lists()
            sel_regions = reg_df.iloc[list(r)]["name"].tolist() if not reg_df.empty and r else []
            sel_features = feat_df.iloc[list(f)]["name"].tolist() if not feat_df.empty and f else []
            
            return ui.div(
                ui.h6("Currently Selected:", class_="fw-bold mb-2"),
                ui.div(ui.span("Regions:", class_="fw-semibold me-2"), ui.span(", ".join(sel_regions) or "None", class_="text-muted"), class_="mb-1 text-break"),
                ui.div(ui.span("Features:", class_="fw-semibold me-2"), ui.span(", ".join(sel_features) or "None", class_="text-muted"), class_="text-break"),
                class_="mt-3 mb-3 p-3 border rounded bg-light"
            )
        except Exception:
            return ui.div(ui.h6("Currently Selected:", class_="fw-bold mb-2"), class_="mt-3 mb-3 p-3 border rounded bg-light")

app = App(app_ui, server)

