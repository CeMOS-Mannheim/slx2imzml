"""
SCiLS Exporter — Flask/HTML5 GUI
"""
from __future__ import annotations

import base64
import datetime
import io
import json
import math
import os
import pathlib
import socket
import subprocess
import sys
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request, Response
from PIL import Image

# ---------------------------------------------------------------------------
# Bootstrap: ensure slx2imzml is importable
# ---------------------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def _ensure_package() -> None:
    if importlib.util.find_spec("slx2imzml") is None:
        print(f"[bootstrap] Installing slx2imzml from {ROOT_DIR} …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(ROOT_DIR)],
            stdout=subprocess.DEVNULL,
        )


_ensure_package()

import scilslab as sl  # noqa: E402  (must come after bootstrap)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Single-user in-process state (local tool — no auth needed)
STATE: dict = {
    "slx_path": None,
    "regions": [],          # list[dict]  — full region metadata
    "feature_lists": [],    # list[dict]
    "normalizations": [],   # list[dict]
    "features_cache": {},   # list_id -> list[dict]
    "plot_traces": [],      # list[dict]  — Plotly trace data
    "plot_image": None,     # dict | None — Plotly layout image
}


def _json_safe(value):
    """Recursively convert values to JSON-safe primitives.

    - NaN/Inf -> None
    - numpy scalar types -> native Python types
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def _get_free_local_port() -> int:
    """Return an available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Optical-image helper
# ---------------------------------------------------------------------------
def _process_optical_image(dataset) -> dict | None:
    try:
        df_ids = dataset.optical_images.get_ids()
        if df_ids.empty:
            return None

        overview = df_ids[df_ids["name"] == "Overview Image"]
        img_id = overview.iloc[0]["id"] if not overview.empty else df_ids.iloc[0]["id"]
        opt_img = dataset.optical_images.get_image(img_id)

        pil = Image.open(io.BytesIO(opt_img.data)).convert("RGBA")
        max_dim = 1500
        if pil.width > max_dim or pil.height > max_dim:
            ratio = min(max_dim / pil.width, max_dim / pil.height)
            pil = pil.resize(
                (int(pil.width * ratio), int(pil.height * ratio)),
                Image.Resampling.LANCZOS,
            )
            W, H = pil.width / ratio, pil.height / ratio
        else:
            W, H = float(pil.width), float(pil.height)

        T = np.array(opt_img.px2world)
        x0, y0 = float(T[0, 3]), float(T[1, 3])
        sx, sy = float(T[0, 0]) * W, float(T[1, 1]) * H

        if sx < 0:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        if sy < 0:
            pil = pil.transpose(Image.FLIP_TOP_BOTTOM)

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return dict(
            source="data:image/png;base64," + b64,
            xref="x", yref="y",
            x=min(x0, x0 + sx),
            y=min(y0, y0 + sy),
            xanchor="left", yanchor="top",
            sizex=abs(sx), sizey=abs(sy),
            sizing="stretch", opacity=1.0, layer="below",
        )
    except Exception as exc:
        print(f"[optical] {exc}")
        return None


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.post("/api/browse")
def api_browse():
    """Open a native file dialog and return the chosen path.

    Tkinter file dialogs require the main thread. Flask requests may run in
    worker threads, so we launch the dialog in a short-lived subprocess to
    keep this endpoint safe regardless of server threading mode.
    """
    dialog_script = (
        "import tkinter as tk; "
        "from tkinter import filedialog; "
        "root=tk.Tk(); "
        "root.withdraw(); "
        "root.attributes('-topmost', True); "
        "path=filedialog.askopenfilename(filetypes=[('SCiLS Lab files','*.slx')]); "
        "root.destroy(); "
        "print(path or '')"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", dialog_script],
            check=False,
            capture_output=True,
            text=True,
        )
        path = (completed.stdout or "").strip()
        return jsonify({"path": path})
    except Exception as exc:
        return jsonify({"path": "", "error": str(exc)}), 500


@app.post("/api/load")
def api_load():
    """Load an .slx file and populate STATE."""
    body = request.get_json(force=True)
    path: str = body.get("path", "").strip()

    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": f"File not found: {path}"}), 400

    session = None
    try:
        # SCiLS defaults to port 8086. That port may already be used by another
        # SCiLS process, so pick a free localhost port per session.
        session = sl.LocalSession(path, port=_get_free_local_port())
        with session as slx:
            dataset = slx.dataset_proxy

            # ── Feature lists ─────────────────────────────────────────────
            fl = dataset.feature_table.get_feature_lists()
            fl_display = fl.rename(columns={
                "num_features": "nFeat",
                "has_mz_features": "mzFeat",
                "has_mobility_intervals": "mobilityIntervals",
                "has_ccs_features": "ccsFeatures",
            })
            if "has_external_features" in fl_display.columns:
                fl_display = fl_display.drop(columns=["has_external_features"])
            STATE["feature_lists"] = fl_display.to_dict("records")

            # ── Feature cache ─────────────────────────────────────────────
            cache: dict = {}
            for list_id in fl["id"]:
                cache[list_id] = dataset.feature_table.get_features(list_id, mode="area").to_dict("records")
            STATE["features_cache"] = cache

            # ── Normalizations ────────────────────────────────────────────
            norms = dataset.get_normalizations()
            STATE["normalizations"] = [{"name": name} for _, name in norms.items()]

            # ── Optical image ─────────────────────────────────────────────
            STATE["plot_image"] = _process_optical_image(dataset)

            # ── Regions ───────────────────────────────────────────────────
            root_region = dataset.get_region_tree()
            all_regions = root_region.get_all_regions()

            cmap = plt.get_cmap("tab20")
            regions_data, traces = [], []
            idx = 0

            for r in all_regions:
                if r.name == "Regions":
                    continue

                try:
                    if hasattr(r, "spots") and isinstance(r.spots, dict) and "spot_id" in r.spots:
                        num_spots = len(r.spots["spot_id"])
                    else:
                        spots = dataset.get_region_spots(r.id)
                        num_spots = len(spots.get("spot_id", [])) if isinstance(spots, dict) else 0
                except Exception:
                    num_spots = 0

                display_name = r.name[8:] if r.name.startswith("Regions/") else r.name
                color = cmap(idx % 20)
                hex_color = "#%02x%02x%02x" % (
                    int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                )
                r_int = int(color[0] * 255)
                g_int = int(color[1] * 255)
                b_int = int(color[2] * 255)

                regions_data.append({
                    "color": hex_color,
                    "name": display_name,
                    "nPx": int(num_spots),
                    "full_name": r.name,
                })

                if hasattr(r, "polygons"):
                    xs, ys = [], []
                    for poly in r.polygons:
                        if poly:
                            px = [p.x for p in poly]
                            py = [p.y for p in poly]
                            px.append(px[0]); py.append(py[0])
                            xs.extend(px); xs.append(None)
                            ys.extend(py); ys.append(None)
                    if xs:
                        traces.append({
                            "x": xs, "y": ys,
                            "fill": "toself", "mode": "lines",
                            "line": {"color": hex_color, "width": 3},
                            "fillcolor": f"rgba({r_int},{g_int},{b_int},0.0)",
                            "name": r.name,
                            "hoverinfo": "name",
                            "hoverlabel": {"namelength": -1},
                            "type": "scatter",
                        })

                idx += 1

            STATE["slx_path"] = path
            STATE["regions"] = regions_data
            STATE["plot_traces"] = traces

    except PermissionError:
        return jsonify({"ok": False, "error": f"Permission denied — is the file open in SCiLS Lab?"}), 403
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        # Explicit shutdown after selection data is generated.
        # (close() is idempotent in scilslab and safe if the context manager
        # has already closed the process.)
        if session is not None:
            try:
                session.close()
            except Exception as close_exc:
                print(f"[scilslab] warning: failed to close LocalSession cleanly: {close_exc}")

    return jsonify({
        "ok": True,
        "regions": STATE["regions"],
        "feature_lists": _json_safe(STATE["feature_lists"]),
        "normalizations": STATE["normalizations"],
        "plot_traces": STATE["plot_traces"],
        "plot_image": STATE["plot_image"],
    })


@app.post("/api/features")
def api_features():
    """Return merged feature details for a list of feature-list IDs."""
    body = request.get_json(force=True)
    ids: list = body.get("ids", [])
    include_ccs: bool = bool(body.get("include_ccs", True))

    cache = STATE["features_cache"]
    all_rows: list[dict] = []
    seen_ids: set = set()

    for lid in ids:
        # IDs may arrive as strings or ints
        key = lid if lid in cache else (int(lid) if isinstance(lid, str) else str(lid))
        rows = cache.get(key, [])
        for row in rows:
            rid = row.get("id")
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_rows.append(dict(row))

    # Compute derived columns
    def _safe_float(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return v

    def _safe_round(value, digits):
        v = _safe_float(value)
        if v is None:
            return None
        return round(v, digits)

    for row in all_rows:
        mz_lo_raw = row.get("mz_low")
        mz_hi_raw = row.get("mz_high")
        mz_lo = _safe_float(mz_lo_raw)
        mz_hi = _safe_float(mz_hi_raw)

        if mz_lo is not None and mz_hi is not None:
            mz_c = (mz_lo + mz_hi) / 2
            row["mz_center"] = _safe_round(mz_c, 4)
            row["mz_low"] = _safe_round(mz_lo, 4)
            row["mz_high"] = _safe_round(mz_hi, 4)
            row["mz_width_ppm"] = _safe_round(((mz_hi - mz_lo) / mz_c) * 1e6, 1) if mz_c else None
        else:
            row["mz_center"] = None
            row["mz_low"] = _safe_round(mz_lo_raw, 4)
            row["mz_high"] = _safe_round(mz_hi_raw, 4)
            row["mz_width_ppm"] = None
        if include_ccs:
            ccs_lo = _safe_round(row.get("ccs_low"), 2) if "ccs_low" in row else None
            ccs_hi = _safe_round(row.get("ccs_high"), 2) if "ccs_high" in row else None
            if "ccs_low" in row:
                row["ccs_low"] = ccs_lo
            if "ccs_high" in row:
                row["ccs_high"] = ccs_hi
            if ccs_lo is not None and ccs_hi is not None:
                row["ccs_center"] = _safe_round((ccs_lo + ccs_hi) / 2, 2)
            elif "ccs_low" in row or "ccs_high" in row:
                row["ccs_center"] = None
        else:
            row.pop("ccs_low", None)
            row.pop("ccs_high", None)
            row.pop("ccs_center", None)

    all_rows.sort(key=lambda r: (r.get("mz_center") is None, r.get("mz_center") or 0))
    return jsonify({"ok": True, "features": _json_safe(all_rows)})


@app.post("/api/export")
def api_export():
    """Write a JSON config and invoke the slx2imzml exporter."""
    body = request.get_json(force=True)
    selected_region_indices: list[int] = body.get("region_indices", [])
    selected_feature_indices: list[int] = body.get("feature_indices", [])
    include_ccs: bool = bool(body.get("include_ccs", True))
    slice_thickness: int = int(body.get("slice_thickness", 10))

    path = STATE["slx_path"]
    if not path:
        return jsonify({"ok": False, "error": "No file loaded."}), 400

    reg_df = STATE["regions"]
    feat_df = STATE["feature_lists"]

    selected_regions: list[str] = []
    seen_regions: set[str] = set()
    for i in selected_region_indices:
        row = reg_df[i]
        region_name = row.get("full_name")
        if region_name and region_name not in seen_regions:
            seen_regions.add(region_name)
            selected_regions.append(region_name)

    sel_features = [feat_df[i]["name"] for i in selected_feature_indices]

    data = {
        "description": "SCiLS-2-ImzML::@::Cemos",
        "version": "0.1",
        "date": str(datetime.datetime.now()),
        "data_exports": [{
            "filename": path,
            "outputpath": None,
            "slice_thickness": slice_thickness,
            "spot_images": None,
            "optical_images": None,
            "featurelists": sel_features,
            "include_ccs": include_ccs,
            "regions": selected_regions,
            "regions_as_labels": None,
            "labels": None,
        }],
    }

    json_file = os.path.join(
        os.path.dirname(path), f"{pathlib.Path(path).stem}.json"
    )
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "slx2imzml", os.path.abspath(json_file)],
            check=False,
        )
        if proc.returncode == 0:
            return jsonify({"ok": True, "output_dir": os.path.dirname(path)})
        return jsonify({"ok": False, "error": f"Exporter exited with code {proc.returncode}."}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5001")
    app.run(debug=False, port=5001, threaded=False)
