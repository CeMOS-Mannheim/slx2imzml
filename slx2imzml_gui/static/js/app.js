/* ═══════════════════════════════════════════════════════════════════════════
   SCiLS Exporter — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  regions: [],           // [{color, name, nPx, full_name}]
  featureLists: [],      // [{id, name, nFeat, mzFeat, …}]
  plotTraces: [],
  plotImage: null,
  selectedRegions: new Set(),   // row indices
  selectedFeatures: new Set(),  // row indices
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const btnBrowse      = $("btnBrowse");
const btnProcess     = $("btnProcess");
const btnRegionsAll  = $("btnRegionsAll");
const btnRegionsNone = $("btnRegionsNone");
const btnFeatAll     = $("btnFeaturesAll");
const btnFeatNone    = $("btnFeaturesNone");
const filePath       = $("filePath");
const regionBody     = $("regionBody");
const featureBody    = $("featureBody");
const normBody       = $("normBody");
const fdHead         = $("featureDetailsHead");
const fdBody         = $("featureDetailsBody");
const statusRegions  = $("statusRegions");
const statusFeatures = $("statusFeatures");
const plotDiv        = $("regionPlot");

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = "info", duration = 5000) {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span>${msg}</span>`;
  $("toastContainer").appendChild(el);
  setTimeout(() => {
    el.style.animation = "toastOut .2s ease forwards";
    setTimeout(() => el.remove(), 220);
  }, duration);
}

// ── Loading overlay ───────────────────────────────────────────────────────────
function showLoading(label = "Loading…") {
  let ol = document.querySelector(".loading-overlay");
  if (!ol) {
    ol = document.createElement("div");
    ol.className = "loading-overlay";
    ol.innerHTML = `<div class="spinner"></div><div class="loading-label">${label}</div>`;
    document.body.appendChild(ol);
  } else {
    ol.querySelector(".loading-label").textContent = label;
    ol.style.display = "flex";
  }
}
function hideLoading() {
  const ol = document.querySelector(".loading-overlay");
  if (ol) ol.style.display = "none";
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function api(endpoint, body) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

// ── Browse ────────────────────────────────────────────────────────────────────
btnBrowse.addEventListener("click", async () => {
  showLoading("Opening file dialog…");
  try {
    const data = await api("/api/browse", {});
    if (!data.path) { hideLoading(); return; }
    filePath.textContent = data.path;
    filePath.classList.add("loaded");
    await loadFile(data.path);
  } catch (e) {
    toast("Failed to open file dialog: " + e.message, "error");
    hideLoading();
  }
});

// ── Load file ─────────────────────────────────────────────────────────────────
async function loadFile(path) {
  showLoading("Reading SCiLS file…");
  try {
    const data = await api("/api/load", { path });
    if (!data.ok) { toast(data.error, "error"); hideLoading(); return; }

    state.regions      = data.regions      || [];
    state.featureLists = data.feature_lists || [];
    state.plotTraces   = data.plot_traces   || [];
    state.plotImage    = data.plot_image    || null;
    state.selectedRegions.clear();
    state.selectedFeatures.clear();

    renderNorm(data.normalizations || []);
    renderRegionTable();
    renderFeatureTable();
    renderPlot();
    requestAnimationFrame(() => {
      if (window.Plotly && plotDiv) {
        Plotly.Plots.resize(plotDiv);
      }
    });
    updateStatus();
    hideLoading();
    toast(`Loaded ${state.regions.length} regions and ${state.featureLists.length} feature lists.`, "success");
  } catch (e) {
    toast("Load error: " + e.message, "error");
    hideLoading();
  }
}

// ── Normalizations ────────────────────────────────────────────────────────────
function renderNorm(norms) {
  normBody.innerHTML = norms.length
    ? norms.map(n => `<tr><td>${esc(n.name)}</td></tr>`).join("")
    : `<tr><td class="muted">—</td></tr>`;
}

// ── Region table ──────────────────────────────────────────────────────────────
function renderRegionTable() {
  regionBody.innerHTML = state.regions.map((r, i) => `
    <tr data-idx="${i}" class="${state.selectedRegions.has(i) ? "selected" : ""}">
      <td class="col-color"><span class="color-dot" style="background:${esc(r.color)}"></span></td>
      <td>${esc(r.name)}</td>
      <td class="col-num">${r.nPx.toLocaleString()}</td>
    </tr>`).join("");

  regionBody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", e => toggleRow(tr, state.selectedRegions, e.shiftKey, renderRegionTable, afterRegionSelect));
  });
}

function afterRegionSelect() {
  updateStatus();
  highlightPlotTraces();
}

// ── Feature table ─────────────────────────────────────────────────────────────
function renderFeatureTable() {
  featureBody.innerHTML = state.featureLists.map((f, i) => `
    <tr data-idx="${i}" class="${state.selectedFeatures.has(i) ? "selected" : ""}">
      <td>${esc(f.name)}</td>
      <td class="col-num">${f.nFeat ?? "—"}</td>
      <td class="col-num">${f.mzFeat != null ? (f.mzFeat ? "✓" : "✗") : "—"}</td>
    </tr>`).join("");

  featureBody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", e => toggleRow(tr, state.selectedFeatures, e.shiftKey, renderFeatureTable, afterFeatureSelect));
  });
}

async function afterFeatureSelect() {
  updateStatus();
  await loadFeatureDetails();
}

async function loadFeatureDetails() {
  const ids = [...state.selectedFeatures].map(i => state.featureLists[i].id);
  if (!ids.length) {
    fdHead.innerHTML = `<tr><td class="muted" colspan="99">Select a feature list to see details.</td></tr>`;
    fdBody.innerHTML = "";
    return;
  }
  const data = await api("/api/features", {
    ids,
    include_ccs: $("includeCCS").checked,
  });
  if (!data.ok) {
    toast("Failed to load feature details: " + (data.error || "Unknown error"), "error");
    return;
  }
  renderFeatureDetails(data.features);
}

function renderFeatureDetails(rows) {
  if (!rows.length) {
    fdHead.innerHTML = `<tr><td class="muted" colspan="99">No features found.</td></tr>`;
    fdBody.innerHTML = "";
    return;
  }

  // Determine columns
  const baseCols = ["id", "name", "mz_center", "mz_width_ppm", "mz_low", "mz_high"];
  const ccsCols  = ["ccs_center", "ccs_low", "ccs_high"];
  const hasCcs   = rows.some(r => r.ccs_center != null);
  const cols = hasCcs ? [...baseCols, ...ccsCols] : baseCols;
  const labels = {
    id: "ID", name: "Name", mz_center: "m/z", mz_width_ppm: "ppm",
    mz_low: "m/z Lo", mz_high: "m/z Hi",
    ccs_center: "CCS", ccs_low: "CCS Lo", ccs_high: "CCS Hi",
  };

  fdHead.innerHTML = `<tr>${cols.map(c => `<th>${labels[c] || c}</th>`).join("")}</tr>`;
  fdBody.innerHTML = rows.map(r =>
    `<tr>${cols.map(c => `<td>${r[c] ?? "—"}</td>`).join("")}</tr>`
  ).join("");
}

// ── Row selection helper (with shift-range) ───────────────────────────────────
let _lastClickedIdx = null;
function toggleRow(tr, selSet, shiftKey, rerender, afterFn) {
  const idx = parseInt(tr.dataset.idx, 10);
  if (shiftKey && _lastClickedIdx !== null) {
    const lo = Math.min(idx, _lastClickedIdx);
    const hi = Math.max(idx, _lastClickedIdx);
    for (let i = lo; i <= hi; i++) selSet.add(i);
  } else {
    if (selSet.has(idx)) selSet.delete(idx);
    else selSet.add(idx);
  }
  _lastClickedIdx = idx;
  rerender();
  afterFn();
}

// ── Select All / Clear ────────────────────────────────────────────────────────
btnRegionsAll.addEventListener("click", () => {
  state.regions.forEach((_, i) => state.selectedRegions.add(i));
  renderRegionTable(); afterRegionSelect();
});
btnRegionsNone.addEventListener("click", () => {
  state.selectedRegions.clear();
  renderRegionTable(); afterRegionSelect();
});
btnFeatAll.addEventListener("click", () => {
  state.featureLists.forEach((_, i) => state.selectedFeatures.add(i));
  renderFeatureTable(); afterFeatureSelect();
});
btnFeatNone.addEventListener("click", () => {
  state.selectedFeatures.clear();
  renderFeatureTable(); afterFeatureSelect();
});

// ── Status bar ────────────────────────────────────────────────────────────────
function updateStatus() {
  const selectedRegionNames = new Set();
  for (const i of state.selectedRegions) {
    const r = state.regions[i];
    if (r?.full_name) selectedRegionNames.add(r.full_name);
  }
  const featNames = [...state.selectedFeatures].map(i => state.featureLists[i].name);

  const rDisplay = [...selectedRegionNames].map(n => n.startsWith("Regions/") ? n.slice(8) : n);
  statusRegions.textContent = rDisplay.length
    ? (rDisplay.length <= 4 ? rDisplay.join(", ") : rDisplay.slice(0, 3).join(", ") + ` … (+${rDisplay.length - 3})`)
    : "None";
  statusRegions.className = "status-value" + (rDisplay.length ? "" : " muted");

  statusFeatures.textContent = featNames.length ? featNames.join(", ") : "None";
  statusFeatures.className = "status-value" + (featNames.length ? "" : " muted");

  const ready = state.regions.length && state.selectedRegions.size && state.selectedFeatures.size;
  btnProcess.disabled = !ready;
}

$("includeCCS").addEventListener("change", () => {
  if (state.selectedFeatures.size) {
    loadFeatureDetails();
  }
});

// ── Plotly rendering ──────────────────────────────────────────────────────────
function renderPlot() {
  const layout = {
    yaxis: { autorange: "reversed", title: null, showticklabels: false, showgrid: false, zeroline: false, scaleanchor: "x", scaleratio: 1 },
    xaxis: { title: null, showticklabels: false, showgrid: false, zeroline: false },
    showlegend: false,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    dragmode: "pan",
    hovermode: "closest",
    plot_bgcolor: "rgba(0,0,0,0)",
    paper_bgcolor: "rgba(0,0,0,0)",
    images: state.plotImage ? [state.plotImage] : [],
  };

  const config = { responsive: true, displayModeBar: true, scrollZoom: true, modeBarButtonsToRemove: ["select2d","lasso2d"] };

  // Build trace array — deep copy so we can mutate opacity later
  const traces = state.plotTraces.map(t => ({ ...t, line: { ...t.line }, }));

  Plotly.react(plotDiv, traces, layout, config);

  // Store mapping from trace index for highlight use
  state._plotInitialized = true;
}

function highlightPlotTraces() {
  if (!state._plotInitialized || !state.plotTraces.length) return;

  // Collect selected full region names
  const selectedRegionNames = new Set();
  for (const i of state.selectedRegions) {
    const r = state.regions[i];
    if (r?.full_name) selectedRegionNames.add(r.full_name);
  }

  // Build update arrays
  const n = state.plotTraces.length;
  const lineWidths = new Array(n);
  const fillColors = new Array(n);

  for (let i = 0; i < n; i++) {
    const t = state.plotTraces[i];
    const hex = t.line.color;
    const rgb = hexToRgb(hex);
    const isSelected = selectedRegionNames.has(t.name);
    lineWidths[i] = isSelected ? 5 : 3;
    fillColors[i] = isSelected
      ? `rgba(${rgb.r},${rgb.g},${rgb.b},0.45)`
      : `rgba(${rgb.r},${rgb.g},${rgb.b},0.0)`;
  }

  Plotly.restyle(plotDiv, { "line.width": lineWidths, fillcolor: fillColors });
}

// ── Export ────────────────────────────────────────────────────────────────────
btnProcess.addEventListener("click", async () => {
  const regionIndices  = [...state.selectedRegions];
  const featureIndices = [...state.selectedFeatures];

  if (!regionIndices.length || !featureIndices.length) {
    toast("Please select at least one region and one feature list.", "warning");
    return;
  }

  showLoading("Running export…");
  try {
    const data = await api("/api/export", {
      region_indices:  regionIndices,
      feature_indices: featureIndices,
      include_ccs:     $("includeCCS").checked,
      slice_thickness: parseInt($("sliceThickness").value, 10) || 10,
    });
    hideLoading();
    if (data.ok) {
      toast(`Export complete! Files saved to: ${data.output_dir}`, "success", 10000);
    } else {
      toast("Export failed: " + data.error, "error", 10000);
    }
  } catch (e) {
    hideLoading();
    toast("Export error: " + e.message, "error");
  }
});

// ── Utilities ─────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function hexToRgb(hex) {
  const m = hex.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  return m ? { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) } : { r: 128, g: 128, b: 128 };
}
