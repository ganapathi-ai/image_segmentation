/* app.js — Diabetic Foot Ulcer Segmentation
   U-Net + ResNet34 · ONNX Runtime inference
   Professional clinical imaging interface
   */

// ── State ──────────────────────────────────────────────────
let weightsLoaded = false;

// ── Helpers ────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

function setStatus(dotEl, textEl, state, label) {
  dotEl.className = "status-dot " + state;
  textEl.textContent = label;
}

function log(msg, type = "") {
  const bar = $("log-bar");
  const time = new Date().toLocaleTimeString("en-US", { hour12: false });
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `<span class="log-time">${time}</span><span class="log-msg ${type}">${msg}</span>`;
  bar.appendChild(line);
  bar.scrollTop = bar.scrollHeight;
}

function setupDropzone(dropId, inputId, onFile) {
  const drop  = $(dropId);
  const input = $(inputId);

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });

  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag-over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) onFile(input.files[0]);
  });
}

// ── Boot ───────────────────────────────────────────────────
async function boot() {
  const dot  = $("status-dot");
  const txt  = $("status-text");

  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.weights_loaded) {
      weightsLoaded = true;
      setStatus(dot, txt, "ready", "Model ready");
      enableImageUpload();
      log("Model weights pre-loaded from server.", "success");
    } else {
      setStatus(dot, txt, "", "Awaiting weights");
    }
  } catch {
    setStatus(dot, txt, "", "Checking…");
  }
}

function enableImageUpload() {
  const zone = $("image-drop");
  zone.classList.remove("locked");
}

// ── Step 1: Weights Upload ─────────────────────────────────
setupDropzone("weights-drop", "weights-input", async (file) => {
  const statusEl = $("weights-status");

  try {
    statusEl.textContent = "Loading…";
    statusEl.style.color = "var(--accent-amber)";

    const form = new FormData();
    form.append("file", file);

    const res  = await fetch("/api/upload-weights", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = data.error || "Upload failed";
      statusEl.style.color = "var(--accent)";
      log("Weights upload failed: " + (data.error || "unknown"), "error");
      return;
    }

    weightsLoaded = true;
    statusEl.textContent = "Loaded — " + file.name;
    statusEl.style.color = "var(--accent-green)";
    setStatus($("status-dot"), $("status-text"), "ready", "Model ready");

    // Unlock image upload
    enableImageUpload();
    log(`Weights loaded: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`, "success");

  } catch (err) {
    statusEl.textContent = "Network error";
    statusEl.style.color = "var(--accent)";
    log("Upload error: " + err.message, "error");
  }
});

// ── Step 2: Image Upload ───────────────────────────────────
setupDropzone("image-drop", "image-input", (file) => {
  if (!file.type.startsWith("image/")) {
    log("Unsupported file type.", "warn");
    return;
  }

  const url = URL.createObjectURL(file);
  const previewImg = $("preview-img");
  previewImg.src = url;

  $("image-name").textContent = file.name;
  $("image-size").textContent = `${(file.size / 1024).toFixed(1)} KB · ${file.type.split("/")[1].toUpperCase()}`;

  const previewRow = $("image-preview");
  previewRow.classList.add("visible");

  // Enable segment button
  $("segment-btn").disabled = false;
  $("segment-btn")._imageFile = file;

  log(`Image loaded: ${file.name}`, "success");
});

$("clear-image")?.addEventListener("click", () => {
  $("image-preview").classList.remove("visible");
  $("image-input").value = "";
  $("segment-btn")._imageFile = null;
  $("segment-btn").disabled = true;
});

// ── Step 3: Run Segmentation ───────────────────────────────
$("segment-btn").addEventListener("click", async () => {
  const btn  = $("segment-btn");
  const file = btn._imageFile;
  if (!file) return;

  btn.disabled = true;
  btn.classList.add("loading");

  // Show viewport loading
  const viewportLoading = $("viewport-loading");
  viewportLoading.classList.remove("hidden");
  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "none";

  // Show sidebar loading
  $("sidebar-loading").style.display = "block";

  const startTime = performance.now();
  log("Running inference…");

  try {
    const form = new FormData();
    form.append("image", file);

    const res  = await fetch("/api/segment", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      log("Inference failed: " + (data.error || "unknown"), "error");
      return;
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

    // Display results
    $("res-original").src = "data:image/jpeg;base64," + data.original;
    $("res-overlay").src  = "data:image/jpeg;base64," + data.overlay;

    $("dl-overlay").href = "data:image/png;base64," + data.overlay;
    $("dl-mask").href    = "data:image/png;base64," + data.mask;

    // Update metrics
    const coverage = parseFloat(data.coverage_pct);
    $("metric-coverage").textContent = coverage.toFixed(1) + "%";
    $("metric-coverage").classList.remove("pending");

    const pixels = Math.round(data.mask.length * 0.75 * coverage / 100); // rough estimate
    $("metric-pixels").textContent = new Intl.NumberFormat().format(pixels);
    $("metric-pixels").classList.remove("pending");

    $("metric-size").textContent = `${data.image_size.width}×${data.image_size.height}`;
    $("metric-size").classList.remove("pending");

    $("metric-time").textContent = elapsed + "s";
    $("metric-time").classList.remove("pending");

    // Show results
    $("metrics-bar").style.display = "flex";
    $("actions-bar").style.display = "flex";
    $("viewport-loading").classList.add("hidden");
    $("viewport-grid").style.display = "grid";
    $("sidebar-loading").style.display = "none";

    // Complete progress bar
    $("inference-fill").style.width = "100%";
    $("inference-pct").textContent = "100";

    log(`Inference complete: ${elapsed}s · ${coverage.toFixed(1)}% coverage`, "success");

  } catch (err) {
    log("Network error: " + err.message, "error");
    viewportLoading.classList.add("hidden");
    $("viewport-empty").style.display = "";
    $("sidebar-loading").style.display = "none";
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
  }
});

// ── Boot ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  boot();
  log("Session initialized. Upload weights to begin.");
});
