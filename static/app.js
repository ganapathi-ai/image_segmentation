/* app.js — Diabetic Foot Ulcer Segmentation
   U-Net + ResNet34 · ONNX Runtime inference
   */

let modelReady = false;
let selectedFile = null;

const $ = (id) => document.getElementById(id);

function log(msg, type) {
  const bar = $("log-bar");
  const time = new Date().toLocaleTimeString("en-US", { hour12: false });
  const line = document.createElement("div");
  line.className = "log-line";
  const cls = type ? " log-msg-" + type : "";
  line.innerHTML = '<span class="log-time">' + time + '</span><span class="log-msg' + cls + '">' + msg + '</span>';
  bar.appendChild(line);
  bar.scrollTop = bar.scrollHeight;
}

function setStatus(dot, txt, state, label) {
  dot.className = "status-dot " + (state || "");
  txt.textContent = label;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

// ── Boot ───────────────────────────────────────────────────
async function boot() {
  const dot = $("status-dot");
  const txt = $("status-text");

  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    if (data.weights_loaded) {
      modelReady = true;
      setStatus(dot, txt, "ready", "Model ready");
      log("Model loaded — ready for inference.", "success");
    } else {
      setStatus(dot, txt, "error", "Model unavailable");
      log("Model not loaded on server.", "error");
    }
  } catch (err) {
    setStatus(dot, txt, "", "Checking…");
    log("Status error: " + err.message, "warn");
  }
}

// ── File Input (native click via overlay) ──────────────────
const imageInput = $("image-input");
const imageDrop  = $("image-drop");

imageInput.addEventListener("change", function() {
  console.log("[DFU] input change:", this.files.length);
  if (this.files && this.files[0]) {
    onFileSelected(this.files[0]);
  }
});

// ── Drag and Drop ──────────────────────────────────────────
imageDrop.addEventListener("dragover", function(e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.add("drag-over");
});

imageDrop.addEventListener("dragleave", function(e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.remove("drag-over");
});

imageDrop.addEventListener("drop", function(e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.remove("drag-over");
  console.log("[DFU] drop:", e.dataTransfer.files.length);
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    onFileSelected(e.dataTransfer.files[0]);
  }
});

// ── File Selected ──────────────────────────────────────────
function onFileSelected(file) {
  console.log("[DFU] file:", file.name, file.type, file.size);

  if (!file.type.startsWith("image/")) {
    log("Not an image: " + file.type, "warn");
    return;
  }

  selectedFile = file;

  // Preview
  $("preview-img").src = URL.createObjectURL(file);
  $("image-name").textContent = file.name;
  $("image-size").textContent = formatSize(file.size);
  $("image-preview").classList.add("visible");

  // Enable Run button
  $("segment-btn").disabled = false;

  log("Loaded: " + file.name + " (" + formatSize(file.size) + ")");
}

// ── Clear ──────────────────────────────────────────────────
$("clear-image").addEventListener("click", function() {
  $("image-preview").classList.remove("visible");
  imageInput.value = "";
  selectedFile = null;
  $("segment-btn").disabled = true;

  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "";
  $("metrics-bar").style.display = "none";
  $("actions-bar").style.display = "none";
});

// ── Run Inference ──────────────────────────────────────────
$("segment-btn").addEventListener("click", async function() {
  const btn = $("segment-btn");

  if (!selectedFile) {
    log("No image selected.", "warn");
    return;
  }
  if (!modelReady) {
    log("Model not ready.", "warn");
    return;
  }

  btn.disabled = true;

  // Loading state
  $("viewport-loading").classList.remove("hidden");
  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "none";

  const t0 = performance.now();
  log("Running inference…");

  try {
    const form = new FormData();
    form.append("image", selectedFile);

    const res = await fetch("/api/segment", {
      method: "POST",
      body: form,
    });

    console.log("[DFU] status:", res.status);

    if (!res.ok) {
      let err = "Server error " + res.status;
      try { err = (await res.json()).error || err; } catch {}
      log("Error: " + err, "error");
      resetView();
      return;
    }

    const data = await res.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    // Results
    $("res-original").src = "data:image/jpeg;base64," + data.original;
    $("res-overlay").src  = "data:image/jpeg;base64," + data.overlay;

    $("dl-overlay").href = "data:image/png;base64," + data.overlay;
    $("dl-mask").href    = "data:image/png;base64," + data.mask;

    // Metrics
    const cov = parseFloat(data.coverage_pct);
    $("metric-coverage").textContent = cov.toFixed(1) + "%";
    $("metric-coverage").classList.remove("pending");

    const totalPx = data.image_size.width * data.image_size.height;
    const fgPx = Math.round(totalPx * cov / 100);
    $("metric-pixels").textContent = fgPx.toLocaleString();
    $("metric-pixels").classList.remove("pending");

    $("metric-size").textContent = data.image_size.width + "x" + data.image_size.height;
    $("metric-size").classList.remove("pending");

    $("metric-time").textContent = elapsed + "s";
    $("metric-time").classList.remove("pending");

    // Reveal
    $("metrics-bar").style.display = "flex";
    $("actions-bar").style.display = "flex";
    $("viewport-loading").classList.add("hidden");
    $("viewport-grid").style.display = "grid";

    log("Done: " + elapsed + "s, " + cov.toFixed(1) + "% coverage", "success");

  } catch (err) {
    console.error("[DFU] error:", err);
    log("Error: " + err.message, "error");
    resetView();
  } finally {
    btn.disabled = false;
  }
});

function resetView() {
  $("viewport-loading").classList.add("hidden");
  $("viewport-empty").style.display = "";
}

// ── Boot ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function() {
  boot();
  log("Session initialized.");
});
