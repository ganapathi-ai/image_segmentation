/* app.js — Diabetic Foot Ulcer Segmentation
   U-Net + ResNet34 · ONNX Runtime inference
   */

let modelReady = false;
let selectedFile = null;

const $  = (id) => document.getElementById(id);

function log(msg, type) {
  const bar = $("log-bar");
  const time = new Date().toLocaleTimeString("en-US", { hour12: false });
  const line = document.createElement("div");
  line.className = "log-line";
  const cls = type ? " log-msg-" + type : "";
  line.innerHTML = `<span class="log-time">${time}</span><span class="log-msg${cls}">${msg}</span>`;
  bar.appendChild(line);
  bar.scrollTop = bar.scrollHeight;
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

function setStatus(dot, txt, state, label) {
  dot.className = "status-dot " + (state || "");
  txt.textContent = label;
}

// ── File Input ─────────────────────────────────────────────
const imageInput = $("image-input");
const imageDrop  = $("image-drop");
const segmentBtn = $("segment-btn");

// Click the drop zone opens file picker
imageDrop.addEventListener("click", function(e) {
  e.preventDefault();
  console.log("[DFU] dropzone clicked, opening file picker");
  imageInput.click();
});

// File picker changed
imageInput.addEventListener("change", function() {
  console.log("[DFU] file input changed, files:", this.files.length);
  if (this.files && this.files[0]) {
    onFileSelected(this.files[0]);
  }
});

// Drag and drop
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
  console.log("[DFU] drop event, files:", e.dataTransfer.files.length);
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    onFileSelected(e.dataTransfer.files[0]);
  }
});

// ── File Selected ──────────────────────────────────────────
function onFileSelected(file) {
  console.log("[DFU] onFileSelected:", file.name, file.type, file.size);

  if (!file.type.startsWith("image/")) {
    log("Not an image file: " + file.type, "warn");
    return;
  }

  selectedFile = file;

  // Show preview
  const url = URL.createObjectURL(file);
  $("preview-img").src = url;
  $("image-name").textContent = file.name;
  $("image-size").textContent = formatSize(file.size);
  $("image-preview").classList.add("visible");

  // Enable Run button
  segmentBtn.disabled = false;

  log("Image loaded: " + file.name + " (" + formatSize(file.size) + ")");
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// ── Clear Image ────────────────────────────────────────────
$("clear-image").addEventListener("click", function() {
  $("image-preview").classList.remove("visible");
  imageInput.value = "";
  selectedFile = null;
  segmentBtn.disabled = true;

  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "";
  $("metrics-bar").style.display = "none";
  $("actions-bar").style.display = "none";
});

// ── Run Inference ──────────────────────────────────────────
segmentBtn.addEventListener("click", async function() {
  if (!selectedFile) {
    log("No image selected.", "warn");
    return;
  }
  if (!modelReady) {
    log("Model not ready. Please wait.", "warn");
    return;
  }

  segmentBtn.disabled = true;
  segmentBtn.textContent = "Running…";

  // Show loading
  $("viewport-loading").classList.remove("hidden");
  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "none";

  const t0 = performance.now();
  log("Running inference on " + selectedFile.name + "…");

  try {
    const form = new FormData();
    form.append("image", selectedFile);

    log("Uploading image to server…");

    const res = await fetch("/api/segment", {
      method: "POST",
      body: form,
    });

    console.log("[DFU] Response:", res.status, res.statusText);
    log("Server responded: " + res.status, res.ok ? "success" : "");

    if (!res.ok) {
      let errMsg = "Server error " + res.status;
      try {
        const errData = await res.json();
        errMsg = errData.error || errMsg;
      } catch {}
      log("Error: " + errMsg, "error");
      resetUI();
      return;
    }

    const data = await res.json();
    console.log("[DFU] Result keys:", Object.keys(data));
    log("Processing results…", "success");

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    // Show results
    $("res-original").src = "data:image/jpeg;base64," + data.original;
    $("res-overlay").src  = "data:image/jpeg;base64," + data.overlay;

    // Download links
    $("dl-overlay").href = "data:image/png;base64," + data.overlay;
    $("dl-mask").href    = "data:image/png;base64," + data.mask;

    // Metrics
    const coverage = parseFloat(data.coverage_pct);
    $("metric-coverage").textContent = coverage.toFixed(1) + "%";
    $("metric-coverage").classList.remove("pending");

    const totalPx = data.image_size.width * data.image_size.height;
    const fgPx = Math.round(totalPx * coverage / 100);
    $("metric-pixels").textContent = fgPx.toLocaleString();
    $("metric-pixels").classList.remove("pending");

    $("metric-size").textContent = data.image_size.width + "×" + data.image_size.height;
    $("metric-size").classList.remove("pending");

    $("metric-time").textContent = elapsed + "s";
    $("metric-time").classList.remove("pending");

    // Reveal results
    $("metrics-bar").style.display = "flex";
    $("actions-bar").style.display = "flex";
    $("viewport-loading").classList.add("hidden");
    $("viewport-grid").style.display = "grid";

    log("Done: " + elapsed + "s · " + coverage.toFixed(1) + "% coverage · " + fgPx.toLocaleString() + " px", "success");

  } catch (err) {
    console.error("[DFU] Error:", err);
    log("Error: " + err.message, "error");
    resetUI();
  } finally {
    segmentBtn.disabled = false;
    segmentBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Inference';
  }
});

function resetUI() {
  $("viewport-loading").classList.add("hidden");
  $("viewport-empty").style.display = "";
}

// ── Boot ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function() {
  boot();
  log("Session initialized.");
});
