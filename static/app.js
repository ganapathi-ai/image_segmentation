/* app.js — Diabetic Foot Ulcer Segmentation
   U-Net + ResNet34 · ONNX Runtime inference
   */

let modelReady = false;

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

// ── Boot: check model status ───────────────────────────────
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
      log("Model weights not loaded on server.", "error");
    }
  } catch (err) {
    setStatus(dot, txt, "", "Checking…");
    log("Status check error: " + err.message, "warn");
  }
}

// ── File input handler ─────────────────────────────────────
function handleFile(file) {
  console.log("[DFU] handleFile:", file?.name, file?.type, file?.size);
  if (!file) {
    log("No file selected.", "warn");
    return;
  }
  if (!file.type.startsWith("image/")) {
    log("Unsupported type: " + file.type, "warn");
    return;
  }

  // Show preview
  const url = URL.createObjectURL(file);
  $("preview-img").src = url;
  $("image-name").textContent = file.name;
  $("image-size").textContent = `${(file.size / 1024).toFixed(1)} KB`;
  $("image-preview").classList.add("visible");

  // Store file and enable button
  $("segment-btn")._imageFile = file;
  $("segment-btn").disabled = false;

  log(`Loaded: ${file.name}`);
}

// ── Wire up upload zone ────────────────────────────────────
const imageInput = $("image-input");
const imageDrop  = $("image-drop");

// Click zone → open file picker
imageDrop.addEventListener("click", (e) => {
  e.preventDefault();
  console.log("[DFU] Dropzone clicked");
  imageInput.click();
});

imageDrop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    imageInput.click();
  }
});

// File selected via picker
imageInput.addEventListener("change", () => {
  console.log("[DFU] Input changed:", imageInput.files?.[0]?.name);
  if (imageInput.files?.[0]) {
    handleFile(imageInput.files[0]);
  }
});

// Drag and drop
imageDrop.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
  imageDrop.classList.add("drag-over");
});

imageDrop.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  imageDrop.classList.remove("drag-over");
});

imageDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  imageDrop.classList.remove("drag-over");
  console.log("[DFU] Files dropped:", e.dataTransfer.files?.length);
  const file = e.dataTransfer.files?.[0];
  if (file) handleFile(file);
});

// ── Clear image ────────────────────────────────────────────
$("clear-image")?.addEventListener("click", () => {
  $("image-preview").classList.remove("visible");
  imageInput.value = "";
  $("segment-btn")._imageFile = null;
  $("segment-btn").disabled = true;

  // Hide results
  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "";
  $("metrics-bar").style.display = "none";
  $("actions-bar").style.display = "none";
});

// ── Run Inference ──────────────────────────────────────────
$("segment-btn").addEventListener("click", async () => {
  const btn  = $("segment-btn");
  const file = btn._imageFile;

  if (!file) {
    log("No image file attached to button.", "warn");
    return;
  }
  if (!modelReady) {
    log("Model not ready yet. Please wait.", "warn");
    return;
  }

  btn.disabled = true;

  // Show loading state
  const viewportLoading = $("viewport-loading");
  viewportLoading.classList.remove("hidden");
  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "none";
  $("sidebar-loading").style.display = "block";

  const startTime = performance.now();
  log("Running inference…");

  try {
    const form = new FormData();
    form.append("image", file);

    log(`Sending ${file.name} (${(file.size / 1024).toFixed(1)} KB) to /api/segment`);

    const res  = await fetch("/api/segment", {
      method: "POST",
      body: form,
    });

    console.log("[DFU] Response status:", res.status);
    const data = await res.json();

    if (!res.ok) {
      log("Error: " + (data.error || "Server error"), "error");
      console.error("[DFU] Error response:", data);
      return;
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

    // Display results
    $("res-original").src = "data:image/jpeg;base64," + data.original;
    $("res-overlay").src  = "data:image/jpeg;base64," + data.overlay;

    $("dl-overlay").href = "data:image/png;base64," + data.overlay;
    $("dl-mask").href    = "data:image/png;base64," + data.mask;

    // Metrics
    const coverage = parseFloat(data.coverage_pct);
    $("metric-coverage").textContent = coverage.toFixed(1) + "%";
    $("metric-coverage").classList.remove("pending");

    const totalPixels = data.image_size.width * data.image_size.height;
    const fgPixels = Math.round(totalPixels * coverage / 100);
    $("metric-pixels").textContent = new Intl.NumberFormat().format(fgPixels);
    $("metric-pixels").classList.remove("pending");

    $("metric-size").textContent = `${data.image_size.width}×${data.image_size.height}`;
    $("metric-size").classList.remove("pending");

    $("metric-time").textContent = elapsed + "s";
    $("metric-time").classList.remove("pending");

    // Show results
    $("metrics-bar").style.display = "flex";
    $("actions-bar").style.display = "flex";
    viewportLoading.classList.add("hidden");
    $("viewport-grid").style.display = "grid";
    $("sidebar-loading").style.display = "none";

    log(`Done: ${elapsed}s · ${coverage.toFixed(1)}% coverage · ${new Intl.NumberFormat().format(fgPixels)} px`, "success");

  } catch (err) {
    log("Network error: " + err.message, "error");
    console.error("[DFU] Fetch error:", err);
    viewportLoading.classList.add("hidden");
    $("viewport-empty").style.display = "";
    $("sidebar-loading").style.display = "none";
  } finally {
    btn.disabled = false;
  }
});

// ── Boot ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  boot();
  log("Session initialized.");
});
