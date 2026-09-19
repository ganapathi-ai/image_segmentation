/* app.js — Diabetic Foot Ulcer Segmentation
   U-Net + ResNet34 · ONNX Runtime inference
   Professional clinical imaging interface
   */

// ── State ──────────────────────────────────────────────────
let modelReady = false;

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
  const dot = $("status-dot");
  const txt = $("status-text");

  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.weights_loaded) {
      modelReady = true;
      setStatus(dot, txt, "ready", "Model ready");
      log("Model loaded — ready for inference.", "success");
    } else {
      setStatus(dot, txt, "error", "Model unavailable");
      log("Model weights not loaded on server.", "error");
    }
  } catch {
    setStatus(dot, txt, "", "Checking…");
  }
}

// ── Image Upload + Auto Inference ──────────────────────────
setupDropzone("image-drop", "image-input", async (file) => {
  if (!file.type.startsWith("image/")) {
    log("Unsupported file type.", "warn");
    return;
  }

  const url = URL.createObjectURL(file);
  const previewImg = $("preview-img");
  previewImg.src = url;

  $("image-name").textContent = file.name;
  $("image-size").textContent = `${(file.size / 1024).toFixed(1)} KB · ${file.type.split("/")[1].toUpperCase()}`;
  $("image-preview").classList.add("visible");

  log(`Image loaded: ${file.name}`);

  if (!modelReady) {
    log("Model not ready — cannot run inference.", "error");
    return;
  }

  // Auto-run inference
  await runInference(file);
});

$("clear-image")?.addEventListener("click", () => {
  $("image-preview").classList.remove("visible");
  $("image-input").value = "";
  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "";
  $("metrics-bar").style.display = "none";
  $("actions-bar").style.display = "none";
});

// ── Inference ──────────────────────────────────────────────
async function runInference(file) {
  const viewportLoading = $("viewport-loading");
  viewportLoading.classList.remove("hidden");
  $("viewport-grid").style.display = "none";
  $("viewport-empty").style.display = "none";

  const startTime = performance.now();
  log("Running inference…");

  try {
    const form = new FormData();
    form.append("image", file);

    const res  = await fetch("/api/segment", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      log("Inference failed: " + (data.error || "unknown"), "error");
      viewportLoading.classList.add("hidden");
      $("viewport-empty").style.display = "";
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

    const maskBytes = data.mask.length;
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

    log(`Complete: ${elapsed}s · ${coverage.toFixed(1)}% coverage · ${new Intl.NumberFormat().format(fgPixels)} px`, "success");

  } catch (err) {
    log("Network error: " + err.message, "error");
    viewportLoading.classList.add("hidden");
    $("viewport-empty").style.display = "";
  }
}

// ── Boot ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  boot();
  log("Session initialized.");
});
