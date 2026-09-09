/* app.js — frontend logic for the U-Net segmentation web app */

// ── State ──────────────────────────────────────────────────
let weightsLoaded = false;

// ── Helpers ────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const setStatus = (el, msg, type) => {
  el.textContent = msg;
  el.className = `status-msg ${type || ''}`;
};

function setupDropzone(dropId, inputId, onFile) {
  const drop = $(dropId);
  const input = $(inputId);

  drop.addEventListener("click", () => input.click());

  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("dragover");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) onFile(input.files[0]);
  });
}

function base64ToDataUrl(b64, mime = "image/jpeg") {
  return `data:${mime};base64,${b64}`;
}

function activateSection(section) {
  section.style.opacity = "1";
  section.style.pointerEvents = "auto";
  section.classList.add("active");
}

// ── Step 1: Weights Upload ─────────────────────────────────
setupDropzone("weights-drop", "weights-input", async (file) => {
  const status = $("weights-status");
  setStatus(status, "Loading weights…", "");

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/upload-weights", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(status, data.error || "Upload failed", "error");
      return;
    }
    weightsLoaded = true;
    setStatus(status, "✓ Weights loaded successfully!", "success");
    $("weights-drop").style.borderColor = "var(--success)";
    activateSection($("image-section"));
  } catch (err) {
    setStatus(status, "Network error: " + err.message, "error");
  }
});

// ── Step 2: Image Upload ───────────────────────────────────
setupDropzone("image-drop", "image-input", (file) => {
  if (!file.type.startsWith("image/")) return;

  const url = URL.createObjectURL(file);
  $("preview-img").src = url;
  $("image-name").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  $("image-preview").style.display = "flex";

  // Enable step 3
  activateSection($("segment-section"));
  $("segment-btn").disabled = false;

  // Store file reference for later
  $("segment-btn")._imageFile = file;
});

// ── Step 3: Segment ────────────────────────────────────────
$("segment-btn").addEventListener("click", async () => {
  const btn = $("segment-btn");
  const file = btn._imageFile;
  if (!file) return;

  // Show loading
  btn.disabled = true;
  $("loading").style.display = "flex";
  $("results-section").style.display = "none";

  const form = new FormData();
  form.append("image", file);

  try {
    const res = await fetch("/api/segment", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "Segmentation failed");
      return;
    }

    // Populate results
    $("res-original").src = base64ToDataUrl(data.original);
    $("res-mask").src = base64ToDataUrl(data.mask);
    $("res-overlay").src = base64ToDataUrl(data.overlay);

    // Download links
    $("dl-overlay").href = base64ToDataUrl(data.overlay);
    $("dl-mask").href = base64ToDataUrl(data.mask);

    $("metric-coverage").textContent = data.coverage_pct + "%";

    $("results-section").style.display = "block";
    $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    alert("Network error: " + err.message);
  } finally {
    $("loading").style.display = "none";
    btn.disabled = false;
  }
});
