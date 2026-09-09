/* app.js — Segment, U-Net Image Segmentation
   Design: Impeccable + Taste-Skill principles
   — Spring physics (ease-out expo), no bounce
   — Scroll reveals via IntersectionObserver
   — Staggered entry, spring hover feedback
   — Geist + JetBrains Mono token system
   */

// ── State ──────────────────────────────────────────────────
let weightsLoaded = false;

// ── Boot — check if weights are preloaded ──────────────────
async function boot() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.weights_loaded) {
      weightsLoaded = true;
      const drop = $("weights-section");
      if (drop) drop.style.display = "none";
      unlockBlock("image-section");
      unlockBlock("segment-section");
    }
  } catch {
    // Silently fail — user can upload weights manually
  }
}

// ── Helpers ────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

const spring = "cubic-bezier(0.16, 1, 0.3, 1)";

function setStatus(el, msg, type) {
  el.textContent = msg;
  el.className = `status-msg ${type || ''}`;
}

function setupDropzone(dropId, inputId, onFile) {
  const drop  = $(dropId);
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

function b64ToUrl(b64, mime = "image/jpeg") {
  return `data:${mime};base64,${b64}`;
}

// ── Unlock a step block ────────────────────────────────────
function unlockBlock(blockId) {
  const block = $(blockId);
  if (block) {
    block.classList.add("unlocked");
    block.removeAttribute("data-locked");
  }
}

// ── Scroll reveal — staggered, spring-based ────────────────
function initScrollReveal() {
  // Only reveal elements that are NOT locked app-blocks
  const targets = $$(".step, .hero-float-card, .bento-cell");
  const lockedBlocks = $$(".app-block[data-locked]");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const idx = Array.from(targets).indexOf(el);
          const delay = Math.min(idx * 70, 500);

          setTimeout(() => {
            el.style.opacity = "1";
            el.style.transform = "translateY(0)";
            el.style.transition =
              `opacity 0.7s ${spring}, transform 0.7s ${spring}`;

            // Start perpetual float after reveal settles
            if (el.classList.contains("hero-float-card")) {
              setTimeout(() => {
                el.style.animation = `heroFloat 5s ease-in-out infinite`;
              }, 700);
            }
          }, delay);
          observer.unobserve(el);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -30px 0px" }
  );

  targets.forEach((el) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(18px)";
    observer.observe(el);
  });
}

// ── Simulated inference progress bar ───────────────────────
function startInferenceProgress(onDone) {
  const fill = $("inference-fill");
  const text = fill?.parentElement?.nextElementSibling;
  let progress = 0;
  const interval = setInterval(() => {
    // Ease-in-out feel: slow start, fast middle, slow end
    const remaining = 100 - progress;
    const increment = Math.max(0.4, remaining * 0.04 + Math.random() * 1.5);
    progress = Math.min(progress + increment, 92);
    if (fill) fill.style.width = progress + "%";
    if (text) text.textContent = `Running U-Net inference… ${Math.round(progress)}%`;
    if (progress >= 92) {
      clearInterval(interval);
      setTimeout(() => {
        if (fill) fill.style.width = "100%";
        if (text) text.textContent = "Complete.";
        setTimeout(onDone, 200);
      }, 300);
    }
  }, 80);
}

function resetProgress() {
  const fill = $("inference-fill");
  if (fill) fill.style.width = "0%";
}

// ── Step 1: Weights Upload ─────────────────────────────────
setupDropzone("weights-drop", "weights-input", async (file) => {
  const status = $("weights-status");
  setStatus(status, "Loading weights…", "");

  const form = new FormData();
  form.append("file", file);

  try {
    const res  = await fetch("/api/upload-weights", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      setStatus(status, data.error || "Upload failed", "error");
      return;
    }
    weightsLoaded = true;
    setStatus(status, "Weights loaded — ready for inference.", "success");

    const drop = $("weights-drop");
    drop.classList.add("loaded");

    // Unlock step 2
    unlockBlock("image-section");
  } catch (err) {
    setStatus(status, "Network error: " + err.message, "error");
  }
});

// ── Step 2: Image Upload ───────────────────────────────────
setupDropzone("image-drop", "image-input", (file) => {
  if (!file.type.startsWith("image/")) return;

  const url = URL.createObjectURL(file);
  const previewImg = $("preview-img");
  previewImg.src = url;

  $("image-name").textContent  = file.name;
  $("image-size").textContent  = `${(file.size / 1024).toFixed(1)} KB`;

  const previewRow = $("image-preview");
  previewRow.style.display = "flex";
  previewRow.style.animation = `fadeInUp 0.4s ${spring} both`;

  // Enable step 3
  unlockBlock("segment-section");
  $("segment-btn").disabled = false;
  $("segment-btn")._imageFile = file;
});

// Clear image
$("clear-image")?.addEventListener("click", () => {
  $("image-preview").style.display = "none";
  $("image-input").value = "";
  $("segment-btn")._imageFile = null;
  $("segment-btn").disabled = true;
  $("segment-section").setAttribute("data-locked", "true");
  $("segment-section").classList.remove("unlocked");
});

// ── Step 3: Segment ────────────────────────────────────────
$("segment-btn").addEventListener("click", async () => {
  const btn  = $("segment-btn");
  const file = btn._imageFile;
  if (!file) return;

  btn.disabled = true;
  resetProgress();
  $("loading").style.display = "flex";
  $("results-section").style.display = "none";

  const form = new FormData();
  form.append("image", file);

  startInferenceProgress(async () => {
    try {
      const res  = await fetch("/api/segment", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        alert(data.error || "Segmentation failed");
        return;
      }

      // Results
      $("res-original").src = b64ToUrl(data.original);
      $("res-mask").src     = b64ToUrl(data.mask);
      $("res-overlay").src  = b64ToUrl(data.overlay);

      $("dl-overlay").href = b64ToUrl(data.overlay);
      $("dl-mask").href    = b64ToUrl(data.mask);

      $("metric-coverage").textContent = data.coverage_pct + "%";

      $("results-section").style.display = "block";
      $("results-section").style.animation = `fadeInUp 0.7s ${spring} both`;

      setTimeout(() => {
        $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      alert("Network error: " + err.message);
    } finally {
      btn.disabled = false;
    }
  });
});

// ── Boot ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initScrollReveal();
  boot();
});
