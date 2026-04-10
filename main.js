import { startCamera, captureFrame } from "./camera.js";
import { Mizhi } from "./mizhi.js";
import { Overlay } from "./overlay.js";
import { CoverageSession } from "./coverage.js";

const video = document.getElementById("video");
const overlayCanvas = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const captureBtn = document.getElementById("capture-btn");
const saveBtn = document.getElementById("save-btn");
const sessionBtn = document.getElementById("session-btn");
const confirmBtn = document.getElementById("confirm-btn");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let imageCapture = null;
let overlay = null;
let detector = null;
let lastResult = null;

// Session state
let session = null;
let animFrameId = null;
let inferenceRunning = false;

async function requestLandscape() {
  try {
    if (document.documentElement.requestFullscreen && !document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    }
    if (screen.orientation && screen.orientation.lock) {
      await screen.orientation.lock("landscape");
    }
  } catch (err) {
    console.warn("Landscape lock skipped:", err);
  }
}

async function setup() {
  statusEl.textContent = "Loading model...";

  // Initialize detector
  detector = new Mizhi();
  await detector.init("./assets/model.onnx", "./assets/labels.json");

  statusEl.textContent = "Starting camera...";
  const stream = await startCamera(video);
  const activeStream = stream || video.srcObject;
  const track =
    activeStream && activeStream.getVideoTracks
      ? activeStream.getVideoTracks()[0]
      : null;
  if (window.ImageCapture && track) {
    imageCapture = new ImageCapture(track);
  }

  function resizeOverlay() {
    const rect = overlayCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(rect.width);
    const height = Math.round(rect.height);
    const targetW = Math.round(width * dpr);
    const targetH = Math.round(height * dpr);
    if (overlayCanvas.width !== targetW || overlayCanvas.height !== targetH) {
      overlayCanvas.width = targetW;
      overlayCanvas.height = targetH;
    }
    if (overlay) {
      overlay.setDisplaySize(width, height, dpr);
    }
  }

  resizeOverlay();
  window.addEventListener("resize", resizeOverlay);

  const INPUT_SIZE = Mizhi.INPUT_SIZE;

  overlay = new Overlay(overlayCanvas, detector.labels, INPUT_SIZE);
  resizeOverlay();
  statusEl.textContent = "";
  statusEl.style.display = "none";

  async function captureBitmap() {
    if (imageCapture && imageCapture.grabFrame) {
      return await imageCapture.grabFrame();
    }
    captureFrame(video, captureCanvas, captureCtx);
    return await createImageBitmap(captureCanvas);
  }

  // --- On-demand capture ---
  async function runOnce() {
    await requestLandscape();
    statusEl.textContent = "Capturing...";
    statusEl.style.display = "block";

    const bitmap = await captureBitmap();
    captureCanvas.width = bitmap.width;
    captureCanvas.height = bitmap.height;
    captureCtx.drawImage(bitmap, 0, 0);
    bitmap.close();

    const result = await detector.predict(captureCanvas);

    resizeOverlay();

    overlay.renderSingle(
      result.bboxNormalized,
      result.usable,
      result.viewIndex,
      result.meta
    );

    lastResult = result;

    console.log("prediction", {
      view: result.view,
      viewConfidence: result.viewConfidence.toFixed(3),
      usable: result.usable.toFixed(3),
      bbox: result.bbox.map((v) => Math.round(v)),
      isGoodCapture: detector.isGoodCapture(result),
      guidance: detector.getGuidance(result),
    });

    saveBtn.style.display = "inline-block";
    statusEl.textContent = "";
    statusEl.style.display = "none";
  }

  function saveFrame() {
    if (!lastResult) return;
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const view = lastResult.view;
    const usable = Math.round(lastResult.usable * 100);
    const filename = `mizhi_${ts}_${view}_u${usable}.jpg`;

    captureCanvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }, "image/jpeg", 0.92);
  }

  // --- Continuous session ---
  async function continuousInfer() {
    if (!session || !session.active) return;
    if (inferenceRunning) {
      animFrameId = requestAnimationFrame(continuousInfer);
      return;
    }

    inferenceRunning = true;
    try {
      const bitmap = await captureBitmap();
      captureCanvas.width = bitmap.width;
      captureCanvas.height = bitmap.height;
      captureCtx.drawImage(bitmap, 0, 0);
      bitmap.close();

      const result = await detector.predict(captureCanvas);
      lastResult = result;

      resizeOverlay();
      overlay.update(
        result.bboxNormalized,
        result.usable,
        result.viewIndex,
        result.meta
      );

      session.trackFrame(overlay.usableEMA);
      overlay.drawCoveragePanel(session.getAllViews());

      if (session.isCarOutOfView()) {
        overlay.drawWarning("Car not detected — point camera at vehicle");
      }
    } catch (err) {
      console.error("Inference error:", err);
    }
    inferenceRunning = false;

    if (session && session.active) {
      animFrameId = requestAnimationFrame(continuousInfer);
    }
  }

  async function startSession() {
    await requestLandscape();
    session = new CoverageSession();
    session.start();
    lastResult = null;

    // Hide on-demand buttons, show confirm
    captureBtn.style.display = "none";
    saveBtn.style.display = "none";
    confirmBtn.style.display = "inline-block";
    sessionBtn.textContent = "End Session";

    animFrameId = requestAnimationFrame(continuousInfer);
  }

  function stopSession() {
    if (session) session.stop();
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
    inferenceRunning = false;
    session = null;
    lastResult = null;

    // Exit fullscreen
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }
    if (screen.orientation && screen.orientation.unlock) {
      screen.orientation.unlock();
    }

    // Clear overlay and restore on-demand buttons
    const ctx = overlayCanvas.getContext("2d");
    const dw = overlay ? (overlay.displayWidth || overlayCanvas.width) : overlayCanvas.width;
    const dh = overlay ? (overlay.displayHeight || overlayCanvas.height) : overlayCanvas.height;
    ctx.clearRect(0, 0, dw, dh);

    captureBtn.style.display = "inline-block";
    confirmBtn.style.display = "none";
    sessionBtn.textContent = "Start Session";
  }

  // --- Event listeners ---
  saveBtn.addEventListener("click", saveFrame);

  captureBtn.addEventListener("click", () => {
    runOnce().catch((err) => {
      statusEl.textContent = `Error: ${err.message}`;
      console.error(err);
    });
  });

  sessionBtn.addEventListener("click", () => {
    if (session && session.active) {
      stopSession();
    } else {
      startSession();
    }
  });

  confirmBtn.addEventListener("click", () => {
    if (!session || !session.active || !lastResult) return;
    if (lastResult.usable < 0.6) return;

    const info = overlay.lastInfo;
    if (!info) return;

    const [x1, y1, x2, y2] = info.bboxOrig;
    const edgeMargin = 0.06;
    const isEdgeTouching =
      x1 / info.origW < edgeMargin ||
      x2 / info.origW > 1 - edgeMargin ||
      y1 / info.origH < edgeMargin ||
      y2 / info.origH > 1 - edgeMargin;

    session.confirmView(info.viewName, isEdgeTouching);

    // Brief visual feedback
    confirmBtn.style.background = isEdgeTouching ? "#e0a020" : "#30b050";
    confirmBtn.textContent = isEdgeTouching ? "Partial!" : "Confirmed!";
    setTimeout(() => {
      confirmBtn.style.background = "";
      confirmBtn.textContent = "Confirm View";
    }, 600);

    const progress = session.getProgress();
    console.log("View confirmed:", info.viewName, isEdgeTouching ? "(partial)" : "(full)", progress);
  });
}

setup().catch((err) => {
  statusEl.textContent = `Error: ${err.message}`;
  console.error(err);
});
