import { startCamera, captureFrame } from "./camera.js";
import { CarCapture } from "./car-capture.js";
import { Overlay } from "./overlay.js";

const video = document.getElementById("video");
const overlayCanvas = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const debugToggle = document.getElementById("toggle-debug");
const captureBtn = document.getElementById("capture-btn");
const debugWrap = document.getElementById("debug-wrap");
const debugCanvas = document.getElementById("debug");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let showDebug = false;
let imageCapture = null;
let overlay = null;
let detector = null;

debugToggle.addEventListener("click", () => {
  showDebug = !showDebug;
  debugWrap.style.display = showDebug ? "block" : "none";
});

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
  detector = new CarCapture();
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

  const INPUT_SIZE = CarCapture.INPUT_SIZE;
  debugCanvas.width = INPUT_SIZE;
  debugCanvas.height = INPUT_SIZE;

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

  async function runOnce() {
    await requestLandscape();
    statusEl.textContent = "Capturing...";
    statusEl.style.display = "block";

    // Capture frame
    const bitmap = await captureBitmap();
    captureCanvas.width = bitmap.width;
    captureCanvas.height = bitmap.height;
    captureCtx.drawImage(bitmap, 0, 0);

    // Run inference using CarCapture module
    const result = await detector.predict(captureCanvas);

    resizeOverlay();

    // Render overlay using existing Overlay class
    overlay.renderSingle(
      result.bboxNormalized,
      result.usable,
      result.viewIndex,
      result.meta
    );

    // Draw debug view
    if (showDebug) {
      const dbgCtx = debugCanvas.getContext("2d");
      dbgCtx.drawImage(result.letterCanvas, 0, 0);

      // Draw bbox on letterboxed image
      const [cx, cy, w, h] = result.bboxNormalized;
      const x1 = (cx - w / 2) * INPUT_SIZE;
      const y1 = (cy - h / 2) * INPUT_SIZE;
      const bw = w * INPUT_SIZE;
      const bh = h * INPUT_SIZE;
      dbgCtx.strokeStyle = "rgba(0, 255, 0, 0.9)";
      dbgCtx.lineWidth = 2;
      dbgCtx.strokeRect(x1, y1, bw, bh);
    }

    console.log("prediction", {
      view: result.view,
      viewConfidence: result.viewConfidence.toFixed(3),
      usable: result.usable.toFixed(3),
      bbox: result.bbox.map((v) => Math.round(v)),
      isGoodCapture: detector.isGoodCapture(result),
      guidance: detector.getGuidance(result),
    });

    statusEl.textContent = "";
    statusEl.style.display = "none";
  }

  captureBtn.addEventListener("click", () => {
    runOnce().catch((err) => {
      statusEl.textContent = `Error: ${err.message}`;
      console.error(err);
    });
  });
}

setup().catch((err) => {
  statusEl.textContent = `Error: ${err.message}`;
  console.error(err);
});
