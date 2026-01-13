/**
 * Mizhi - Standalone Inference Module
 *
 * A modular detection and viewpoint classification module for web apps.
 *
 * Usage:
 *   const detector = new Mizhi();
 *   await detector.init('./model.onnx', './labels.json');
 *   const result = await detector.predict(canvas);
 */

const INPUT_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export class Mizhi {
  constructor() {
    this.session = null;
    this.labels = null;
    this.letterCanvas = null;
  }

  /**
   * Initialize the model and labels
   * @param {string} modelPath - Path to model.onnx file
   * @param {string} labelsPath - Path to labels.json file
   */
  async init(modelPath = './model.onnx', labelsPath = './labels.json') {
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
    });
    const res = await fetch(labelsPath);
    this.labels = await res.json();
  }

  /**
   * Preprocess image for model input
   * @param {HTMLCanvasElement} sourceCanvas - Source image as canvas
   * @returns {Object} - { input: Float32Array, meta: Object, letterCanvas: HTMLCanvasElement }
   */
  preprocess(sourceCanvas) {
    const srcW = sourceCanvas.width;
    const srcH = sourceCanvas.height;
    const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = Math.floor((INPUT_SIZE - newW) / 2);
    const padY = Math.floor((INPUT_SIZE - newH) / 2);

    if (!this.letterCanvas || this.letterCanvas.width !== INPUT_SIZE) {
      this.letterCanvas = document.createElement('canvas');
      this.letterCanvas.width = INPUT_SIZE;
      this.letterCanvas.height = INPUT_SIZE;
    }

    const ctx = this.letterCanvas.getContext('2d');
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(sourceCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const planeSize = INPUT_SIZE * INPUT_SIZE;

    for (let i = 0; i < planeSize; i++) {
      input[i] = (imageData[i * 4] / 255.0 - MEAN[0]) / STD[0];
      input[i + planeSize] = (imageData[i * 4 + 1] / 255.0 - MEAN[1]) / STD[1];
      input[i + 2 * planeSize] = (imageData[i * 4 + 2] / 255.0 - MEAN[2]) / STD[2];
    }

    return {
      input,
      meta: { scale, padX, padY, origW: srcW, origH: srcH },
      letterCanvas: this.letterCanvas,
    };
  }

  /**
   * Run inference on an image
   * @param {HTMLCanvasElement|HTMLVideoElement|HTMLImageElement} source - Image source
   * @returns {Object} - Prediction result
   */
  async predict(source) {
    // Convert source to canvas if needed
    let canvas;
    if (source instanceof HTMLCanvasElement) {
      canvas = source;
    } else {
      canvas = document.createElement('canvas');
      canvas.width = source.videoWidth || source.naturalWidth || source.width;
      canvas.height = source.videoHeight || source.naturalHeight || source.height;
      canvas.getContext('2d').drawImage(source, 0, 0);
    }

    const { input, meta, letterCanvas } = this.preprocess(canvas);

    const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const outputs = await this.session.run({ input: tensor });

    const bbox = Array.from(outputs.bbox_cxcywh.data);
    const viewLogits = Array.from(outputs.view_logits.data);
    const usableLogit = outputs.usable_logit.data[0];

    const viewIdx = this.argmax(viewLogits);
    const usableProb = this.sigmoid(usableLogit);
    const viewProbs = this.softmax(viewLogits);
    const bboxOriginal = this.bboxToOriginal(bbox, meta);

    return {
      bbox: bboxOriginal,
      bboxNormalized: bbox,
      view: this.labels[viewIdx],
      viewIndex: viewIdx,
      viewConfidence: viewProbs[viewIdx],
      viewProbabilities: Object.fromEntries(this.labels.map((l, i) => [l, viewProbs[i]])),
      usable: usableProb,
      meta,
      letterCanvas,
    };
  }

  /**
   * Convert bbox from letterbox space to original image coordinates
   * @param {number[]} bbox - [cx, cy, w, h] normalized
   * @param {Object} meta - Preprocessing metadata
   * @returns {number[]} - [x1, y1, x2, y2] in original image pixels
   */
  bboxToOriginal(bbox, meta) {
    const [cx, cy, w, h] = bbox;
    const x1 = (cx - w / 2) * INPUT_SIZE;
    const y1 = (cy - h / 2) * INPUT_SIZE;
    const x2 = (cx + w / 2) * INPUT_SIZE;
    const y2 = (cy + h / 2) * INPUT_SIZE;

    const ox1 = (x1 - meta.padX) / meta.scale;
    const oy1 = (y1 - meta.padY) / meta.scale;
    const ox2 = (x2 - meta.padX) / meta.scale;
    const oy2 = (y2 - meta.padY) / meta.scale;

    return [
      Math.max(0, Math.min(meta.origW, ox1)),
      Math.max(0, Math.min(meta.origH, oy1)),
      Math.max(0, Math.min(meta.origW, ox2)),
      Math.max(0, Math.min(meta.origH, oy2)),
    ];
  }

  /**
   * Check if the capture meets quality criteria
   * @param {Object} result - Prediction result
   * @param {Object} options - Threshold options
   * @returns {boolean}
   */
  isGoodCapture(result, options = {}) {
    const {
      usableThreshold = 0.6,
      edgeMargin = 0.06,
      minArea = 0.08,
      maxArea = 0.75,
    } = options;

    const [x1, y1, x2, y2] = result.bbox;
    const { origW, origH } = result.meta;

    const w = x2 - x1;
    const h = y2 - y1;
    const area = (w * h) / (origW * origH);

    const x1n = x1 / origW;
    const y1n = y1 / origH;
    const x2n = x2 / origW;
    const y2n = y2 / origH;

    const edgesOk = x1n > edgeMargin && y1n > edgeMargin &&
                    x2n < (1 - edgeMargin) && y2n < (1 - edgeMargin);
    const areaOk = area > minArea && area < maxArea;
    const usableOk = result.usable >= usableThreshold;

    return usableOk && edgesOk && areaOk;
  }

  /**
   * Get user guidance based on prediction result
   * @param {Object} result - Prediction result
   * @returns {Object} - { message: string, isReady: boolean }
   */
  getGuidance(result) {
    const margin = 0.06;

    if (result.usable < 0.6) {
      return { message: 'Point camera at vehicle', isReady: false };
    }

    const [x1, y1, x2, y2] = result.bbox;
    const { origW, origH } = result.meta;

    if (x1 / origW < margin) return { message: 'Move camera left', isReady: false };
    if (x2 / origW > 1 - margin) return { message: 'Move camera right', isReady: false };
    if (y1 / origH < margin) return { message: 'Tilt camera down', isReady: false };
    if (y2 / origH > 1 - margin) return { message: 'Tilt camera up', isReady: false };

    const area = ((x2 - x1) * (y2 - y1)) / (origW * origH);
    if (area < 0.08) return { message: 'Move closer', isReady: false };
    if (area > 0.75) return { message: 'Move back', isReady: false };

    return { message: `Ready: ${result.view}`, isReady: true };
  }

  /**
   * Draw bounding box on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Object} result - Prediction result
   * @param {Object} options - Drawing options
   */
  drawBbox(ctx, result, options = {}) {
    const {
      color = result.usable > 0.6 ? 'rgba(80, 220, 120, 0.9)' : 'rgba(255, 90, 90, 0.9)',
      lineWidth = 3,
      showLabel = true,
    } = options;

    const [x1, y1, x2, y2] = result.bbox;

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (showLabel && result.usable > 0.3) {
      const label = `${result.view} ${(result.usable * 100).toFixed(0)}%`;
      ctx.font = '14px sans-serif';
      const labelWidth = ctx.measureText(label).width + 12;

      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(x1, Math.max(0, y1 - 24), labelWidth, 24);

      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 + 6, Math.max(16, y1 - 6));
    }
  }

  // Utility functions
  argmax(arr) {
    return arr.indexOf(Math.max(...arr));
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sum);
  }

  // Static constants
  static get INPUT_SIZE() {
    return INPUT_SIZE;
  }

  static get LABELS() {
    return ['front', 'rear', 'left', 'right', 'front_left', 'rear_left', 'front_right', 'rear_right'];
  }
}

export default Mizhi;
