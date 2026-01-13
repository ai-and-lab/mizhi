# Car Capture Lite - Web Integration Guide

A lightweight car detection and viewpoint classification model optimized for mobile web browsers.

## Model Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `bbox_cxcywh` | `[1, 4]` | Bounding box as normalized `(cx, cy, w, h)` in letterboxed space |
| `view_logits` | `[1, 8]` | Raw logits for 8 viewpoint classes |
| `usable_logit` | `[1, 1]` | Usability score (apply sigmoid for probability) |

**View Classes:** `front`, `rear`, `left`, `right`, `front_left`, `rear_left`, `front_right`, `rear_right`

## Quick Start

### 1. Include ONNX Runtime

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```

### 2. Copy Required Files

```
your-project/
├── model.onnx
├── model.onnx.data
├── labels.json
└── car-capture.js  (see below)
```

### 3. Create Inference Module

Create `car-capture.js`:

```javascript
const INPUT_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export class CarCapture {
  constructor() {
    this.session = null;
    this.labels = null;
  }

  async init(modelPath = './model.onnx', labelsPath = './labels.json') {
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
    });
    const res = await fetch(labelsPath);
    this.labels = await res.json();
  }

  preprocess(sourceCanvas) {
    const srcW = sourceCanvas.width;
    const srcH = sourceCanvas.height;
    const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = Math.floor((INPUT_SIZE - newW) / 2);
    const padY = Math.floor((INPUT_SIZE - newH) / 2);

    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d');

    // Gray letterbox padding
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(sourceCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

    // Convert to normalized tensor
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
    };
  }

  async predict(sourceCanvas) {
    const { input, meta } = this.preprocess(sourceCanvas);

    const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const outputs = await this.session.run({ input: tensor });

    const bbox = Array.from(outputs.bbox_cxcywh.data);
    const viewLogits = Array.from(outputs.view_logits.data);
    const usableLogit = outputs.usable_logit.data[0];

    // Post-process
    const viewIdx = viewLogits.indexOf(Math.max(...viewLogits));
    const usableProb = 1 / (1 + Math.exp(-usableLogit));

    // Convert bbox from letterbox to original image coordinates
    const bboxOriginal = this.bboxToOriginal(bbox, meta);

    return {
      bbox: bboxOriginal,           // [x1, y1, x2, y2] in original image pixels
      bboxNormalized: bbox,         // [cx, cy, w, h] normalized to letterbox
      view: this.labels[viewIdx],   // e.g., "front_left"
      viewIndex: viewIdx,
      viewConfidence: this.softmax(viewLogits)[viewIdx],
      usable: usableProb,           // 0-1 probability
      meta,
    };
  }

  bboxToOriginal(bbox, meta) {
    const [cx, cy, w, h] = bbox;
    const x1 = (cx - w / 2) * INPUT_SIZE;
    const y1 = (cy - h / 2) * INPUT_SIZE;
    const x2 = (cx + w / 2) * INPUT_SIZE;
    const y2 = (cy + h / 2) * INPUT_SIZE;

    // Remove letterbox padding and scale back
    const ox1 = (x1 - meta.padX) / meta.scale;
    const oy1 = (y1 - meta.padY) / meta.scale;
    const ox2 = (x2 - meta.padX) / meta.scale;
    const oy2 = (y2 - meta.padY) / meta.scale;

    // Clamp to image bounds
    return [
      Math.max(0, Math.min(meta.origW, ox1)),
      Math.max(0, Math.min(meta.origH, oy1)),
      Math.max(0, Math.min(meta.origW, ox2)),
      Math.max(0, Math.min(meta.origH, oy2)),
    ];
  }

  softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sum);
  }

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
}
```

## Usage Examples

### Basic Usage

```javascript
import { CarCapture } from './car-capture.js';

const detector = new CarCapture();
await detector.init('./assets/model.onnx', './assets/labels.json');

// From video element
const video = document.getElementById('video');
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
canvas.getContext('2d').drawImage(video, 0, 0);

const result = await detector.predict(canvas);
console.log(result);
// {
//   bbox: [120, 80, 540, 320],     // [x1, y1, x2, y2] pixels
//   view: "front_left",
//   viewConfidence: 0.94,
//   usable: 0.98,
//   ...
// }
```

### Integration with Camera Capture Workflow

```javascript
import { CarCapture } from './car-capture.js';

class CameraCapture {
  constructor() {
    this.detector = new CarCapture();
    this.video = document.createElement('video');
    this.canvas = document.createElement('canvas');
  }

  async init() {
    // Initialize detector
    await this.detector.init();

    // Start camera
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
    });
    this.video.srcObject = stream;
    await this.video.play();
  }

  captureFrame() {
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    this.canvas.getContext('2d').drawImage(this.video, 0, 0);
    return this.canvas;
  }

  async analyzeFrame() {
    const frame = this.captureFrame();
    const result = await this.detector.predict(frame);

    return {
      ...result,
      isGoodCapture: this.detector.isGoodCapture(result),
      guidance: this.getGuidance(result),
    };
  }

  getGuidance(result) {
    if (result.usable < 0.6) return 'Point camera at vehicle';

    const [x1, y1, x2, y2] = result.bbox;
    const { origW, origH } = result.meta;
    const margin = 0.06;

    if (x1 / origW < margin) return 'Move camera left';
    if (x2 / origW > 1 - margin) return 'Move camera right';
    if (y1 / origH < margin) return 'Tilt camera down';
    if (y2 / origH > 1 - margin) return 'Tilt camera up';

    const area = ((x2 - x1) * (y2 - y1)) / (origW * origH);
    if (area < 0.08) return 'Move closer';
    if (area > 0.75) return 'Move back';

    return `Ready: ${result.view}`;
  }
}

// Usage
const capture = new CameraCapture();
await capture.init();

document.getElementById('capture-btn').onclick = async () => {
  const result = await capture.analyzeFrame();

  if (result.isGoodCapture) {
    console.log('Good capture!', result.view);
    // Save or upload the image
  } else {
    console.log('Guidance:', result.guidance);
  }
};
```

### Drawing Bounding Box

```javascript
function drawBbox(ctx, result, color = 'green') {
  const [x1, y1, x2, y2] = result.bbox;

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // Label
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(x1, y1 - 24, 120, 24);
  ctx.fillStyle = 'white';
  ctx.font = '14px sans-serif';
  ctx.fillText(`${result.view} ${(result.usable * 100).toFixed(0)}%`, x1 + 4, y1 - 6);
}
```

## Model Specifications

| Property | Value |
|----------|-------|
| Input Size | 320 x 320 |
| Input Format | RGB, normalized (ImageNet mean/std) |
| Backbone | MobileNetV3-Small |
| Model Size | ~4 MB (ONNX) |
| Inference Time | 30-80ms (mobile browser) |

## Preprocessing Details

The model expects letterboxed input:
1. Resize image to fit within 320x320 while preserving aspect ratio
2. Pad with gray (RGB: 114, 114, 114) to make square
3. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
4. Convert to CHW format (channels first)

## View Classes

| Index | Class |
|-------|-------|
| 0 | front |
| 1 | rear |
| 2 | left |
| 3 | right |
| 4 | front_left |
| 5 | rear_left |
| 6 | front_right |
| 7 | rear_right |

## Browser Compatibility

- Chrome 80+ (recommended)
- Safari 14+
- Firefox 78+
- Edge 80+

Requires WebAssembly support. For best performance on mobile, use Chrome on Android or Safari on iOS.
