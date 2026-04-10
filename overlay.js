function majorityVote(values, fallback = 0) {
  if (!values.length) {
    return fallback;
  }
  const counts = new Map();
  for (const val of values) {
    counts.set(val, (counts.get(val) || 0) + 1);
  }
  let best = fallback;
  let bestCount = -1;
  for (const [key, count] of counts.entries()) {
    if (count > bestCount) {
      bestCount = count;
      best = key;
    }
  }
  return best;
}

export class Overlay {
  constructor(canvas, labels, inputSize) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.labels = labels;
    this.inputSize = inputSize;
    this.usableEMA = 0;
    this.bboxEMA = null;
    this.viewHistory = [];
    this.goodHistory = [];
    this.lowUsableCount = 0;
    this.goodState = false;
    this.lastInfo = null;
    this.displayWidth = canvas.width;
    this.displayHeight = canvas.height;
    this.dpr = 1;
  }

  setDisplaySize(width, height, dpr) {
    this.displayWidth = width;
    this.displayHeight = height;
    this.dpr = dpr || 1;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  _smoothBBox(bbox) {
    if (!this.bboxEMA) {
      this.bboxEMA = bbox.slice();
      return this.bboxEMA;
    }
    const alpha = 0.2;
    this.bboxEMA = this.bboxEMA.map((v, i) => v * (1 - alpha) + bbox[i] * alpha);
    return this.bboxEMA;
  }

  _updateHistory(arr, val, maxLen) {
    arr.push(val);
    if (arr.length > maxLen) {
      arr.shift();
    }
  }

  _bboxToOrigPixel(bboxNorm, meta) {
    const [cx, cy, w, h] = bboxNorm;
    const size = this.inputSize;
    const boxW = w * size;
    const boxH = h * size;
    const x1L = cx * size - boxW / 2;
    const y1L = cy * size - boxH / 2;
    const x2L = x1L + boxW;
    const y2L = y1L + boxH;
    const x1 = (x1L - meta.padX) / meta.scale;
    const y1 = (y1L - meta.padY) / meta.scale;
    const x2 = (x2L - meta.padX) / meta.scale;
    const y2 = (y2L - meta.padY) / meta.scale;
    const x1c = Math.max(0, Math.min(meta.origW, x1));
    const y1c = Math.max(0, Math.min(meta.origH, y1));
    const x2c = Math.max(0, Math.min(meta.origW, x2));
    const y2c = Math.max(0, Math.min(meta.origH, y2));
    return [x1c, y1c, x2c, y2c];
  }

  _bboxToPixel(bboxNorm, meta) {
    const bboxOrig = this._bboxToOrigPixel(bboxNorm, meta);
    return this._mapToDisplay(bboxOrig, meta);
  }

  _mapToDisplay(bboxPix, meta) {
    const [x1, y1, x2, y2] = bboxPix;
    const displayW = this.displayWidth || this.canvas.width;
    const displayH = this.displayHeight || this.canvas.height;
    const scale = Math.min(displayW / meta.origW, displayH / meta.origH);
    const offsetX = (displayW - meta.origW * scale) / 2;
    const offsetY = (displayH - meta.origH * scale) / 2;
    const nx1 = x1 * scale + offsetX;
    const ny1 = y1 * scale + offsetY;
    const nx2 = x2 * scale + offsetX;
    const ny2 = y2 * scale + offsetY;
    return [
      Math.max(0, Math.min(displayW, nx1)),
      Math.max(0, Math.min(displayH, ny1)),
      Math.max(0, Math.min(displayW, nx2)),
      Math.max(0, Math.min(displayH, ny2)),
    ];
  }

  _guidance(usableProb, bboxPix, viewIdx, meta) {
    const usableThresh = 0.6;
    const edgeMargin = 0.06;
    const areaMin = 0.08;
    const areaMax = 0.75;
    const [x1, y1, x2, y2] = bboxPix;
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const area = (w * h) / (meta.origW * meta.origH + 1e-6);
    const x1n = x1 / meta.origW;
    const y1n = y1 / meta.origH;
    const x2n = x2 / meta.origW;
    const y2n = y2 / meta.origH;

    if (usableProb < usableThresh) {
      this.lowUsableCount += 1;
    } else {
      this.lowUsableCount = 0;
    }
    if (this.lowUsableCount >= 10) {
      return { text: "Point camera at the vehicle", good: false };
    }

    const edgesOk =
      x1n > edgeMargin &&
      y1n > edgeMargin &&
      x2n < 1 - edgeMargin &&
      y2n < 1 - edgeMargin;
    const areaOk = area > areaMin && area < areaMax;
    const goodCandidate = usableProb >= usableThresh && edgesOk && areaOk;
    this._updateHistory(this.goodHistory, goodCandidate, 8);
    const goodCount = this.goodHistory.filter(Boolean).length;

    if (!this.goodState && goodCount >= 6) {
      this.goodState = true;
    }
    const lastThree = this.goodHistory.slice(-3);
    if (this.goodState && lastThree.length === 3 && lastThree.every((v) => !v)) {
      this.goodState = false;
    }

    if (this.goodState) {
      const view = this.labels[viewIdx] || "unknown";
      return { text: `Good: ${view}. Hold steady...`, good: true };
    }

    if (x1n <= edgeMargin) {
      return { text: "Move phone left", good: false };
    }
    if (x2n >= 1 - edgeMargin) {
      return { text: "Move phone right", good: false };
    }
    if (y1n <= edgeMargin) {
      return { text: "Tilt down", good: false };
    }
    if (y2n >= 1 - edgeMargin) {
      return { text: "Tilt up", good: false };
    }
    if (area < areaMin) {
      return { text: "Move closer", good: false };
    }
    if (area > areaMax) {
      return { text: "Move back", good: false };
    }
    const view = this.labels[viewIdx] || "unknown";
    return { text: `Adjust framing (${view})`, good: false };
  }

  _singleFrameGood(usableProb, bboxOrig, meta) {
    const usableThresh = 0.6;
    const edgeMargin = 0.06;
    const areaMin = 0.08;
    const areaMax = 0.75;
    const [x1, y1, x2, y2] = bboxOrig;
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const area = (w * h) / (meta.origW * meta.origH + 1e-6);
    const x1n = x1 / meta.origW;
    const y1n = y1 / meta.origH;
    const x2n = x2 / meta.origW;
    const y2n = y2 / meta.origH;
    const edgesOk =
      x1n > edgeMargin &&
      y1n > edgeMargin &&
      x2n < 1 - edgeMargin &&
      y2n < 1 - edgeMargin;
    const areaOk = area > areaMin && area < areaMax;
    return usableProb >= usableThresh && edgesOk && areaOk;
  }

  update(bboxNorm, usableProb, viewIdx, meta) {
    this.usableEMA = this.usableEMA * 0.8 + usableProb * 0.2;
    const smoothBBox = this._smoothBBox(Array.from(bboxNorm));
    this._updateHistory(this.viewHistory, viewIdx, 8);
    const majorityView = majorityVote(this.viewHistory, viewIdx);
    const viewName = this.labels[majorityView] || "unknown";
    const bboxOrig = this._bboxToOrigPixel(smoothBBox, meta);
    const bboxPix = this._mapToDisplay(bboxOrig, meta);
    const guidance = this._guidance(this.usableEMA, bboxOrig, majorityView, meta);

    this.lastInfo = {
      bboxPix,
      bboxOrig,
      usableProb: this.usableEMA,
      viewName,
      origW: meta.origW,
      origH: meta.origH,
      displayW: this.displayWidth || this.canvas.width,
      displayH: this.displayHeight || this.canvas.height,
      dpr: this.dpr,
    };
    this._draw(bboxPix, guidance.text, guidance.good, this.usableEMA, viewName);
  }

  renderSingle(bboxNorm, usableProb, viewIdx, meta) {
    this.usableEMA = usableProb;
    this.bboxEMA = Array.from(bboxNorm);
    this.viewHistory = [viewIdx];
    this.goodHistory = [];
    this.lowUsableCount = 0;
    this.goodState = false;
    const viewName = this.labels[viewIdx] || "unknown";
    const bboxOrig = this._bboxToOrigPixel(this.bboxEMA, meta);
    const bboxPix = this._mapToDisplay(bboxOrig, meta);
    const good = this._singleFrameGood(usableProb, bboxOrig, meta);
    this.lastInfo = {
      bboxPix,
      bboxOrig,
      usableProb,
      viewName,
      origW: meta.origW,
      origH: meta.origH,
      displayW: this.displayWidth || this.canvas.width,
      displayH: this.displayHeight || this.canvas.height,
      dpr: this.dpr,
    };
    this._draw(bboxPix, "", good, usableProb, viewName);
  }

  drawGroundTruth(bboxPix, meta) {
    const ctx = this.ctx;
    const [x1, y1, x2, y2] = this._mapToDisplay(bboxPix, meta);
    ctx.save();
    ctx.strokeStyle = "rgba(80,220,120,0.95)";
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(x1, Math.max(8, y1 - 20), 36, 18);
    ctx.fillStyle = "#fff";
    ctx.font = "12px sans-serif";
    ctx.fillText("GT", x1 + 6, Math.max(20, y1 - 6));
    ctx.restore();
  }

  getDebugInfo() {
    return this.lastInfo;
  }

  _draw(bboxPix, text, good, usableProb, viewName) {
    const ctx = this.ctx;
    const w = this.displayWidth || this.canvas.width;
    const h = this.displayHeight || this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    let [x1, y1, x2, y2] = bboxPix;
    x1 = Math.max(0, Math.min(w, x1));
    y1 = Math.max(0, Math.min(h, y1));
    x2 = Math.max(0, Math.min(w, x2));
    y2 = Math.max(0, Math.min(h, y2));
    ctx.font = "16px sans-serif";
    if (usableProb > 0.3) {
      ctx.strokeStyle = good ? "rgba(80,220,120,0.95)" : "rgba(255,90,90,0.95)";
      ctx.lineWidth = 3;
      const bw = x2 - x1;
      const bh = y2 - y1;
      if (bw > 1 && bh > 1) {
        ctx.strokeRect(x1, y1, bw, bh);
        const label = `${viewName} ${usableProb.toFixed(2)}`;
        const labelWidth = ctx.measureText(label).width + 12;
        const lx = Math.max(6, x1);
        const ly = Math.max(20, y1 - 8);
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(lx - 6, ly - 16, labelWidth, 20);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, lx, ly);
      }
    }
    if (usableProb > 0.3) {
      const summary = `${viewName} | ${usableProb.toFixed(2)}`;
      const summaryWidth = ctx.measureText(summary).width + 16;
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(10, 10, summaryWidth, 22);
      ctx.fillStyle = "#fff";
      ctx.fillText(summary, 18, 26);
    }
    if (text) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(0, h - 54, w, 54);
      ctx.font = "18px sans-serif";
      ctx.fillStyle = "#fff";
      ctx.fillText(text, 12, h - 20);
    }
  }

  drawCoveragePanel(viewStatuses) {
    const ctx = this.ctx;
    const w = this.displayWidth || this.canvas.width;
    const labels = [
      ['F', 'front'], ['RE', 'rear'], ['L', 'left'], ['R', 'right'],
      ['FL', 'front_left'], ['FR', 'front_right'], ['RL', 'rear_left'], ['RR', 'rear_right'],
    ];
    const pillW = 36;
    const pillH = 22;
    const gap = 4;
    const cols = 4;
    const padRight = 10;
    const padTop = 10;
    const panelW = cols * pillW + (cols - 1) * gap;
    const startX = w - panelW - padRight;
    const startY = padTop;

    const colors = { uncovered: 'rgba(120,120,120,0.7)', partial: 'rgba(230,160,40,0.85)', full: 'rgba(60,190,90,0.85)' };

    ctx.save();
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < labels.length; i++) {
      const [abbr, key] = labels[i];
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = startX + col * (pillW + gap);
      const y = startY + row * (pillH + gap);
      const status = viewStatuses[key] || 'uncovered';

      ctx.fillStyle = colors[status];
      ctx.beginPath();
      ctx.roundRect(x, y, pillW, pillH, 6);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.fillText(abbr, x + pillW / 2, y + pillH / 2);
    }
    ctx.restore();
  }

  drawWarning(message) {
    const ctx = this.ctx;
    const w = this.displayWidth || this.canvas.width;
    const h = this.displayHeight || this.canvas.height;
    const barH = 48;
    ctx.save();
    ctx.fillStyle = "rgba(200,40,40,0.75)";
    ctx.fillRect(0, h - barH, w, barH);
    ctx.font = "bold 16px sans-serif";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message, w / 2, h - barH / 2);
    ctx.restore();
  }
}
