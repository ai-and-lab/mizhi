const VIEW_NAMES = ['front', 'rear', 'left', 'right', 'front_left', 'front_right', 'rear_left', 'rear_right'];

export class CoverageSession {
  constructor() {
    this.views = new Map();
    this.active = false;
    this.outOfViewCount = 0;
    VIEW_NAMES.forEach(v => this.views.set(v, 'uncovered'));
  }

  start() {
    this.active = true;
    this.outOfViewCount = 0;
    VIEW_NAMES.forEach(v => this.views.set(v, 'uncovered'));
  }

  stop() {
    this.active = false;
  }

  trackFrame(usableProb) {
    if (usableProb < 0.6) {
      this.outOfViewCount++;
    } else {
      this.outOfViewCount = 0;
    }
  }

  isCarOutOfView() {
    return this.outOfViewCount >= 18;
  }

  confirmView(viewName, isEdgeTouching) {
    const current = this.views.get(viewName);
    if (current === undefined) return;
    if (isEdgeTouching) {
      if (current !== 'full') {
        this.views.set(viewName, 'partial');
      }
    } else {
      this.views.set(viewName, 'full');
    }
  }

  getAllViews() {
    return Object.fromEntries(this.views);
  }

  getProgress() {
    let full = 0, partial = 0, uncovered = 0;
    for (const status of this.views.values()) {
      if (status === 'full') full++;
      else if (status === 'partial') partial++;
      else uncovered++;
    }
    return { full, partial, uncovered };
  }
}
