(() => {
  "use strict";

  function initializeLiveCode() {
  document.addEventListener("click", (event) => {
    const drawerClose = event.target.closest?.(".drawer-close");
    if (!drawerClose) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    drawerClose.closest("dialog")?.removeAttribute("open");
  }, true);

  const stage = document.querySelector(".hero-code-stage");
  const source = stage?.querySelector(".hero-live-code pre");
  if (!stage || !source) return;

  const pageTitle = document.querySelector(".bd-article > section > h1");
  const heroCopy = document.querySelector(".hero-copy");
  if (pageTitle && heroCopy) {
    heroCopy.children[0].after(pageTitle);
    const ctaRow = heroCopy.querySelector(":scope > p:last-child");
    for (const node of ctaRow?.childNodes || []) {
      if (node.nodeType === Node.TEXT_NODE && node.textContent.includes("·")) {
        node.remove();
      }
    }
    const primaryCta = ctaRow?.querySelector("a:first-child");
    primaryCta?.insertAdjacentHTML(
      "beforeend",
      '<i class="fa-solid fa-arrow-right" aria-hidden="true"></i>',
    );
  }
  stage.querySelector(":scope > ul")?.remove();

  const workflow = document.querySelector(".workflow-strip");
  if (workflow) {
    workflow.insertAdjacentHTML(
      "beforeend",
      `<div class="workflow-steps" role="list" aria-label="Core data flow">
        <div class="workflow-node" role="listitem"><strong>generate</strong><span>observables</span></div>
        <i class="workflow-arrow fa-solid fa-arrow-right" aria-hidden="true"></i>
        <div class="workflow-node" role="listitem"><strong>Runtime.observe()</strong><span>values</span></div>
        <i class="workflow-arrow fa-solid fa-arrow-right" aria-hidden="true"></i>
        <div class="workflow-node is-optional" role="listitem"><strong>ValueSink</strong><span>optional storage</span></div>
      </div>`,
    );
  }

  const installSection = document.querySelector("section#install");
  const coreSection = document.querySelector("section#core-data-flow");
  if (installSection && coreSection) {
    const details = document.createElement("div");
    details.className = "home-details-grid";
    installSection.before(details);
    details.append(installSection, coreSection);
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isChinese = document.documentElement.lang.startsWith("zh");
  const labels = isChinese
    ? { running: "运行中", paused: "已暂停", pause: "暂停", play: "播放", hint: "悬停可加速 · 空格键暂停/播放" }
    : { running: "running", paused: "paused", pause: "Pause", play: "Play", hint: "Hover to speed up · Space pauses/plays" };
  const series = {
    param_norm: [2.31, 2.36, 2.41, 2.38, 2.46, 2.52, 2.49, 2.57],
    observed_values: [12, 14, 13, 17, 18, 21, 20, 24],
  };

  const header = document.createElement("div");
  header.className = "live-code-head";
  header.innerHTML = `<span class="live-code-label"><span class="live-code-dot"></span><span>train.py — <span data-live-state>${labels.running}</span></span></span><span data-live-step>step 0</span>`;

  const metrics = document.createElement("div");
  metrics.className = "live-code-metrics";
  for (const name of Object.keys(series)) {
    const row = document.createElement("div");
    row.className = "live-code-metric";
    row.innerHTML = `<span class="live-code-metric__label">${name}</span><canvas width="240" height="34" aria-hidden="true" data-series="${name}"></canvas><span class="live-code-metric__value" data-value="${name}">—</span>`;
    metrics.append(row);
  }

  const footer = document.createElement("div");
  footer.className = "live-code-foot";
  footer.innerHTML = `<button class="live-code-control" type="button" aria-pressed="false" data-live-toggle>${labels.pause}</button><span>${labels.hint}</span>`;

  const originalLines = source.innerHTML.replace(/\n$/, "").split("\n");
  source.innerHTML = "";
  const lines = originalLines.map((line, index) => {
    const element = document.createElement("span");
    element.className = "live-code-line";
    element.dataset.line = String(index + 1);
    element.innerHTML = line || " ";
    source.append(element);
    return element;
  });

  stage.prepend(header);
  source.closest(".hero-live-code").after(metrics, footer);

  const stateLabel = header.querySelector("[data-live-state]");
  const stepLabel = header.querySelector("[data-live-step]");
  const toggle = footer.querySelector("[data-live-toggle]");
  const observeLine = Math.max(0, lines.findIndex((line) => line.textContent.includes("runtime.observe")));
  let activeLine = reducedMotion ? observeLine : 0;
  let pointCount = reducedMotion ? series.param_norm.length : 1;
  let step = reducedMotion ? 700 : 0;
  let paused = reducedMotion;
  let timer = 0;
  let fast = false;

  function cssColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function draw(canvas, values, count) {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(80, canvas.clientWidth);
    const height = Math.max(24, canvas.clientHeight);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);

    const visible = values.slice(0, Math.max(1, count));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    context.beginPath();
    visible.forEach((value, index) => {
      const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
      const y = height - 4 - ((value - min) / range) * (height - 8);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = cssColor("--color-accent");
    context.lineWidth = 1.5;
    context.stroke();
  }

  function renderMetrics() {
    for (const [name, values] of Object.entries(series)) {
      const count = Math.min(pointCount, values.length);
      const value = values[count - 1];
      const formatted = name === "param_norm" ? value.toFixed(2) : String(value);
      metrics.querySelector(`[data-value="${name}"]`).textContent = formatted;
      draw(metrics.querySelector(`[data-series="${name}"]`), values, count);
    }
  }

  function render() {
    lines.forEach((line, index) => line.classList.toggle("is-active", index === activeLine));
    stepLabel.textContent = `step ${step}`;
    stateLabel.textContent = paused ? labels.paused : labels.running;
    stage.classList.toggle("is-paused", paused);
    toggle.textContent = paused ? labels.play : labels.pause;
    toggle.setAttribute("aria-pressed", String(paused));
    renderMetrics();
  }

  function schedule() {
    window.clearTimeout(timer);
    if (paused) return;
    timer = window.setTimeout(advance, fast ? 220 : 700);
  }

  function advance() {
    activeLine = (activeLine + 1) % lines.length;
    if (activeLine === observeLine) {
      step += 100;
      pointCount = Math.min(pointCount + 1, series.param_norm.length);
    }
    render();
    schedule();
  }

  function setPaused(next) {
    paused = next;
    render();
    schedule();
  }

  toggle.addEventListener("click", () => setPaused(!paused));
  stage.addEventListener("mouseenter", () => {
    fast = true;
    schedule();
  });
  stage.addEventListener("mouseleave", () => {
    fast = false;
    schedule();
  });
  document.addEventListener("keydown", (event) => {
    const tag = event.target instanceof Element ? event.target.tagName : "";
    if (event.code !== "Space" || ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tag)) return;
    const rect = stage.getBoundingClientRect();
    if (rect.bottom < 0 || rect.top > window.innerHeight) return;
    event.preventDefault();
    setPaused(!paused);
  });
  window.addEventListener("resize", renderMetrics);
  new MutationObserver(renderMetrics).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  render();
  schedule();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeLiveCode, { once: true });
  } else {
    initializeLiveCode();
  }
})();
