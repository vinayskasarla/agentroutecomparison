// Shared helpers for both pages: formatting, theme toggle, tooltip, segmented controls, bar chart.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const STAGES = [
  { kind: "llm", label: "LLM", color: "var(--s1)" },
  { kind: "gateway", label: "AI Gateway", color: "var(--s2)" },
  { kind: "cache", label: "Cache", color: "var(--s3)" },
  { kind: "hop", label: "Network hop", color: "var(--s4)" },
  { kind: "jev", label: "Jev (System 1)", color: "var(--s5)" },
  { kind: "runtime", label: "Agent runtime", color: "var(--s6)" },
  { kind: "error", label: "Failed attempt", color: "var(--crit)" },
];

// ---------- theme
(function theme() {
  let t = null; try { t = localStorage.getItem("plab-theme"); } catch (e) {}
  if (t) document.documentElement.dataset.theme = t;
  $("#theme").onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("plab-theme", next); } catch (e) {}
  };
})();

// ---------- segmented controls
function seg(id, onChange) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    onChange(b.dataset.v);
  });
}

const fmt$ = (v) => v >= 1000 ? "$" + Math.round(v).toLocaleString() : v >= 1 ? "$" + v.toFixed(2) : v >= 0.01 ? "$" + v.toFixed(3) : "$" + v.toFixed(4);
const fmtMs = (v) => v < 0.1 ? "<0.1 ms" : v >= 1000 ? (v / 1000).toFixed(2) + " s" : v >= 10 ? Math.round(v) + " ms" : v.toFixed(1) + " ms";
const fmtPct = (v) => v == null ? "—" : Math.round(v * 100) + "%";

// generic horizontal stacked bar chart (inline SVG)
function hbar(el, rows, segs, fmt, tipFn) {
  const W = el.clientWidth || 520, valW = 64, barH = 18, gap = 14, top = 6;
  const labelW = Math.min(W < 440 ? 118 : 230, Math.max(90, ...rows.map((r) => r.label.length * 6.6 + 14)));
  const plotW = Math.max(80, W - labelW - valW);
  const max = Math.max(...rows.map((r) => segs.reduce((a, s) => a + (r.vals[s.key] || 0), 0)), 1e-9);
  const H = top + rows.length * (barH + gap) + 18;
  const ticks = plotW < 300 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];
  let svg = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img">`;
  ticks.forEach((t) => { const x = labelW + t * plotW;
    svg += `<line class="gridline" x1="${x}" x2="${x}" y1="${top - 4}" y2="${H - 18}"/><text x="${x}" y="${H - 4}" text-anchor="middle">${fmt(max * t)}</text>`; });
  rows.forEach((r, i) => {
    const y = top + i * (barH + gap); let x = labelW; const total = segs.reduce((a, s) => a + (r.vals[s.key] || 0), 0);
    svg += `<text class="lbl" x="${labelW - 10}" y="${y + barH / 2 + 4}" text-anchor="end">${esc(r.label.length * 6.6 > labelW - 12 ? r.label.slice(0, Math.floor((labelW - 12) / 6.6) - 1) + "…" : r.label)}</text>`;
    const drawn = segs.filter((s) => (r.vals[s.key] || 0) / max * plotW >= 0.5);
    drawn.forEach((s, j) => {
      const w = (r.vals[s.key] / max) * plotW, last = j === drawn.length - 1;
      const ww = Math.max(0.5, w - (last ? 0 : 2));
      const rx = last ? Math.min(4, ww / 2) : 0;
      svg += last && ww > 4
        ? `<path d="M${x},${y}h${ww - rx}a${rx},${rx} 0 0 1 ${rx},${rx}v${barH - 2 * rx}a${rx},${rx} 0 0 1 -${rx},${rx}h-${ww - rx}z" fill="${s.color}" data-tip="${esc(tipFn(r, s))}"/>`
        : `<rect x="${x}" y="${y}" width="${ww}" height="${barH}" fill="${s.color}" data-tip="${esc(tipFn(r, s))}"/>`;
      x += w;
    });
    svg += `<rect x="${labelW}" y="${y - 3}" width="${plotW}" height="${barH + 6}" fill="transparent" data-tip="${esc(tipFn(r, null))}" style="pointer-events:${drawn.length ? "none" : "all"}"/>`;
    svg += `<text class="val" x="${labelW + (total / max) * plotW + 6}" y="${y + barH / 2 + 4}">${fmt(total)}</text>`;
  });
  svg += `<line class="baseline" x1="${labelW}" x2="${labelW}" y1="${top - 4}" y2="${H - 18}"/></svg>`;
  el.innerHTML = svg;
}

// tooltip
document.addEventListener("mousemove", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]"), tip = $("#tip");
  if (!t) { tip.style.display = "none"; return; }
  tip.textContent = t.dataset.tip; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 12, innerWidth - 290) + "px"; tip.style.top = e.clientY + 12 + "px";
});
