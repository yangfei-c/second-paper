export function formatVa(va) {
  if (!Array.isArray(va)) return "V 0.00 / A 0.00";
  return `V ${Number(va[0]).toFixed(2)} / A ${Number(va[1]).toFixed(2)}`;
}

export function buildTrajectoryPlot(planned = [], actual = []) {
  const values = [...planned, ...actual].filter(Array.isArray);
  const bounds = getBounds(values.length ? values : [[0, 0]]);
  const point = (va) => ({
    x: 20 + ((Number(va[0]) - bounds.vMin) / (bounds.vMax - bounds.vMin)) * 180,
    y: 140 - ((Number(va[1]) - bounds.aMin) / (bounds.aMax - bounds.aMin)) * 120,
  });
  const polyline = (items) => items.map((va) => {
    const p = point(va); return `${p.x},${p.y}`;
  }).join(" ");
  const nodes = [
    ...planned.map((va, index) => ({ ...point(va), label: index === 0 ? "文" : `音${index}`, type: index === 0 ? "text" : "music" })),
    ...actual.map((va, index) => ({ ...point(va), label: index === 0 ? "用" : `感${index}`, type: index === 0 ? "user" : "felt" })),
  ];
  return { axis: axis(bounds), plannedPolyline: polyline(planned), actualPolyline: polyline(actual), nodes };
}

function getBounds(values) {
  const vs = values.map((x) => Number(x[0]));
  const as = values.map((x) => Number(x[1]));
  const pad = (min, max) => {
    const half = Math.max(0.14, (max - min) * 0.75);
    const mid = (min + max) / 2;
    return [Math.max(-1, mid - half), Math.min(1, mid + half)];
  };
  const [vMin, vMax] = pad(Math.min(...vs), Math.max(...vs));
  const [aMin, aMax] = pad(Math.min(...as), Math.max(...as));
  return { vMin, vMax, aMin, aMax };
}

function axis(bounds) {
  return {
    x: 20 + ((0 - bounds.vMin) / (bounds.vMax - bounds.vMin)) * 180,
    y: 140 - ((0 - bounds.aMin) / (bounds.aMax - bounds.aMin)) * 120,
    showX: bounds.vMin <= 0 && bounds.vMax >= 0,
    showY: bounds.aMin <= 0 && bounds.aMax >= 0,
  };
}
