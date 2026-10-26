"use strict";
// Renders saved runs from runs/*.json. No API calls. Everything from the data,
// including model-written text, goes in as text, never as HTML.

const $ = (id) => document.getElementById(id);
const WARMTH = ["hot weather only", "light", "mild", "cool", "cold", "freezing"];
const FORMALITY = ["", "loungewear", "casual", "smart casual", "business / dressy", "formal"];
const CATEGORIES = ["top", "bottom", "dress", "outer", "shoes", "accessory"];
// Closet colour names a browser doesn't know (or knows as something else, like
// a purple "indigo"), for the dot on each thumbnail.
const COLOR_NAMES = {
  indigo: "#2e3d5f", brown: "#7a5232", camel: "#c19a6b", rust: "#b7410e", charcoal: "#36454f", cream: "#f3ead3",
  mustard: "#d6a630", burgundy: "#800020", olive: "#6b6b2a",
  green: "#2f5a3a", navy: "#1f2d4d", black: "#19191b", white: "#f6f5f1", brown: "#6e4a2f",
};

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child !== null && child !== undefined && child !== false) node.append(child);
  return node;
}

function notice(kind, lines, title) {
  const box = el("div", { class: `notice ${kind}`, role: kind === "err" ? "alert" : "note" });
  if (title) box.append(el("b", { text: title }), lines.length ? " " : "");
  if (lines.length === 1) box.append(lines[0]);
  else if (lines.length) box.append(el("ul", {}, ...lines.map((line) => el("li", { text: line }))));
  return box;
}

const label = (item) =>
  `${item.colors.join("/")} ${item.pattern !== "solid" ? item.pattern + " " : ""}${item.subcategory}`;

const _ctx = document.createElement("canvas").getContext("2d");
function colorOf(name) {
  const key = (name || "").replace(/\s+/g, "");
  return COLOR_NAMES[key] || (CSS.supports("color", key) ? key : "");
}
// Readable text colour (dark or light) for a given background colour.
function inkFor(color) {
  try {
    _ctx.fillStyle = "#000"; _ctx.fillStyle = color;
    const hex = _ctx.fillStyle;
    const n = parseInt(hex.slice(1), 16);
    const lum = (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
    return lum > 0.6 ? "#1c1b19" : "#fffdf9";
  } catch { return "#1c1b19"; }
}
function fillFor(item) {
  const cols = item.colors.slice(0, 2).map(colorOf).filter(Boolean);
  if (!cols.length) return { bg: "var(--thumb)", ink: "var(--muted)" };
  const bg = cols.length > 1 ? `linear-gradient(135deg, ${cols[0]} 0 62%, ${cols[1]} 62%)` : cols[0];
  return { bg, ink: inkFor(cols[0]) };
}

const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, ...kids) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  for (const kid of kids) if (kid) node.append(kid);
  return node;
}
let _patternId = 0;

// Each garment is drawn by hand in a 100 x 125 box: a body in the item's colour, then
// details (collar, buttons, pockets, seams, laces) in a tone that stays visible on both
// light and dark fabrics. Colours only ever enter as attributes, never as markup.
function drawGarment(item, id, fill, base, sole) {
  const sub = item.subcategory.toLowerCase();
  const cat = item.category;
  const dark = inkFor(base) !== "#1c1b19";
  const rgb = dark ? "255,255,255" : "0,0,0";
  const edge = { stroke: "rgba(0,0,0,.3)", "stroke-width": 0.8, "stroke-linejoin": "round" };
  const g = svg("g");
  const tools = (p) => ({
    ln: (d, w = 0.9, op = 0.3, dash) => p.append(svg("path", {
      d, fill: "none", stroke: `rgba(${rgb},${op})`, "stroke-width": w, "stroke-linecap": "round",
      ...(dash ? { "stroke-dasharray": dash } : {}) })),
    dot: (x, y, r = 1.3) => p.append(svg("circle", { cx: x, cy: y, r, fill: `rgba(${rgb},.45)` })),
    shp: (d, op = 0.12) => p.append(svg("path", {
      d, fill: `rgba(${rgb},${op})`, stroke: `rgba(${rgb},.3)`, "stroke-width": 0.7, "stroke-linejoin": "round" })),
  });
  const T = tools(g);
  const both = (draw) => {            // draw the left half, then its mirror image
    draw(tools(g));
    const m = svg("g", { transform: "matrix(-1 0 0 1 100 0)" });
    g.append(m);
    draw(tools(m));
  };
  const body = (d) => {
    g.append(svg("path", { d, fill, ...edge }));
    g.append(svg("path", { d, fill: `url(#${id}g)` }));
  };
  const solid = (d, op = 0.1) => {
    g.append(svg("path", { d, fill, ...edge }));
    g.append(svg("path", { d, fill: `rgba(${rgb},${op})` }));
  };
  const buttons = (x, ys) => ys.forEach((y) => T.dot(x, y));

  if (cat === "top" && sub.includes("turtleneck")) {
    body("M39 20L21 27Q13 31 10 46L5 82L16 86L27 54L28 106H72L73 54L84 86L95 82L90 46Q87 31 79 27L61 20Q50 28 39 20Z");
    solid("M39 7Q39 5 41 5H59Q61 5 61 7V21Q50 28 39 21Z", 0.08);
    for (let x = 43; x <= 57; x += 5) T.ln(`M${x} 7V24`, 0.7, 0.22);
    for (let x = 33; x <= 67; x += 5) T.ln(`M${x} 36V100`, 0.6, 0.12);
    T.ln("M28 100H72", 0.9, 0.22);
    both((t) => { t.ln("M5.5 76L17.5 80", 0.9, 0.25); });
  } else if (cat === "top" && /sweater|knit|jumper/.test(sub)) {
    body("M38 18L21 26Q13 30 10 46L5 82L16 86L27 54L28 106H72L73 54L84 86L95 82L90 46Q87 30 79 26L62 18Q50 28 38 18Z");
    T.ln("M38 18Q50 30 62 18", 1.6, 0.3);
    T.ln("M40.5 21.5Q50 32 59.5 21.5", 0.7, 0.18);
    for (let x = 33; x <= 67; x += 5) T.ln(`M${x} 100V106`, 0.7, 0.2);
    T.ln("M28 99H72", 0.9, 0.22);
    both((t) => { t.ln("M5.5 76L17.5 80", 0.9, 0.25); });
  } else if (cat === "top" && sub.includes("camp")) {
    body("M38 15L20 22Q12 26 8 38L4 58L21 63L27 46L28 106H72L73 46L79 63L96 58L92 38Q88 26 80 22L62 15Q50 25 38 15Z");
    both((t) => { t.shp("M38 15L31 31L50 54L50 30Z", 0.14); });
    T.ln("M50 54V106", 0.7, 0.28);
    buttons(50, [64, 78, 92]);
  } else if (cat === "top" && sub.includes("tee")) {
    body("M30 17L16 25Q9 29 7 46L3 82L15 86L27 52L28 108H72L73 52L85 86L97 82L93 46Q91 29 84 25L70 17Q50 28 30 17Z");
    T.ln("M31 21Q50 32 69 21", 0.8, 0.22);
    T.ln("M28 102H72", 0.8, 0.2);
    both((t) => { t.ln("M4.5 77L16 81", 0.8, 0.22); });
  } else if (cat === "top" && sub.includes("blouse")) {
    body("M38 15L20 23Q11 27 8 46L4 84L18 88L27 54L24 108Q50 114 76 108L73 54L82 88L96 84L92 46Q89 27 80 23L62 15Q50 25 38 15Z");
    T.ln("M38 15L50 40L62 15", 1.1, 0.3);
    T.ln("M50 40V110", 0.7, 0.18);
    buttons(50, [52, 66, 80, 94]);
    T.ln("M34 60Q40 82 34 104", 0.8, 0.12);
    T.ln("M66 60Q60 82 66 104", 0.8, 0.12);
    both((t) => { t.ln("M5 78L19 82", 0.9, 0.25); });
  } else if (cat === "top") {                       // oxford / button-down shirt
    body("M38 15L20 23Q11 27 8 46L4 82L16 86L27 52L28 108H72L73 52L84 86L96 82L92 46Q89 27 80 23L62 15Q50 25 38 15Z");
    both((t) => { t.shp("M38 15L33 31L49 36L50 26Z", 0.16); t.ln("M5 75L18 79", 0.9, 0.28); });
    T.ln("M50 36V108", 0.7, 0.28);
    buttons(50, [46, 60, 74, 88, 100]);
    T.shp("M57 50H68V62H57Z", 0.06);
  } else if (cat === "dress") {
    body("M38 12L26 18Q19 22 17 40L12 80L23 83L31 50Q33 66 31 82L32 114H68L69 82Q67 66 69 50L77 83L88 80L83 40Q81 22 74 18L62 12Q50 23 38 12Z");
    T.ln("M40 16Q50 26 60 16", 0.8, 0.22);
    T.ln("M41 46Q44 58 40 68", 0.7, 0.16);
    T.ln("M59 46Q56 58 60 68", 0.7, 0.16);
    T.ln("M32 108H68", 0.8, 0.2);
    both((t) => { t.ln("M13.5 74L23.5 77", 0.8, 0.22); });
  } else if (cat === "outer" && sub.includes("blazer")) {
    body("M36 12L17 21L7 80L19 84L27 48L26 98H74L73 48L81 84L93 80L83 21L64 12L50 34Z");
    both((t) => {
      t.shp("M36 12L29 24L36 29L33 40L46 62L50 62Z", 0.12);
      t.ln("M8 74L20 78", 0.9, 0.25);
      t.shp("M31 76H44V88H31Z", 0.07);
    });
    T.ln("M58 46H67", 0.9, 0.3);
    T.ln("M50 62V98", 0.7, 0.25);
    buttons(51.5, [68, 80]);
  } else if (cat === "outer" && /coat/.test(sub)) {
    body("M36 10L16 20L6 82L18 86L27 50L25 120H75L73 50L82 86L94 82L84 20L64 10L50 30Z");
    both((t) => {
      t.shp("M36 10L29 22L36 27L32 38L45 62L50 62Z", 0.12);
      t.ln("M7 76L19 80", 0.9, 0.25);
      t.ln("M31 94H45", 1, 0.3);
      t.ln("M30 91L31 97", 0.8, 0.25);
    });
    T.ln("M50 62V120", 0.7, 0.25);
    buttons(51.5, [72, 88, 104]);
  } else if (cat === "outer") {                     // trucker jacket / overshirt
    const isDenim = sub.includes("trucker") || sub.includes("jacket");
    const hem = isDenim ? 94 : 102;
    body(`M36 12L18 20L7 78L19 82L27 46L26 ${hem}H74L73 46L81 82L93 78L82 20L64 12L50 28Z`);
    both((t) => {
      t.shp("M36 12L30 27L44 33L50 28Z", 0.16);
      t.shp("M31 42H44V50L37.5 53L31 50Z", 0.1);
      t.ln("M8 72L20 76", 0.9, 0.25);
      if (isDenim) { t.ln("M37 54V86", 0.8, 0.3, "2 1.6"); t.ln(`M26 87H37`, 0.8, 0.25); }
      else { t.shp("M31 70H44V80L37.5 83L31 80Z", 0.1); }
    });
    T.ln(`M50 28V${hem}`, 0.7, 0.28);
    buttons(51.5, isDenim ? [38, 52, 66, 80] : [40, 56, 72, 88]);
    if (isDenim) T.ln(`M26 87H74`, 0.8, 0.25, "2 1.6");
  } else if (cat === "bottom" && sub.includes("wide")) {
    body("M27 12H73L86 118H56L50 58L44 118H14Z");
    T.ln("M27 21H73", 0.9, 0.3);
    T.ln("M50 21V40", 0.7, 0.22);
    both((t) => { t.ln("M29 22L38 46", 0.9, 0.28); t.ln("M37 30L29 118", 0.8, 0.2); });
    T.dot(50, 15.5, 1.4);
  } else if (cat === "bottom" && sub.includes("jeans")) {
    body("M28 12H72L76 118H55L50 52L45 118H24Z");
    T.ln("M28 21H72", 0.9, 0.3);
    [34, 42, 58, 66].forEach((x) => T.ln(`M${x} 12V21`, 0.8, 0.22));
    T.ln("M50 21V42Q50 47 45 49", 0.8, 0.3, "2 1.6");
    both((t) => { t.ln("M30 22Q32 35 42 37", 0.9, 0.3, "2 1.6"); t.ln("M24 112H45", 0.8, 0.25, "2 1.6"); });
    T.dot(50, 16, 1.6);
  } else if (cat === "bottom" && sub.includes("chino")) {
    body("M29 12H71L74 116H54L50 50L46 116H26Z");
    T.ln("M29 20H71", 0.9, 0.3);
    [35, 43, 57, 65].forEach((x) => T.ln(`M${x} 12V20`, 0.8, 0.22));
    T.ln("M50 20V40", 0.7, 0.25);
    both((t) => { t.ln("M31 21L41 42", 0.9, 0.3); t.ln("M27 110H46", 0.8, 0.22); });
    T.dot(50, 15.5, 1.5);
  } else if (cat === "bottom") {                    // tailored trousers
    body("M29 12H71L73 118H55L50 54L45 118H27Z");
    T.ln("M29 21H71", 0.9, 0.3);
    T.ln("M38 24V118", 0.8, 0.2);
    T.ln("M62 24V118", 0.8, 0.2);
    T.ln("M50 21V40", 0.7, 0.22);
    both((t) => { t.ln("M31 22L40 44", 0.9, 0.28); });
    T.dot(50, 15.5, 1.4);
  } else if (cat === "shoes" && sub.includes("boot")) {
    g.append(svg("rect", { x: 13, y: 84, width: 24, height: 10, rx: 2, fill: sole }));
    body("M16 90V32Q16 28 20 28H46Q50 28 50 32V58Q54 64 66 66Q88 68 96 78Q99 82 98 90Z");
    T.shp("M26 38Q36 34 46 38L47 62Q36 66 25 62Z", 0.2);
    for (let y = 44; y <= 60; y += 4) T.ln(`M26.5 ${y}Q36 ${y - 3} 46.5 ${y}`, 0.6, 0.22);
    solid("M19 30V18Q19 16 21 16H24Q26 16 26 18V30Z", 0.12);
    g.append(svg("rect", { x: 11, y: 88, width: 89, height: 7, rx: 3, fill: sole }));
    T.ln("M70 68Q74 80 72 88", 0.8, 0.22);
  } else if (cat === "shoes" && sub.includes("loafer")) {
    g.append(svg("rect", { x: 9, y: 85, width: 20, height: 9, rx: 2, fill: sole }));
    body("M10 90V64Q10 60 14 60H30Q35 60 40 64Q48 70 62 72Q84 74 94 82Q98 86 97 90Z");
    T.shp("M43 66L60 71L58 79L41 73Z", 0.16);
    T.ln("M46 71L56 74.5", 0.8, 0.35);
    T.ln("M72 75Q69 83 71 90", 0.8, 0.22);
    g.append(svg("rect", { x: 8, y: 89, width: 91, height: 5, rx: 2.5, fill: sole }));
  } else if (cat === "shoes" && sub.includes("derby")) {
    g.append(svg("rect", { x: 9, y: 85, width: 20, height: 9, rx: 2, fill: sole }));
    body("M10 90V62Q10 58 14 58H28Q34 58 38 62Q46 68 58 70Q82 72 94 80Q98 84 97 90Z");
    ["M40 63L36 69", "M45 66L41 72", "M51 68L47 74", "M57 70L53 76"].forEach((d) => T.ln(d, 1.3, 0.5));
    T.shp("M30 58Q34 54 38 56L44 60L38 64Q34 62 30 58Z", 0.1);
    T.ln("M72 72Q68 80 70 90", 0.8, 0.25);
    g.append(svg("rect", { x: 8, y: 89, width: 91, height: 6, rx: 3, fill: sole }));
  } else if (cat === "shoes") {                     // sneakers
    body("M10 86V56Q10 52 14 52H25Q29 52 32 48L39 49Q43 58 52 61Q74 63 88 70Q97 74 97 86Z");
    T.shp("M10 86V64Q32 66 46 76Q56 80 62 86Z", 0.07);
    ["M41 52L38 59", "M46 56L43 63", "M51 59L48 66", "M56 61L53 68"].forEach((d) => T.ln(d, 1.4, 0.4));
    T.shp("M10 58H16V76H10Z", 0.12);
    T.ln("M80 70Q76 78 78 86", 0.8, 0.25);
    g.append(svg("rect", { x: 7, y: 82, width: 92, height: 13, rx: 6, fill: sole, stroke: "rgba(0,0,0,.25)", "stroke-width": 0.8 }));
    T.ln("M12 88H93", 0.7, 0.18);
  } else if (cat === "accessory" && sub.includes("silk")) {
    body("M28 14Q50 8 72 14L68 96Q50 102 32 96Z");
    for (let y = 28, row = 0; y <= 88; y += 15, row++) {
      for (let x = row % 2 ? 44 : 37; x <= 66; x += 15) {
        g.append(svg("circle", { cx: x, cy: y, r: 2.4, fill: `rgba(${rgb},.22)` }));
      }
    }
    T.ln("M31 56Q50 61 69 56", 0.8, 0.14);
    T.ln("M30 34Q50 38 70 34", 0.8, 0.1);
  } else if (cat === "accessory") {                 // folded scarf
    body("M30 10H70V98H30Z");
    T.ln("M30 38H70", 0.8, 0.18);
    T.ln("M30 66H70", 0.8, 0.18);
    for (let x = 36; x <= 64; x += 7) T.ln(`M${x} 12V96`, 0.6, 0.1);
    for (let x = 32; x <= 68; x += 3.2) {
      g.append(svg("path", { d: `M${x} 97V110`, fill: "none", stroke: base, "stroke-width": 1.7, "stroke-linecap": "round" }));
      g.append(svg("path", { d: `M${x} 97V110`, fill: "none", stroke: "rgba(0,0,0,.25)", "stroke-width": 0.5 }));
    }
  } else {
    body("M25 20H75V105H25Z");
  }
  return g;
}

function garmentSvg(item) {
  const [c1, c2] = item.colors.map(colorOf);
  const base = c1 || "#bdbdbd";
  const id = `pat${_patternId++}`;
  const root = svg("svg", { viewBox: "0 0 100 125", class: "garment", "aria-hidden": "true", focusable: "false" });
  const defs = svg("defs");
  // light from the upper left: a soft white-to-dark wash over the colour
  defs.append(svg("linearGradient", { id: `${id}g`, x1: "0", y1: "0", x2: "1", y2: "1" },
    svg("stop", { offset: "0", "stop-color": "#fff", "stop-opacity": ".22" }),
    svg("stop", { offset: "1", "stop-color": "#000", "stop-opacity": ".18" })));
  let fill = base;
  if (item.pattern === "striped" && c2) {
    defs.append(svg("pattern", { id, width: 100, height: 12, patternUnits: "userSpaceOnUse" },
      svg("rect", { width: 100, height: 12, fill: base }), svg("rect", { y: 6, width: 100, height: 6, fill: c2 })));
    fill = `url(#${id})`;
  } else if (item.pattern !== "solid" && c2) {
    defs.append(svg("pattern", { id, width: 14, height: 14, patternUnits: "userSpaceOnUse" },
      svg("rect", { width: 14, height: 14, fill: base }),
      svg("rect", { width: 7, height: 14, fill: c2, opacity: ".55" }),
      svg("rect", { width: 14, height: 7, fill: c2, opacity: ".55" })));
    fill = `url(#${id})`;
  }
  root.append(defs);
  // shoes: a pale sole for white sneakers, a dark one for leather and suede
  const sole = item.subcategory.toLowerCase().includes("sneaker") ? "#f1efe9" : item.subcategory.toLowerCase().includes("derby") ? "#7a5a42" : "#2b2420";
  const shape = drawGarment(item, id, fill, base, sole);
  if (item.category === "shoes") shape.setAttribute("transform", item.subcategory.toLowerCase().includes("boot") ? "translate(0 5)" : "translate(0 -10)");
  root.append(shape);
  return root;
}

function thumb(item) {
  const box = el("div", { class: "thumb", "aria-hidden": "true" });
  box.append(garmentSvg(item));
  return box;
}

function itemRow(item, detailed) {
  const parts = [item.fabric, `warmth ${item.warmth}`, FORMALITY[item.formality]];
  if (detailed && item.tags.length) parts.push(item.tags.join(", "));
  const text = el("div", {}, el("div", { class: "name", text: label(item) }),
    el("div", { class: "meta", text: parts.join(" · ") }));
  if (detailed && item.notes) text.append(el("div", { class: "meta", text: item.notes }));
  return el("div", { class: "item" }, thumb(item), text);
}

function fmtDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-GB",
    { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}
const deg = (n) => `${Math.round(n)}°C`;

// ---- tabs ---------------------------------------------------------------
const TABS = ["today", "closet", "buy"];
function selectTab(name, push = true) {
  if (!TABS.includes(name)) name = "today";
  for (const button of document.querySelectorAll("nav button")) {
    const active = button.dataset.tab === name;
    button.setAttribute("aria-selected", String(active));
    $(`tab-${button.dataset.tab}`).classList.toggle("hidden", !active);
  }
  if (push) history.replaceState(null, "", name === "today" ? location.pathname : `#${name}`);
}
for (const button of document.querySelectorAll("nav button")) {
  button.addEventListener("click", () => { selectTab(button.dataset.tab); window.scrollTo(0, 0); });
}
window.addEventListener("hashchange", () => selectTab(location.hash.slice(1), false));

// ---- today --------------------------------------------------------------
function renderRunList(runs, onPick) {
  const list = $("run-list");
  list.replaceChildren();
  runs.forEach((run, index) => {
    const button = el("button", { type: "button", "aria-pressed": "false", text: run.label });
    button.addEventListener("click", () => {
      for (const other of list.children) other.setAttribute("aria-pressed", String(other === button));
      onPick(run);
    });
    list.append(button);
    if (index === 0) button.setAttribute("aria-pressed", "true");
  });
}

function field(key, value, wide) {
  return el("div", { class: `field${wide ? " wide" : ""}` },
    el("span", { class: "k", text: key }), el("span", { class: "v", text: value }));
}

function renderInputs(run) {
  const f = run.forecast;
  const formality = run.inputs.formality ? `${run.inputs.formality} · ${FORMALITY[run.inputs.formality]}` : "No preference";
  return el("div", {},
    el("div", { class: "fields" },
      field("Location", f.location), field("Date", fmtDate(f.date)),
      field("Occasion", run.inputs.occasion || "—", true),
      field("Formality", formality),
      field("Closet", `demo, ${run.report ? run.report.closet_item_count : 15} items`)),
    el("p", { class: "blurb", text: run.blurb }));
}

function renderTrace(run) {
  const f = run.forecast;
  const report = run.report;
  const c = report ? report.constraints : run.rules_constraints;
  const usedModelForWeather = c.source === "rules+llm";
  const steps = el("ol");
  steps.append(el("li", {}, el("span", { class: "tag code", text: "code" }), " ",
    `Forecast: ${deg(f.temp_min_c)} to ${deg(f.temp_max_c)}, ${f.conditions}, ${f.precipitation_probability_pct}% rain` +
    (f.precipitation_mm ? ` (${f.precipitation_mm} mm)` : "") + `, wind ${Math.round(f.wind_kph)} kph. A fixed forecast, so the run is reproducible.`));
  steps.append(el("li", {}, el("span", { class: "tag code", text: "rules" }), " ",
    `Weather rules: warmth ${run.rules_constraints.min_warmth}–${run.rules_constraints.max_warmth}` +
    (run.rules_constraints.needs_waterproof_outer ? ", needs a waterproof layer" : "") +
    (run.ambiguous ? ". The day is ambiguous, so the rules ask Claude." : ". The day is clear-cut, so Claude isn't asked.")));
  if (usedModelForWeather) {
    steps.append(el("li", {}, el("span", { class: "tag model", text: "Claude" }), " ",
      "Weather agent refines the constraints: layering, fabrics, whether an umbrella is enough."));
  }
  const stylistCalls = run.model_calls.filter((p) => !p.includes("forecast"));
  stylistCalls.forEach((purpose, i) => {
    steps.append(el("li", {}, el("span", { class: "tag model", text: "Claude" }), " ",
      i === 0 ? `Stylist picks outfits from the ${report ? report.closet_item_count : 15}-item closet.`
        : "Stylist is asked again, told which picks were rejected and why."));
  });
  steps.append(el("li", {}, el("span", { class: "tag code", text: "code" }), " ",
    report && report.warnings.length
      ? `Guards check every pick and record ${report.warnings.length} warning${report.warnings.length > 1 ? "s" : ""}.`
      : "Guards check every pick: real ids, a complete outfit, no repeats. Nothing was dropped."));
  const model = report && report.model ? ` Model: ${report.model}.` : "";
  return el("div", { class: "trace" },
    el("b", { text: "How this run was made" }),
    el("span", { class: "muted", text: ` · ${run.model_calls.length} model call${run.model_calls.length === 1 ? "" : "s"}.${model}` }),
    steps);
}


const SLOT_ORDER = { outer: 0, top: 1, dress: 1, bottom: 2, shoes: 3, accessory: 4 };
const firstSentence = (text) => (text.match(/^.*?[.!?](?=\s|$)/) || [text])[0];

function renderWeather(c) {
  const range = `${Math.round(c.temp_min_c)}–${Math.round(c.temp_max_c)}°C`;
  const lead = firstSentence(c.layering_advice);
  const rest = c.layering_advice.slice(lead.length).trim();
  const card = el("div", { class: "card weather" },
    el("div", { class: "place", text: `${c.location} · ${fmtDate(c.date)}` }),
    el("div", { class: "temp", text: range }),
    el("div", { class: "cond", text: `${c.temp_band}, ${c.conditions}` }),
    el("div", { class: "stat-row" },
      el("span", { text: `${c.precipitation_probability_pct}% rain` }),
      el("span", { text: `wind ${Math.round(c.wind_kph)} kph` }),
      el("span", { text: c.min_warmth === c.max_warmth ? `warmth ${c.min_warmth}` : `warmth ${c.min_warmth}–${c.max_warmth}` })),
    el("p", { class: "advice", text: lead }));
  const more = [rest, c.notes].filter(Boolean);
  if (more.length) {
    card.append(el("details", { class: "more" }, el("summary", { text: "More about the weather" }),
      ...more.map((t) => el("p", { class: "advice", text: t }))));
  }
  return card;
}

function renderPiece(item) {
  return el("div", { class: "piece", role: "group", "aria-label": label(item) },
    el("div", { class: "chip", "aria-hidden": "true" }, garmentSvg(item)),
    el("div", { class: "pname", "aria-hidden": "true", text: item.subcategory }),
    el("div", { class: "pmeta", "aria-hidden": "true", text: item.fabric }));
}

function renderConfidence(value) {
  const pct = Math.round(value * 100);
  const level = pct >= 80 ? "good" : pct >= 65 ? "mid" : "low";
  const word = { good: "Strong match", mid: "Good option", low: "A compromise" }[level];
  const fill = el("span");
  fill.style.width = `${pct}%`;
  return el("div", { class: `conf ${level}`, role: "img", "aria-label": `Stylist confidence ${pct}%, ${word}` },
    el("div", { class: "meter" }, fill),
    el("span", { title: "Stylist confidence" }, el("b", { text: word }), ` · ${pct}%`));
}

function paletteDots(items) {
  const colours = [...new Set(items.map((i) => colorOf(i.colors[0])).filter(Boolean))].slice(0, 5);
  return el("span", { class: "palette", "aria-hidden": "true" }, ...colours.map((c) => {
    const dot = el("i", { class: "pdot" });
    dot.style.background = c;
    return dot;
  }));
}

function renderOutfit(outfit, index) {
  const card = el("div", { class: `card outfit${index === 0 ? " top-pick" : ""}` },
    el("div", { class: "outfit-head" },
      el("span", { class: "lhs" },
        el("span", { class: "badge", text: index === 0 ? "Top pick" : `Option ${outfit.rank}` }),
        paletteDots(outfit.items)),
      el("span", { class: "muted", text: FORMALITY[outfit.formality] })),
    el("h3", {}, outfit.name));
  const pieces = [...outfit.items].sort((x, y) => (SLOT_ORDER[x.category] ?? 9) - (SLOT_ORDER[y.category] ?? 9));
  card.append(el("div", { class: "lay" }, ...pieces.map(renderPiece)));
  card.append(el("p", { class: "lead", text: firstSentence(outfit.rationale) }));
  card.append(renderConfidence(outfit.confidence));
  card.append(el("details", { class: "more" },
    el("summary", { text: "Full reasoning" }),
    el("p", { class: "why" }, el("b", { text: "Why: " }), outfit.rationale),
    el("p", { class: "why" }, el("b", { text: "Weather: " }), outfit.weather_fit)));
  return card;
}

function renderRecommendation(run) {
  $("sample-flag").classList.toggle("hidden", !run.sample);
  const about = el("details", { class: "about" }, el("summary", { text: "About this run" }),
    renderInputs(run), renderTrace(run));
  about.open = false;
  $("rec-about").replaceChildren(about);
  const out = $("rec-out");
  out.replaceChildren();
  $("rec-weather").replaceChildren();
  if (!run.report) {
    out.append(notice("err", [run.error || "This run produced no result."], "The run failed."));
    return;
  }
  const data = run.report;
  $("rec-weather").append(renderWeather(data.constraints));
  data.suggestions.forEach((outfit, i) => out.append(renderOutfit(outfit, i)));
  if (data.overall_notes) {
    out.append(el("div", { class: "card notes-card" }, el("p", {}, el("b", { text: "Closet notes: " }), data.overall_notes)));
  }
  if (data.warnings.length) out.append(notice("warn", data.warnings, data.warnings.length === 1 ? "Warning:" : "Warnings"));
}

// ---- closet -------------------------------------------------------------
function renderCloset(closet) {
  const out = $("closet-out");
  out.replaceChildren();
  for (const category of CATEGORIES) {
    const group = closet.items.filter((item) => item.category === category);
    if (!group.length) continue;
    out.append(el("h2", { text: `${category} · ${group.length}` }));
    out.append(el("div", { class: "wardrobe" }, ...group.map((item) =>
      el("div", { class: "gcard" },
        el("div", { class: "chip", "aria-hidden": "true" }, garmentSvg(item)),
        el("div", { class: "gname", text: label(item) }),
        el("div", { class: "gmeta", text: `${item.fabric} · warmth ${item.warmth}` }),
        item.notes ? el("div", { class: "gnote", text: item.notes }) : null))));
  }
}

// ---- buy next -----------------------------------------------------------
function renderBuy(run, closet) {
  const out = $("buy-out");
  out.replaceChildren();
  const data = run.report;
  const s = data.stats;
  const byId = Object.fromEntries((closet ? closet.items : []).map((i) => [i.id, i]));

  out.append(el("p", { class: "muted", text:
    "Each candidate is added to the closet and scored by the same rules as everything you own. " +
    "Only looks the closet can't already make count. No model, no network." }));
  out.append(el("div", { class: "card stats" },
    ...[[s.item_count, "items"], [`${s.compatible_pair_count}/${s.total_pair_count}`, "pairs that work"],
      [s.distinct_look_count, "distinct looks today"]].map(([n, l]) =>
      el("div", { class: "stat" }, el("span", { class: "n", text: String(n) }), el("span", { class: "l", text: l })))));

  const top = Math.max(1, ...data.recommendations.map((r) => r.new_combination_count));
  const pchip = (id) => {
    const item = byId[id];
    return el("span", { class: "pchip" },
      item ? el("span", { class: "mini", "aria-hidden": "true" }, garmentSvg(item)) : null,
      item ? label(item) : id);
  };
  for (const r of data.recommendations) {
    const cand = r.candidate;
    const price = cand.price ? ` · price ${Math.round(cand.price)}` : "";
    const dupe = r.verdict === "redundant" && r.new_combination_count > 0;
    const card = el("div", { class: `card cand${r.rank === 1 || dupe ? " highlight" : ""}` },
      el("div", { class: "cand-top" },
        el("div", { class: "chip cand-tile", "aria-hidden": "true" }, garmentSvg(cand)),
        el("div", { class: "cand-main" },
          el("span", { class: `verdict ${r.verdict}`, text: r.verdict.replace("_", " ") }),
          el("h3", {}, el("span", { class: "rank", text: `${r.rank}.` }), label(cand)),
          el("div", { class: "meta muted", text: `${cand.fabric} · warmth ${cand.warmth} · ${FORMALITY[cand.formality]}${price}` }))));
    if (r.verdict !== "not_scored") {
      card.append(el("div", { class: "gain" },
        el("span", {}, el("b", { text: `+${r.new_outfit_count}` }), " new looks"),
        el("span", { class: "muted", text: `${Math.round(r.marginal_gain_pct)}% more variety · ${r.new_combination_count} raw combinations` })));
      const bar = el("div", { class: "bar", role: "img",
        "aria-label": `${r.new_outfit_count} new looks out of ${r.new_combination_count} combinations` });
      const fill = el("span");
      fill.style.width = `${(100 * r.new_outfit_count) / top}%`;
      bar.append(fill);
      card.append(bar);
    }
    card.append(el("p", { class: "why", text: r.explanation }));
    if (cand.notes) card.append(el("p", { class: "muted note", text: cand.notes }));
    const chips = (title, ids) => ids.length
      ? el("div", { class: "chips" }, el("span", { class: "k", text: title }), ...ids.map(pchip))
      : null;
    for (const row of [chips("Pairs with", r.top_partners), chips("Overlaps", r.redundant_with)]) {
      if (row) card.append(row);
    }
    for (const b of r.bridged_pairs.slice(0, 3)) {
      card.append(el("p", { class: "muted", text: `Bridges ${b.label_a} ↔ ${b.label_b}` }));
    }
    out.append(card);
  }
}

// ---- load ---------------------------------------------------------------
async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function main() {
  selectTab(location.hash.slice(1), false);
  let index;
  try {
    index = await getJson("runs/index.json");
  } catch (error) {
    const msg = location.protocol === "file:"
      ? "Open this page over http (python3 -m http.server), not from a file. The browser blocks loading the saved runs from file://."
      : `Could not load the saved runs (${error.message}).`;
    $("rec-out").replaceChildren(notice("err", [msg]));
    return;
  }
  const [closet, recRuns, buyRuns] = await Promise.all([
    index.closet ? getJson(`runs/${index.closet}`).catch(() => null) : null,
    Promise.all(index.recommend.map((n) => getJson(`runs/${n}`).catch(() => null))),
    Promise.all(index.suggest_buy.map((n) => getJson(`runs/${n}`).catch(() => null))),
  ]);
  // Real runs first; hand-written samples only if nothing else is there.
  const runs = recRuns.filter(Boolean).sort((a, b) => (a.sample - b.sample) || (a.order - b.order));

  if (runs.length) {
    renderRunList(runs, renderRecommendation);
    renderRecommendation(runs[0]);
  } else {
    $("rec-out").replaceChildren(el("p", { class: "muted", text: "No saved runs yet." }));
  }
  if (closet) renderCloset(closet);
  else $("closet-out").replaceChildren(el("p", { class: "muted", text: "The closet file is missing." }));
  const buy = buyRuns.find(Boolean);
  if (buy) renderBuy(buy, closet);
  else $("buy-out").replaceChildren(el("p", { class: "muted", text: "No saved run yet." }));
}

main();
