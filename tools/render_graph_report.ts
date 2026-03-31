#!/usr/bin/env bun

import { readFile, writeFile } from "node:fs/promises";

type GraphNode = {
    id: number;
    name: string;
    op: string;
    layer_index: number | null;
    depth: number;
    is_on_critical_path: boolean;
    bottleneck: string;
    total_bytes: number;
};

type GraphHotspot = {
    id: number;
    name: string;
    estimated_share_pct: number;
    bottleneck: string;
};

type CriticalPathStep = {
    id: number;
    name: string;
    depth: number;
};

type GraphReport = {
    name: string;
    node_count: number;
    edge_count: number;
    critical_path_node_count: number;
    max_parallel_width: number;
    assumed_decode_seq_len: number;
    total_bytes: number;
    total_flops: number;
    hardware?: {
        bandwidth_gbps?: number;
        compute_units?: number;
    };
    nodes: GraphNode[];
    hotspots: GraphHotspot[];
    critical_path: CriticalPathStep[];
};

function escapeHtml(value: string): string {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function fmtBytes(value: number): string {
    if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)} GB`;
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} MB`;
    if (value >= 1_000) return `${(value / 1_000).toFixed(1)} KB`;
    return `${value} B`;
}

function fmtFlops(value: number): string {
    if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)} GFLOPs`;
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} MFLOPs`;
    if (value >= 1_000) return `${(value / 1_000).toFixed(1)} KFLOPs`;
    return `${value} FLOPs`;
}

function slug(text: string): string {
    return text
        .toLowerCase()
        .replaceAll(/[^a-z0-9]+/g, "-")
        .replaceAll(/^-+|-+$/g, "");
}

function barRow(label: string, value: number, maxValue: number, hint: string, klass = ""): string {
    const width = maxValue <= 0 ? 0 : (value / maxValue) * 100;
    return `
      <div class="bar-row ${klass}">
        <div class="bar-meta">
          <div class="bar-label">${escapeHtml(label)}</div>
          <div class="bar-hint">${escapeHtml(hint)}</div>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:${width.toFixed(2)}%"></div></div>
      </div>
    `;
}

function hotspotSection(report: GraphReport, nodesById: Map<number, GraphNode>): string {
    const top = report.hotspots.slice(0, 12);
    const maxShare = Math.max(...top.map((entry) => entry.estimated_share_pct), 0);
    return top
        .map((entry) => {
            const node = nodesById.get(entry.id)!;
            const hint = `${entry.estimated_share_pct.toFixed(1)}% • ${fmtBytes(node.total_bytes)} • ${node.bottleneck}`;
            return barRow(entry.name, entry.estimated_share_pct, maxShare, hint, "hotspot");
        })
        .join("");
}

function layerSection(report: GraphReport): string {
    const layerBytes = new Map<number, number>();
    const layerNodes = new Map<number, GraphNode[]>();
    for (const node of report.nodes) {
        if (node.layer_index == null) continue;
        layerBytes.set(node.layer_index, (layerBytes.get(node.layer_index) ?? 0) + node.total_bytes);
        const items = layerNodes.get(node.layer_index) ?? [];
        items.push(node);
        layerNodes.set(node.layer_index, items);
    }
    const ranked = [...layerBytes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 16);
    const maxBytes = Math.max(...ranked.map(([, value]) => value), 0);
    return ranked
        .map(([layer, total]) => {
            const topNodes = (layerNodes.get(layer) ?? []).sort((a, b) => b.total_bytes - a.total_bytes).slice(0, 2);
            const hint = topNodes.map((node) => `${node.name.split(".").at(-1)} ${fmtBytes(node.total_bytes)}`).join(", ") || fmtBytes(total);
            return barRow(`Layer ${layer}`, total, maxBytes, hint, "layer");
        })
        .join("");
}

function opSection(report: GraphReport): string {
    const opBytes = new Map<string, number>();
    const opCount = new Map<string, number>();
    for (const node of report.nodes) {
        opBytes.set(node.op, (opBytes.get(node.op) ?? 0) + node.total_bytes);
        opCount.set(node.op, (opCount.get(node.op) ?? 0) + 1);
    }
    const ranked = [...opBytes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
    const maxBytes = Math.max(...ranked.map(([, value]) => value), 0);
    return ranked
        .map(([op, total]) => barRow(op, total, maxBytes, `${fmtBytes(total)} • ${opCount.get(op) ?? 0} nodes`, `op op-${slug(op)}`))
        .join("");
}

function bottleneckSection(report: GraphReport): string {
    const kindBytes = new Map<string, number>();
    const kindCount = new Map<string, number>();
    for (const node of report.nodes) {
        kindBytes.set(node.bottleneck, (kindBytes.get(node.bottleneck) ?? 0) + node.total_bytes);
        kindCount.set(node.bottleneck, (kindCount.get(node.bottleneck) ?? 0) + 1);
    }
    const ranked = [...kindBytes.entries()].sort((a, b) => b[1] - a[1]);
    const maxBytes = Math.max(...ranked.map(([, value]) => value), 0);
    return ranked
        .map(([kind, total]) => barRow(kind, total, maxBytes, `${fmtBytes(total)} • ${kindCount.get(kind) ?? 0} nodes`, `kind kind-${slug(kind)}`))
        .join("");
}

function insightCards(report: GraphReport, nodesById: Map<number, GraphNode>): string {
    const cards: string[] = [];
    const topHotspot = report.hotspots[0];
    if (topHotspot) {
        cards.push(`
          <div class="insight-card">
            <div class="insight-kicker">Start Here</div>
            <div class="insight-title">${escapeHtml(topHotspot.name)}</div>
            <p>Accounts for <strong>${topHotspot.estimated_share_pct.toFixed(1)}%</strong> of modeled byte traffic and is labeled <strong>${escapeHtml(topHotspot.bottleneck)}</strong>.</p>
          </div>
        `);
    }

    const criticalHotspot = report.hotspots.find((entry) => nodesById.get(entry.id)?.is_on_critical_path);
    if (criticalHotspot) {
        const node = nodesById.get(criticalHotspot.id)!;
        cards.push(`
          <div class="insight-card">
            <div class="insight-kicker">Critical Path</div>
            <div class="insight-title">${escapeHtml(criticalHotspot.name)}</div>
            <p>The heaviest hotspot on the critical path moves <strong>${fmtBytes(node.total_bytes)}</strong> at depth <strong>${node.depth}</strong>.</p>
          </div>
        `);
    }

    const layerBytes = new Map<number, number>();
    for (const node of report.nodes) {
        if (node.layer_index == null) continue;
        layerBytes.set(node.layer_index, (layerBytes.get(node.layer_index) ?? 0) + node.total_bytes);
    }
    const topLayer = [...layerBytes.entries()].sort((a, b) => b[1] - a[1])[0];
    if (topLayer) {
        const [layer, total] = topLayer;
        const share = report.total_bytes <= 0 ? 0 : (total / report.total_bytes) * 100;
        cards.push(`
          <div class="insight-card">
            <div class="insight-kicker">Layer Pressure</div>
            <div class="insight-title">Layer ${layer}</div>
            <p>This layer contributes <strong>${fmtBytes(total)}</strong>, about <strong>${share.toFixed(1)}%</strong> of modeled decode traffic.</p>
          </div>
        `);
    }

    const bottleneckBytes = new Map<string, number>();
    for (const node of report.nodes) {
        bottleneckBytes.set(node.bottleneck, (bottleneckBytes.get(node.bottleneck) ?? 0) + node.total_bytes);
    }
    const dominant = [...bottleneckBytes.entries()].sort((a, b) => b[1] - a[1])[0];
    if (dominant) {
        const [kind, total] = dominant;
        const share = report.total_bytes <= 0 ? 0 : (total / report.total_bytes) * 100;
        cards.push(`
          <div class="insight-card">
            <div class="insight-kicker">Dominant Constraint</div>
            <div class="insight-title">${escapeHtml(kind)}</div>
            <p>Nodes with this label account for <strong>${fmtBytes(total)}</strong>, or <strong>${share.toFixed(1)}%</strong> of total modeled traffic.</p>
          </div>
        `);
    }
    return cards.join("");
}

function criticalPathTable(report: GraphReport, nodesById: Map<number, GraphNode>): string {
    return report.critical_path.slice(0, 40).map((step) => {
        const node = nodesById.get(step.id);
        if (!node) return "";
        return `
          <tr>
            <td>${step.depth}</td>
            <td class="mono">${escapeHtml(step.name)}</td>
            <td>${escapeHtml(node.op)}</td>
            <td>${node.layer_index ?? ""}</td>
            <td>${fmtBytes(node.total_bytes)}</td>
            <td>${escapeHtml(node.bottleneck)}</td>
          </tr>
        `;
    }).join("");
}

function nodeTableRows(report: GraphReport): string {
    const ranked = [...report.nodes].sort((a, b) => b.total_bytes - a.total_bytes);
    return ranked.map((node, index) => {
        const share = report.total_bytes <= 0 ? 0 : (node.total_bytes / report.total_bytes) * 100;
        const search = [node.name, node.op, node.layer_index ?? "", node.bottleneck, node.is_on_critical_path ? "critical" : ""].join(" ").toLowerCase();
        return `
          <tr data-search="${escapeHtml(search)}">
            <td>${index + 1}</td>
            <td class="mono">${escapeHtml(node.name)}</td>
            <td>${escapeHtml(node.op)}</td>
            <td>${node.layer_index ?? ""}</td>
            <td>${node.depth}</td>
            <td>${fmtBytes(node.total_bytes)}</td>
            <td>${share.toFixed(2)}%</td>
            <td>${escapeHtml(node.bottleneck)}</td>
            <td>${node.is_on_critical_path ? "yes" : ""}</td>
          </tr>
        `;
    }).join("");
}

function renderHtml(report: GraphReport): string {
    const nodesById = new Map(report.nodes.map((node) => [node.id, node]));

    const overviewCards = [
        ["Nodes", String(report.node_count)],
        ["Critical Path", String(report.critical_path_node_count)],
        ["Parallel Width", String(report.max_parallel_width)],
        ["Seq Len Assumption", String(report.assumed_decode_seq_len)],
        ["Traffic / Decode", fmtBytes(report.total_bytes)],
        ["Work / Decode", fmtFlops(report.total_flops)],
    ];

    if (report.hardware?.bandwidth_gbps) overviewCards.push(["HW Bandwidth", `${report.hardware.bandwidth_gbps} GB/s`]);
    if (report.hardware?.compute_units) overviewCards.push(["Compute Units", String(report.hardware.compute_units)]);

    const overviewHtml = overviewCards.map(([label, value]) => `
      <div class="metric-card">
        <div class="metric-label">${escapeHtml(label)}</div>
        <div class="metric-value">${escapeHtml(value)}</div>
      </div>
    `).join("");

    return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(report.name)} Graph Report</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 252, 246, 0.88);
      --panel-strong: #fffdfa;
      --text: #1f2937;
      --muted: #5b6470;
      --line: #dccfbf;
      --accent: #9a3412;
      --shadow: 0 18px 50px rgba(91, 100, 112, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(154, 52, 18, 0.10), transparent 25%),
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.08), transparent 24%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }
    .page { max-width: 1500px; margin: 0 auto; padding: 36px 28px 56px; }
    .hero {
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,248,240,0.9));
      border: 1px solid rgba(220, 207, 191, 0.95);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px 30px 24px;
      margin-bottom: 24px;
    }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; color: var(--accent); font-weight: 700; }
    h1 { margin: 10px 0 8px; font-size: 34px; line-height: 1.05; font-weight: 800; }
    .lede { max-width: 960px; color: var(--muted); font-size: 15px; line-height: 1.55; margin: 0; }
    .metric-grid, .insight-grid, .panel-grid { display: grid; gap: 16px; }
    .metric-grid { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-top: 22px; }
    .metric-card, .insight-card, .panel, .table-panel {
      background: var(--panel);
      border: 1px solid rgba(220, 207, 191, 0.95);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }
    .metric-card { padding: 16px 16px 14px; }
    .metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .metric-value { font-size: 24px; font-weight: 750; }
    .insight-grid { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin: 24px 0; }
    .insight-card { padding: 18px 18px 16px; background: linear-gradient(180deg, rgba(255,253,250,0.95), rgba(255,248,240,0.88)); }
    .insight-kicker { color: var(--accent); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .insight-title { font-size: 22px; font-weight: 760; margin-bottom: 8px; }
    .insight-card p { margin: 0; color: var(--muted); line-height: 1.45; font-size: 14px; }
    .panel-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .panel, .table-panel { padding: 20px 20px 18px; }
    .table-panel { margin-top: 18px; }
    .section-title { margin: 0 0 6px; font-size: 21px; font-weight: 760; }
    .section-copy { margin: 0 0 16px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .bar-row { margin-bottom: 12px; }
    .bar-meta { display: flex; justify-content: space-between; gap: 18px; margin-bottom: 7px; align-items: baseline; }
    .bar-label { font-family: "IBM Plex Mono", "Menlo", monospace; font-size: 13px; font-weight: 600; }
    .bar-hint { color: var(--muted); font-size: 12px; text-align: right; }
    .bar-track { height: 13px; background: rgba(220, 207, 191, 0.5); border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg, #d97706, #f59e0b); border-radius: 999px; }
    .layer .bar-fill { background: linear-gradient(90deg, #0f766e, #14b8a6); }
    .kind-kind-occupancy .bar-fill { background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
    .kind-kind-launch-latency .bar-fill { background: linear-gradient(90deg, #7c3aed, #c084fc); }
    .kind-kind-host-sync .bar-fill, .kind-kind-transfer .bar-fill { background: linear-gradient(90deg, #047857, #34d399); }
    .op-op-dmmv .bar-fill { background: linear-gradient(90deg, #9a3412, #ea580c); }
    .op-op-flash-attn .bar-fill { background: linear-gradient(90deg, #1d4ed8, #3b82f6); }
    .mono { font-family: "IBM Plex Mono", "Menlo", monospace; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid rgba(220, 207, 191, 0.7); vertical-align: top; text-align: left; }
    th { color: var(--muted); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
    .controls { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
    input[type="search"] { min-width: 280px; border: 1px solid var(--line); background: var(--panel-strong); border-radius: 999px; padding: 10px 14px; font: inherit; color: inherit; }
    .pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 10px; background: rgba(154, 52, 18, 0.08); color: var(--accent); font-size: 12px; font-weight: 700; }
    @media (max-width: 1080px) { .panel-grid { grid-template-columns: 1fr; } }
    @media (max-width: 720px) {
      .page { padding: 22px 14px 36px; }
      .hero { padding: 22px 18px 18px; }
      h1 { font-size: 28px; }
      .metric-grid, .insight-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      .metric-grid, .insight-grid { grid-template-columns: 1fr; }
      input[type="search"] { min-width: 100%; width: 100%; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Decode Graph Review</div>
      <h1>${escapeHtml(report.name)}</h1>
      <p class="lede">This view compresses the raw graph into the questions you usually care about first: what dominates byte traffic, which layers are heavy, what sits on the critical path, and whether the current analysis thinks you are limited by bandwidth, occupancy, launch size, or host synchronization.</p>
      <div class="metric-grid">${overviewHtml}</div>
    </section>

    <section class="insight-grid">
      ${insightCards(report, nodesById)}
    </section>

    <section class="panel-grid">
      <div class="panel">
        <h2 class="section-title">Top Hotspots</h2>
        <p class="section-copy">These are the heaviest nodes by modeled cost share. In this report, the ranking is driven by byte traffic because the hardware timing floor is not populated.</p>
        ${hotspotSection(report, nodesById)}
      </div>
      <div class="panel">
        <h2 class="section-title">Bottleneck Mix</h2>
        <p class="section-copy">This is byte-weighted, not just raw node counts.</p>
        ${bottleneckSection(report)}
      </div>
      <div class="panel">
        <h2 class="section-title">Layer Pressure</h2>
        <p class="section-copy">The heaviest layers by modeled traffic. This helps spot repeating expensive phases, not just one outlier kernel.</p>
        ${layerSection(report)}
      </div>
      <div class="panel">
        <h2 class="section-title">Operation Mix</h2>
        <p class="section-copy">Total modeled bytes grouped by op type. This is usually the fastest way to decide whether to optimize DMMV, attention, routing, or transfer work first.</p>
        ${opSection(report)}
      </div>
    </section>

    <section class="table-panel">
      <div class="pill">Critical path excerpt</div>
      <h2 class="section-title">Ordered Critical Path</h2>
      <p class="section-copy">The first 40 steps on the critical path, in execution order. Use this when a hotspot matters because it blocks everything behind it.</p>
      <table>
        <thead><tr><th>Depth</th><th>Name</th><th>Op</th><th>Layer</th><th>Bytes</th><th>Bottleneck</th></tr></thead>
        <tbody>${criticalPathTable(report, nodesById)}</tbody>
      </table>
    </section>

    <section class="table-panel">
      <div class="controls">
        <div class="pill">Node explorer</div>
        <input id="node-filter" type="search" placeholder="Filter by name, op, layer, or bottleneck">
      </div>
      <h2 class="section-title">All Nodes By Modeled Cost</h2>
      <p class="section-copy">This table is sorted by total modeled bytes. Filter for <span class="mono">q_proj</span>, <span class="mono">lm_head</span>, <span class="mono">memory_bandwidth</span>, or a layer number such as <span class="mono">7</span>.</p>
      <table>
        <thead><tr><th>#</th><th>Name</th><th>Op</th><th>Layer</th><th>Depth</th><th>Bytes</th><th>Share</th><th>Bottleneck</th><th>Critical</th></tr></thead>
        <tbody id="node-table-body">${nodeTableRows(report)}</tbody>
      </table>
    </section>
  </div>
  <script>
    const filter = document.getElementById("node-filter");
    const rows = Array.from(document.querySelectorAll("#node-table-body tr"));
    filter.addEventListener("input", () => {
      const query = filter.value.trim().toLowerCase();
      for (const row of rows) {
        const haystack = row.dataset.search || "";
        row.style.display = !query || haystack.includes(query) ? "" : "none";
      }
    });
  </script>
</body>
</html>`;
}

async function main() {
    const [, , inputJson, outputHtmlArg] = Bun.argv;
    if (!inputJson) {
        console.error("usage: bun tools/render_graph_report.ts <input.json> [output.html]");
        process.exit(1);
    }

    const outputHtml = outputHtmlArg ?? inputJson.replace(/\.json$/i, ".html");
    const report = JSON.parse(await readFile(inputJson, "utf8")) as GraphReport;
    const htmlText = renderHtml(report);
    await writeFile(outputHtml, htmlText, "utf8");
    console.log(outputHtml);
}

await main();
