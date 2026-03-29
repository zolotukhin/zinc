#!/usr/bin/env python3
"""Render a ZINC decode-graph JSON report as a readable standalone HTML file."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def fmt_bytes(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} GB"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.1f} KB"
    return f"{value} B"


def fmt_flops(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} GFLOPs"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MFLOPs"
    if value >= 1_000:
        return f"{value / 1_000:.1f} KFLOPs"
    return f"{value} FLOPs"


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def bar_row(label: str, value: float, max_value: float, hint: str, klass: str = "") -> str:
    width = 0.0 if max_value <= 0 else value / max_value * 100.0
    label_html = html.escape(label)
    hint_html = html.escape(hint)
    return (
        f'<div class="bar-row {klass}">'
        f'<div class="bar-meta"><div class="bar-label">{label_html}</div>'
        f'<div class="bar-hint">{hint_html}</div></div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>'
        f"</div>"
    )


def hotspot_section(hotspots: list[dict], nodes_by_id: dict[int, dict]) -> str:
    top = hotspots[:12]
    max_share = max((entry["estimated_share_pct"] for entry in top), default=0.0)
    rows = []
    for entry in top:
        node = nodes_by_id[entry["id"]]
        hint = f'{entry["estimated_share_pct"]:.1f}% • {fmt_bytes(node["total_bytes"])} • {node["bottleneck"]}'
        rows.append(bar_row(entry["name"], entry["estimated_share_pct"], max_share, hint, "hotspot"))
    return "".join(rows)


def layer_section(nodes: list[dict]) -> str:
    by_layer_bytes: dict[int, int] = defaultdict(int)
    by_layer_top: dict[int, list[dict]] = defaultdict(list)
    for node in nodes:
        layer = node.get("layer_index")
        if layer is None:
            continue
        by_layer_bytes[layer] += node["total_bytes"]
        by_layer_top[layer].append(node)

    ranked_layers = sorted(by_layer_bytes.items(), key=lambda item: item[1], reverse=True)[:16]
    max_bytes = max((value for _, value in ranked_layers), default=0)
    rows = []
    for layer, total in ranked_layers:
        top_nodes = sorted(by_layer_top[layer], key=lambda node: node["total_bytes"], reverse=True)[:2]
        top_hint = ", ".join(f'{node["name"].split(".")[-1]} {fmt_bytes(node["total_bytes"])}' for node in top_nodes)
        rows.append(bar_row(f"Layer {layer}", float(total), float(max_bytes), top_hint or fmt_bytes(total), "layer"))
    return "".join(rows)


def op_section(nodes: list[dict]) -> str:
    by_op_bytes: dict[str, int] = defaultdict(int)
    by_op_count: dict[str, int] = defaultdict(int)
    for node in nodes:
        op = node["op"]
        by_op_bytes[op] += node["total_bytes"]
        by_op_count[op] += 1

    ranked = sorted(by_op_bytes.items(), key=lambda item: item[1], reverse=True)[:12]
    max_bytes = max((value for _, value in ranked), default=0)
    rows = []
    for op, total in ranked:
        hint = f'{fmt_bytes(total)} • {by_op_count[op]} nodes'
        rows.append(bar_row(op, float(total), float(max_bytes), hint, f"op op-{slug(op)}"))
    return "".join(rows)


def bottleneck_section(nodes: list[dict]) -> str:
    count_by_kind: Counter[str] = Counter()
    bytes_by_kind: dict[str, int] = defaultdict(int)
    for node in nodes:
        kind = node["bottleneck"]
        count_by_kind[kind] += 1
        bytes_by_kind[kind] += node["total_bytes"]

    ranked = sorted(bytes_by_kind.items(), key=lambda item: item[1], reverse=True)
    max_bytes = max((value for _, value in ranked), default=0)
    rows = []
    for kind, total in ranked:
        hint = f'{fmt_bytes(total)} • {count_by_kind[kind]} nodes'
        rows.append(bar_row(kind, float(total), float(max_bytes), hint, f"kind kind-{slug(kind)}"))
    return "".join(rows)


def critical_path_table(report: dict, nodes_by_id: dict[int, dict]) -> str:
    rows = []
    for step in report["critical_path"][:40]:
        node = nodes_by_id.get(step["id"])
        if node is None:
            continue
        rows.append(
            "<tr>"
            f'<td>{step["depth"]}</td>'
            f'<td class="mono">{html.escape(step["name"])}</td>'
            f'<td>{html.escape(node["op"])}</td>'
            f'<td>{"" if node["layer_index"] is None else node["layer_index"]}</td>'
            f'<td>{fmt_bytes(node["total_bytes"])}</td>'
            f'<td>{html.escape(node["bottleneck"])}</td>'
            "</tr>"
        )
    return "".join(rows)


def node_table_rows(nodes: list[dict], total_bytes: int) -> str:
    ranked = sorted(nodes, key=lambda node: node["total_bytes"], reverse=True)
    rows = []
    for index, node in enumerate(ranked, start=1):
        share = 0.0 if total_bytes <= 0 else node["total_bytes"] / total_bytes * 100.0
        search_text = " ".join(
            str(part)
            for part in (
                node["name"],
                node["op"],
                node.get("layer_index", ""),
                node["bottleneck"],
                "critical" if node["is_on_critical_path"] else "",
            )
        ).lower()
        rows.append(
            f'<tr data-search="{html.escape(search_text)}">'
            f"<td>{index}</td>"
            f'<td class="mono">{html.escape(node["name"])}</td>'
            f'<td>{html.escape(node["op"])}</td>'
            f'<td>{"" if node["layer_index"] is None else node["layer_index"]}</td>'
            f'<td>{node["depth"]}</td>'
            f'<td>{fmt_bytes(node["total_bytes"])}</td>'
            f'<td>{share:.2f}%</td>'
            f'<td>{html.escape(node["bottleneck"])}</td>'
            f'<td>{"yes" if node["is_on_critical_path"] else ""}</td>'
            "</tr>"
        )
    return "".join(rows)


def insight_cards(report: dict, nodes: list[dict], hotspots: list[dict], nodes_by_id: dict[int, dict]) -> str:
    total_bytes = report["total_bytes"]
    top_hotspot = hotspots[0] if hotspots else None
    top_critical = next((entry for entry in hotspots if nodes_by_id.get(entry["id"], {}).get("is_on_critical_path")), None)

    by_layer_bytes: dict[int, int] = defaultdict(int)
    for node in nodes:
        layer = node.get("layer_index")
        if layer is not None:
            by_layer_bytes[layer] += node["total_bytes"]
    top_layer = max(by_layer_bytes.items(), key=lambda item: item[1]) if by_layer_bytes else None

    bytes_by_kind: dict[str, int] = defaultdict(int)
    for node in nodes:
        bytes_by_kind[node["bottleneck"]] += node["total_bytes"]
    top_kind = max(bytes_by_kind.items(), key=lambda item: item[1]) if bytes_by_kind else None

    cards = []
    if top_hotspot:
        cards.append(
            "<div class='insight-card'>"
            "<div class='insight-kicker'>Start Here</div>"
            f"<div class='insight-title'>{html.escape(top_hotspot['name'])}</div>"
            f"<p>Accounts for <strong>{top_hotspot['estimated_share_pct']:.1f}% </strong>of modeled byte traffic and is flagged as <strong>{html.escape(top_hotspot['bottleneck'])}</strong>.</p>"
            "</div>"
        )
    if top_critical:
        node = nodes_by_id[top_critical["id"]]
        cards.append(
            "<div class='insight-card'>"
            "<div class='insight-kicker'>Critical Path</div>"
            f"<div class='insight-title'>{html.escape(top_critical['name'])}</div>"
            f"<p>The heaviest visible node on the critical path moves <strong>{fmt_bytes(node['total_bytes'])}</strong> at depth <strong>{node['depth']}</strong>.</p>"
            "</div>"
        )
    if top_layer:
        layer, total = top_layer
        share = 0.0 if total_bytes <= 0 else total / total_bytes * 100.0
        cards.append(
            "<div class='insight-card'>"
            "<div class='insight-kicker'>Layer Pressure</div>"
            f"<div class='insight-title'>Layer {layer}</div>"
            f"<p>This layer contributes <strong>{fmt_bytes(total)}</strong> of modeled traffic, about <strong>{share:.1f}% </strong>of the decode-step total.</p>"
            "</div>"
        )
    if top_kind:
        kind, total = top_kind
        share = 0.0 if total_bytes <= 0 else total / total_bytes * 100.0
        cards.append(
            "<div class='insight-card'>"
            "<div class='insight-kicker'>Dominant Constraint</div>"
            f"<div class='insight-title'>{html.escape(kind)}</div>"
            f"<p>Nodes with this label account for <strong>{fmt_bytes(total)}</strong>, or <strong>{share:.1f}% </strong>of total modeled traffic.</p>"
            "</div>"
        )
    return "".join(cards)


def render_html(report: dict) -> str:
    nodes = report["nodes"]
    hotspots = report.get("hotspots", [])
    nodes_by_id = {node["id"]: node for node in nodes}
    total_bytes = report["total_bytes"]
    total_flops = report["total_flops"]

    overview_cards = [
        ("Nodes", str(report["node_count"])),
        ("Critical Path", str(report["critical_path_node_count"])),
        ("Parallel Width", str(report["max_parallel_width"])),
        ("Seq Len Assumption", str(report["assumed_decode_seq_len"])),
        ("Traffic / Decode", fmt_bytes(total_bytes)),
        ("Work / Decode", fmt_flops(total_flops)),
    ]
    if report.get("hardware", {}).get("bandwidth_gbps"):
        overview_cards.append(("HW Bandwidth", f'{report["hardware"]["bandwidth_gbps"]} GB/s'))
    if report.get("hardware", {}).get("compute_units"):
        overview_cards.append(("Compute Units", str(report["hardware"]["compute_units"])))

    overview_html = "".join(
        "<div class='metric-card'>"
        f"<div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(value)}</div>"
        "</div>"
        for label, value in overview_cards
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report['name'])} Graph Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 252, 246, 0.88);
      --panel-strong: #fffdfa;
      --text: #1f2937;
      --muted: #5b6470;
      --line: #dccfbf;
      --accent: #9a3412;
      --accent-soft: #fed7aa;
      --red: #b91c1c;
      --amber: #b45309;
      --blue: #1d4ed8;
      --violet: #7c3aed;
      --green: #047857;
      --shadow: 0 18px 50px rgba(91, 100, 112, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(154, 52, 18, 0.10), transparent 25%),
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.08), transparent 24%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 36px 28px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,248,240,0.9));
      border: 1px solid rgba(220, 207, 191, 0.95);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px 30px 24px;
      margin-bottom: 24px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
      color: var(--accent);
      font-weight: 700;
    }}
    h1 {{
      margin: 10px 0 8px;
      font-size: 34px;
      line-height: 1.05;
      font-weight: 800;
    }}
    .lede {{
      max-width: 960px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
      margin: 0;
    }}
    .metric-grid, .insight-grid, .panel-grid {{
      display: grid;
      gap: 16px;
    }}
    .metric-grid {{
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      margin-top: 22px;
    }}
    .metric-card, .insight-card, .panel, .table-panel {{
      background: var(--panel);
      border: 1px solid rgba(220, 207, 191, 0.95);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .metric-card {{
      padding: 16px 16px 14px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 750;
    }}
    .insight-grid {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      margin: 24px 0;
    }}
    .insight-card {{
      padding: 18px 18px 16px;
      background: linear-gradient(180deg, rgba(255,253,250,0.95), rgba(255,248,240,0.88));
    }}
    .insight-kicker {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .insight-title {{
      font-size: 22px;
      font-weight: 760;
      margin-bottom: 8px;
    }}
    .insight-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }}
    .panel-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .panel, .table-panel {{
      padding: 20px 20px 18px;
    }}
    .table-panel {{
      margin-top: 18px;
    }}
    .section-title {{
      margin: 0 0 6px;
      font-size: 21px;
      font-weight: 760;
    }}
    .section-copy {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    .bar-row {{
      margin-bottom: 12px;
    }}
    .bar-meta {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 7px;
      align-items: baseline;
    }}
    .bar-label {{
      font-family: "IBM Plex Mono", "Menlo", monospace;
      font-size: 13px;
      font-weight: 600;
    }}
    .bar-hint {{
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }}
    .bar-track {{
      height: 13px;
      background: rgba(220, 207, 191, 0.5);
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #d97706, #f59e0b);
      border-radius: 999px;
    }}
    .layer .bar-fill {{ background: linear-gradient(90deg, #0f766e, #14b8a6); }}
    .kind-kind-occupancy .bar-fill {{ background: linear-gradient(90deg, #1d4ed8, #60a5fa); }}
    .kind-kind-launch-latency .bar-fill {{ background: linear-gradient(90deg, #7c3aed, #c084fc); }}
    .kind-kind-host-sync .bar-fill, .kind-kind-transfer .bar-fill {{ background: linear-gradient(90deg, #047857, #34d399); }}
    .op-op-dmmv .bar-fill {{ background: linear-gradient(90deg, #9a3412, #ea580c); }}
    .op-op-flash-attn .bar-fill {{ background: linear-gradient(90deg, #1d4ed8, #3b82f6); }}
    .mono {{
      font-family: "IBM Plex Mono", "Menlo", monospace;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid rgba(220, 207, 191, 0.7);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    input[type="search"] {{
      min-width: 280px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      color: inherit;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(154, 52, 18, 0.08);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }}
    @media (max-width: 1080px) {{
      .panel-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 720px) {{
      .page {{ padding: 22px 14px 36px; }}
      .hero {{ padding: 22px 18px 18px; }}
      h1 {{ font-size: 28px; }}
      .metric-grid, .insight-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 560px) {{
      .metric-grid, .insight-grid {{ grid-template-columns: 1fr; }}
      input[type="search"] {{ min-width: 100%; width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Decode Graph Review</div>
      <h1>{html.escape(report['name'])}</h1>
      <p class="lede">This view compresses the raw graph into the questions you usually care about first: what dominates byte traffic, which layers are heavy, what sits on the critical path, and whether the report thinks you are limited by bandwidth, occupancy, launch size, or host synchronization.</p>
      <div class="metric-grid">{overview_html}</div>
    </section>

    <section class="insight-grid">
      {insight_cards(report, nodes, hotspots, nodes_by_id)}
    </section>

    <section class="panel-grid">
      <div class="panel">
        <h2 class="section-title">Top Hotspots</h2>
        <p class="section-copy">These are the heaviest nodes by modeled cost share. In this report, the ranking is driven by byte traffic because the hardware-specific timing floor is not present.</p>
        {hotspot_section(hotspots, nodes_by_id)}
      </div>
      <div class="panel">
        <h2 class="section-title">Bottleneck Mix</h2>
        <p class="section-copy">This is the byte-weighted mix of the current bottleneck labels, not just raw node counts.</p>
        {bottleneck_section(nodes)}
      </div>
      <div class="panel">
        <h2 class="section-title">Layer Pressure</h2>
        <p class="section-copy">The heaviest layers by modeled traffic. This helps spot repeating expensive phases, not just single outlier kernels.</p>
        {layer_section(nodes)}
      </div>
      <div class="panel">
        <h2 class="section-title">Operation Mix</h2>
        <p class="section-copy">Total modeled bytes grouped by op type. This is usually the fastest way to decide whether you should optimize DMMV, attention, routing, or transfer paths first.</p>
        {op_section(nodes)}
      </div>
    </section>

    <section class="table-panel">
      <div class="pill">Critical path excerpt</div>
      <h2 class="section-title">Ordered Critical Path</h2>
      <p class="section-copy">The first 40 steps on the critical path, in execution order. Use this when a hotspot matters only because it blocks everything behind it.</p>
      <table>
        <thead>
          <tr><th>Depth</th><th>Name</th><th>Op</th><th>Layer</th><th>Bytes</th><th>Bottleneck</th></tr>
        </thead>
        <tbody>
          {critical_path_table(report, nodes_by_id)}
        </tbody>
      </table>
    </section>

    <section class="table-panel">
      <div class="controls">
        <div class="pill">Node explorer</div>
        <input id="node-filter" type="search" placeholder="Filter by name, op, layer, or bottleneck">
      </div>
      <h2 class="section-title">All Nodes By Modeled Cost</h2>
      <p class="section-copy">This table is sorted by total modeled bytes. Filter for `q_proj`, `lm_head`, `memory_bandwidth`, or a layer number such as `7`.</p>
      <table>
        <thead>
          <tr><th>#</th><th>Name</th><th>Op</th><th>Layer</th><th>Depth</th><th>Bytes</th><th>Share</th><th>Bottleneck</th><th>Critical</th></tr>
        </thead>
        <tbody id="node-table-body">
          {node_table_rows(nodes, total_bytes)}
        </tbody>
      </table>
    </section>
  </div>
  <script>
    const filter = document.getElementById('node-filter');
    const rows = Array.from(document.querySelectorAll('#node-table-body tr'));
    filter.addEventListener('input', () => {{
      const query = filter.value.trim().toLowerCase();
      for (const row of rows) {{
        const haystack = row.dataset.search || '';
        row.style.display = !query || haystack.includes(query) ? '' : 'none';
      }}
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Path to a decode-graph JSON report")
    parser.add_argument("output_html", type=Path, help="Path to write the HTML dashboard")
    args = parser.parse_args()

    report = json.loads(args.input_json.read_text())
    html_text = render_html(report)
    args.output_html.write_text(html_text)
    print(args.output_html)


if __name__ == "__main__":
    main()
