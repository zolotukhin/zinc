#!/usr/bin/env node

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

function parseArgs(argv) {
  const args = {
    ga4: null,
    gscDir: null,
    postsDir: "site/src/content/posts",
    output: "site/.seo/deep-seo-analysis.md",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--ga4") {
      args.ga4 = argv[++i];
    } else if (arg === "--gsc-dir") {
      args.gscDir = argv[++i];
    } else if (arg === "--posts-dir") {
      args.postsDir = argv[++i];
    } else if (arg === "--output" || arg === "-o") {
      args.output = argv[++i];
    } else if (arg === "--help" || arg === "-h") {
      args.help = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  return args;
}

function parseCsvLine(line) {
  const cells = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += ch;
    }
  }

  cells.push(cell);
  return cells;
}

function parseCsvRecords(text) {
  const records = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];

    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        cell += ch;
      }
      continue;
    }

    if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      records.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r") {
      cell += ch;
    }
  }

  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    records.push(row);
  }

  return records.filter((record) => record.some((cellValue) => cellValue !== ""));
}

function readCsv(path) {
  const records = parseCsvRecords(readFileSync(path, "utf8").replace(/^\uFEFF/, ""));
  if (records.length === 0) return [];
  const [header, ...rows] = records;
  return rows.map((cells) => Object.fromEntries(
    header.map((name, index) => [name, cells[index] ?? ""]),
  ));
}

function parseNumber(value) {
  const parsed = Number(String(value ?? "").replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

function parsePercent(value) {
  const text = String(value ?? "").trim();
  if (!text) return 0;
  return parseNumber(text.replace("%", "")) / 100;
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function percent1(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function integer(value) {
  return Math.round(value).toLocaleString("en-US");
}

function decimal(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "";
}

function parseDate(raw) {
  if (!/^\d{8}$/.test(raw)) return raw;
  return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
}

function parseGa4Snapshot(path) {
  const lines = readFileSync(path, "utf8").replace(/^\uFEFF/, "").split(/\r?\n/);
  const sections = [];
  let startDate = null;
  let endDate = null;
  let i = 0;

  while (i < lines.length) {
    const trimmed = lines[i].trim();

    if (trimmed.startsWith("# Start date:")) {
      startDate = parseDate(trimmed.replace("# Start date:", "").trim());
      i += 1;
      continue;
    }

    if (trimmed.startsWith("# End date:")) {
      endDate = parseDate(trimmed.replace("# End date:", "").trim());
      i += 1;
      continue;
    }

    if (!trimmed || trimmed.startsWith("#")) {
      i += 1;
      continue;
    }

    const header = parseCsvLine(lines[i]);
    const rows = [];
    i += 1;

    while (i < lines.length) {
      const rowLine = lines[i];
      const rowTrimmed = rowLine.trim();
      if (!rowTrimmed || rowTrimmed.startsWith("#")) break;
      const cells = parseCsvLine(rowLine);
      rows.push(Object.fromEntries(header.map((name, index) => [name, cells[index] ?? ""])));
      i += 1;
    }

    sections.push({ header, rows });
  }

  const section = (...headerPrefix) => sections.find((entry) => (
    headerPrefix.every((name, index) => entry.header[index] === name)
  ))?.rows ?? [];

  return {
    startDate,
    endDate,
    totals: section("Active users", "New users", "Average engagement time per active user", "Event count")[0] ?? {},
    pages: section("Page title and screen class", "Views", "Active users", "Event count", "Bounce rate"),
    firstUserSources: section("First user source / medium", "Active users"),
    sessionSources: section("Session source / medium", "Sessions"),
    cities: section("City", "Active users"),
  };
}

function normalizePath(rawUrl) {
  try {
    const url = new URL(rawUrl);
    let path = url.pathname;
    if (path !== "/") path = path.replace(/\/+$/, "");
    return path || "/";
  } catch {
    return rawUrl;
  }
}

function displayUrl(path) {
  return `https://zolotukhin.ai${path === "/" ? "/" : path}`;
}

function aggregateGscPages(rows) {
  const groups = new Map();

  for (const row of rows) {
    const rawUrl = row["Top pages"];
    const path = normalizePath(rawUrl);
    const group = groups.get(path) ?? {
      path,
      rawUrls: [],
      clicks: 0,
      impressions: 0,
      positionNumerator: 0,
    };
    const clicks = parseNumber(row.Clicks);
    const impressions = parseNumber(row.Impressions);
    group.rawUrls.push(rawUrl);
    group.clicks += clicks;
    group.impressions += impressions;
    group.positionNumerator += parseNumber(row.Position) * impressions;
    groups.set(path, group);
  }

  return [...groups.values()].map((group) => ({
    ...group,
    ctr: group.impressions > 0 ? group.clicks / group.impressions : 0,
    position: group.impressions > 0 ? group.positionNumerator / group.impressions : 0,
    duplicateVariantCount: new Set(group.rawUrls).size,
  }));
}

function parseFrontmatter(filePath) {
  const text = readFileSync(filePath, "utf8");
  const match = text.match(/^---\n([\s\S]*?)\n---/);
  const frontmatter = match ? match[1] : "";
  const data = {};
  let currentList = null;

  for (const rawLine of frontmatter.split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    const keyMatch = line.match(/^([A-Za-z0-9_-]+):(?:\s*(.*))?$/);
    if (keyMatch) {
      const [, key, rawValue = ""] = keyMatch;
      const value = rawValue.trim();
      if (value === "") {
        data[key] = [];
        currentList = key;
      } else {
        data[key] = value.replace(/^"|"$/g, "");
        currentList = null;
      }
      continue;
    }

    const listMatch = line.match(/^\s*-\s+(.*)$/);
    if (listMatch && currentList) {
      data[currentList].push(listMatch[1].trim().replace(/^"|"$/g, ""));
    }
  }

  return data;
}

function readPosts(postsDir) {
  const dir = resolve(postsDir);
  if (!existsSync(dir)) return [];

  return readdirSync(dir)
    .filter((name) => name.endsWith(".md"))
    .map((name) => {
      const slug = name.replace(/\.md$/, "");
      const data = parseFrontmatter(join(dir, name));
      return {
        slug,
        path: `/blog/${slug}`,
        file: join(dir, name),
        title: data.title ?? slug,
        seoTitle: data.seoTitle,
        date: data.date ?? "",
        excerpt: data.excerpt ?? "",
        seoDescription: data.seoDescription,
        tags: Array.isArray(data.tags) ? data.tags : [],
        keywords: Array.isArray(data.keywords) ? data.keywords : [],
        titleLength: String(data.seoTitle ?? data.title ?? slug).length + " — zolotukhin.ai".length,
        descriptionLength: String(data.seoDescription ?? data.excerpt ?? "").length,
      };
    });
}

function stripSiteSuffix(title) {
  return String(title).replace(/\s+—\s+zolotukhin\.ai$/, "");
}

function ga4PageMap(ga4) {
  const map = new Map();
  for (const row of ga4.pages) {
    map.set(stripSiteSuffix(row["Page title and screen class"]), {
      views: parseNumber(row.Views),
      users: parseNumber(row["Active users"]),
      events: parseNumber(row["Event count"]),
      bounce: parseNumber(row["Bounce rate"]),
    });
  }
  return map;
}

function classifyQuery(query) {
  const q = query.toLowerCase();
  if (q.includes("qwen")) return "Qwen 3.6 / Qwen3";
  if (q.includes("speculative") || q.includes("mtp")) return "Speculative decoding / MTP";
  if (q.includes("rdna") || q.includes("r9700") || q.includes("radeon") || q.includes("vllm")) return "AMD RDNA / Radeon inference";
  if (q.includes("fp4") || q.includes("fp8") || q.includes("quant")) return "Quantization / FP4 / FP8";
  if (q.includes("macbook") || q.includes("apple") || q.includes("amx")) return "Apple Silicon";
  if (q.includes("zig") || q.includes("vulkan") || q.includes("vkqueue")) return "Zig / Vulkan";
  if (q.includes("ggml") || q.includes("drm_ioctl") || q.includes("blk.")) return "Error/debug searches";
  if (q.includes("turboquant")) return "TurboQuant";
  return "Other long tail";
}

function aggregateQueries(rows) {
  const clusters = new Map();
  for (const row of rows) {
    const clusterName = classifyQuery(row["Top queries"]);
    const cluster = clusters.get(clusterName) ?? {
      cluster: clusterName,
      clicks: 0,
      impressions: 0,
      positionNumerator: 0,
      samples: [],
    };
    const clicks = parseNumber(row.Clicks);
    const impressions = parseNumber(row.Impressions);
    cluster.clicks += clicks;
    cluster.impressions += impressions;
    cluster.positionNumerator += parseNumber(row.Position) * impressions;
    cluster.samples.push(row["Top queries"]);
    clusters.set(clusterName, cluster);
  }

  return [...clusters.values()]
    .map((cluster) => ({
      ...cluster,
      ctr: cluster.impressions > 0 ? cluster.clicks / cluster.impressions : 0,
      position: cluster.impressions > 0 ? cluster.positionNumerator / cluster.impressions : 0,
    }))
    .sort((a, b) => b.impressions - a.impressions);
}

function markdownTable(headers, rows) {
  if (rows.length === 0) return "_No rows._";
  const escape = (value) => String(value ?? "").replace(/\|/g, "\\|").replace(/\n/g, " ");
  return [
    `| ${headers.map(escape).join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${headers.map((header) => escape(row[header])).join(" | ")} |`),
  ].join("\n");
}

function sortByNumber(rows, key, limit = rows.length) {
  return [...rows].sort((a, b) => b[key] - a[key]).slice(0, limit);
}

function loadGsc(dir) {
  const gscDir = resolve(dir);
  return {
    pages: readCsv(join(gscDir, "Pages.csv")),
    queries: readCsv(join(gscDir, "Queries.csv")),
    chart: readCsv(join(gscDir, "Chart.csv")),
    devices: readCsv(join(gscDir, "Devices.csv")),
    countries: readCsv(join(gscDir, "Countries.csv")),
    filters: readCsv(join(gscDir, "Filters.csv")),
  };
}

function metricTotalsFromChart(chart) {
  const totals = chart.reduce((acc, row) => {
    const impressions = parseNumber(row.Impressions);
    acc.clicks += parseNumber(row.Clicks);
    acc.impressions += impressions;
    acc.positionNumerator += parseNumber(row.Position) * impressions;
    return acc;
  }, { clicks: 0, impressions: 0, positionNumerator: 0 });

  return {
    ...totals,
    ctr: totals.impressions > 0 ? totals.clicks / totals.impressions : 0,
    position: totals.impressions > 0 ? totals.positionNumerator / totals.impressions : 0,
  };
}

function gscDateRange(chart) {
  const dates = chart.map((row) => row.Date).filter(Boolean).sort();
  return {
    start: dates[0] ?? "unknown",
    end: dates[dates.length - 1] ?? "unknown",
  };
}

function buildReport({ ga4Path, gscDir, postsDir }) {
  const ga4 = parseGa4Snapshot(ga4Path);
  const gsc = loadGsc(gscDir);
  const posts = readPosts(postsDir);
  const pageGroups = aggregateGscPages(gsc.pages);
  const pageByPath = new Map(pageGroups.map((page) => [page.path, page]));
  const pageTitleMap = ga4PageMap(ga4);
  const gscTotals = metricTotalsFromChart(gsc.chart);
  const dateRange = gscDateRange(gsc.chart);
  const queryClusters = aggregateQueries(gsc.queries);

  const blogRows = posts.map((post) => {
    const gscPage = pageByPath.get(post.path) ?? {};
    const ga4Page = pageTitleMap.get(post.title) ?? {};
    return {
      ...post,
      gscClicks: gscPage.clicks ?? 0,
      gscImpressions: gscPage.impressions ?? 0,
      gscCtr: gscPage.ctr ?? 0,
      gscPosition: gscPage.position ?? 0,
      gscVariants: gscPage.duplicateVariantCount ?? 0,
      ga4Views: ga4Page.views ?? 0,
      ga4Users: ga4Page.users ?? 0,
      ga4Bounce: ga4Page.bounce ?? 0,
    };
  });

  const topBlogByGa4 = sortByNumber(blogRows, "ga4Views", 12).map((post) => ({
    Article: post.title,
    Date: post.date,
    Views: integer(post.ga4Views),
    Users: integer(post.ga4Users),
    "Bounce": post.ga4Views > 0 ? percent1(post.ga4Bounce) : "",
    "GSC imps": integer(post.gscImpressions),
    "GSC clicks": integer(post.gscClicks),
  }));

  const topBlogByGsc = sortByNumber(blogRows, "gscImpressions", 14).map((post) => ({
    Article: post.title,
    Date: post.date,
    Impressions: integer(post.gscImpressions),
    Clicks: integer(post.gscClicks),
    CTR: percent(post.gscCtr),
    Pos: decimal(post.gscPosition, 2),
    "GA4 views": integer(post.ga4Views),
  }));

  const pageOpportunities = pageGroups
    .filter((page) => page.impressions >= 100 && page.ctr < 0.01 && page.position <= 10)
    .sort((a, b) => b.impressions - a.impressions)
    .slice(0, 14)
    .map((page) => ({
      Page: displayUrl(page.path),
      Impressions: integer(page.impressions),
      Clicks: integer(page.clicks),
      CTR: percent(page.ctr),
      Pos: decimal(page.position, 2),
      Variants: integer(page.duplicateVariantCount),
    }));

  const duplicateGroups = pageGroups
    .filter((page) => page.duplicateVariantCount > 1)
    .sort((a, b) => b.impressions - a.impressions)
    .slice(0, 12)
    .map((page) => ({
      Canonical: displayUrl(page.path),
      Variants: integer(page.duplicateVariantCount),
      Impressions: integer(page.impressions),
      Clicks: integer(page.clicks),
      "Raw variants": page.rawUrls.join(" ; "),
    }));

  const highBounce = ga4.pages
    .map((row) => ({
      title: row["Page title and screen class"],
      views: parseNumber(row.Views),
      users: parseNumber(row["Active users"]),
      bounce: parseNumber(row["Bounce rate"]),
    }))
    .filter((row) => row.views >= 25 && row.bounce >= 0.45)
    .sort((a, b) => (b.bounce - a.bounce) || (b.views - a.views))
    .slice(0, 12)
    .map((row) => ({
      Page: row.title,
      Views: integer(row.views),
      Users: integer(row.users),
      Bounce: percent1(row.bounce),
    }));

  const metadataIssues = blogRows
    .filter((post) => post.titleLength > 60 || post.descriptionLength > 155)
    .sort((a, b) => (b.gscImpressions + b.ga4Views) - (a.gscImpressions + a.ga4Views))
    .slice(0, 16)
    .map((post) => ({
      Article: post.title,
      "Title chars": integer(post.titleLength),
      "Description chars": integer(post.descriptionLength),
      "GSC imps": integer(post.gscImpressions),
      "GA4 views": integer(post.ga4Views),
    }));

  const queryRows = queryClusters.map((cluster) => ({
    Cluster: cluster.cluster,
    Impressions: integer(cluster.impressions),
    Clicks: integer(cluster.clicks),
    CTR: percent(cluster.ctr),
    Pos: decimal(cluster.position, 2),
    "Example queries": cluster.samples.slice(0, 5).join("; "),
  }));

  const devices = gsc.devices.map((row) => ({
    Device: row.Device,
    Clicks: row.Clicks,
    Impressions: row.Impressions,
    CTR: row.CTR,
    Position: row.Position,
  }));

  const countries = gsc.countries.slice(0, 12).map((row) => ({
    Country: row.Country,
    Clicks: row.Clicks,
    Impressions: row.Impressions,
    CTR: row.CTR,
    Position: row.Position,
  }));

  const totalGa4Users = parseNumber(ga4.totals["Active users"]);
  const organicUsers = ga4.firstUserSources
    .filter((row) => row["First user source / medium"] === "google / organic")
    .reduce((sum, row) => sum + parseNumber(row["Active users"]), 0);

  return `# Deep SEO Analysis

Sources:

- GA4 snapshot: \`${ga4Path}\`
- Search Console export: \`${gscDir}\`
- Blog metadata: \`${postsDir}\`

Periods:

- GA4: ${ga4.startDate} to ${ga4.endDate}
- Search Console: ${dateRange.start} to ${dateRange.end}

Generated: ${new Date().toISOString()}

## Executive Read

- Search Console saw ${integer(gscTotals.impressions)} impressions and ${integer(gscTotals.clicks)} clicks, a ${percent(gscTotals.ctr)} CTR at average position ${decimal(gscTotals.position, 2)}.
- GA4 saw ${integer(totalGa4Users)} active users, but only ${integer(organicUsers)} first-user visits from Google organic (${percent1(organicUsers / Math.max(1, totalGa4Users))}). Organic search is present but not yet a main acquisition channel.
- The site ranks surprisingly well for a narrow Qwen3.6 cluster, but snippets are not earning clicks. The most important query cluster has high average positions and near-zero CTR.
- Blog traffic is strongest around Qwen3.6 architecture, MoE, speculative decoding/MTP, RDNA4 performance, and precision/quantization. Future posts should deepen those clusters rather than scatter into unrelated one-off topics.
- A technical canonicalization issue is visible: Search Console reports both trailing-slash and non-trailing-slash versions for many posts/docs. Add HTTP-level 301s or equivalent canonical routing.
- The current blog metadata model uses long article excerpts as meta descriptions. Many high-potential articles exceed the 155-character SEO snippet target.

## Highest-Leverage Actions

1. Create a separate SEO title/description layer for posts. Keep editorial titles if desired, but add \`seoTitle\` and \`seoDescription\` frontmatter so search snippets can be concise and query-aligned.
2. Retarget the Qwen3.6 architecture article first. It has the strongest search demand: rewrite the title/snippet around "Qwen3.6 architecture details", add a compact answer box, and add explicit sections for architecture, GGUF availability, MoE, speculative decoding, and local inference.
3. Turn strong one-off articles into clusters. Add hub links and follow-ups for Qwen3.6, MoE, speculative decoding/MTP, RDNA4 inference, and FP4/FP8 quantization.
4. Add trailing-slash and HTTPS canonical redirects. The canonical tags are not enough if both URL variants are still being discovered and shown separately.
5. Fix the 404 source. GA4 shows a meaningful Not Found page count; use Cloudflare logs or a GA4 page-path export to find the broken URLs.
6. Improve docs snippets. Getting Started and Apple Silicon reference get impressions but weak CTR; they need search-intent titles, short descriptions, and stronger first-screen answers.
7. Pull page + query data via the Search Console API. The CSV export separates Pages and Queries, so it cannot prove which query maps to which URL. The API should become the canonical workflow.

## Search Query Clusters

${markdownTable(["Cluster", "Impressions", "Clicks", "CTR", "Pos", "Example queries"], queryRows)}

## Page CTR Opportunities

Pages with at least 100 impressions, average position 10 or better, and CTR below 1%.

${markdownTable(["Page", "Impressions", "Clicks", "CTR", "Pos", "Variants"], pageOpportunities)}

## Blog Articles With Most GA4 Views

${markdownTable(["Article", "Date", "Views", "Users", "Bounce", "GSC imps", "GSC clicks"], topBlogByGa4)}

## Blog Articles With Most Search Impressions

${markdownTable(["Article", "Date", "Impressions", "Clicks", "CTR", "Pos", "GA4 views"], topBlogByGsc)}

## High-Bounce Pages

${markdownTable(["Page", "Views", "Users", "Bounce"], highBounce)}

## Duplicate URL Variants In Search Console

${markdownTable(["Canonical", "Variants", "Impressions", "Clicks", "Raw variants"], duplicateGroups)}

## Metadata Issues To Fix First

Title length includes the appended \` — zolotukhin.ai\` suffix currently emitted by the site.

${markdownTable(["Article", "Title chars", "Description chars", "GSC imps", "GA4 views"], metadataIssues)}

## Device Split

${markdownTable(["Device", "Clicks", "Impressions", "CTR", "Position"], devices)}

## Country Split

${markdownTable(["Country", "Clicks", "Impressions", "CTR", "Position"], countries)}

## Recommended Future Blog Briefs

1. \`Qwen3.6 Architecture Details: Hybrid Attention, Sparse MoE, and Local Inference\`  
   Why: the Qwen3.6 query cluster dominates impressions and already ranks. Make the post answer the exact "architecture details" query.

2. \`Qwen3.6 GGUF and Local Inference: What Needs To Exist Before It Runs Locally\`  
   Why: GSC shows repeated GGUF/local-intent queries. This can capture people trying to run the model, not just read architecture commentary.

3. \`Why Speculative Decoding Fails on Qwen3.6-Style MoE/SSM Models\`  
   Why: speculative decoding pages get search impressions and clicks, and the topic connects naturally to MTP and Qwen3.6.

4. \`MoE Inference on GPUs: Router Top-K, Shared Experts, and Why It Bottlenecks\`  
   Why: the MoE article already gets clicks, but GA4 bounce is high. A more practical explainer can become a hub.

5. \`AMD RDNA4 LLM Inference Guide: R9700, RX 9070 XT, Vulkan, and llama.cpp\`  
   Why: RDNA4/Radeon terms appear in GSC and the site already has proprietary expertise and benchmark credibility.

6. \`FP4 vs FP8 for Local LLM Inference on RDNA4\`  
   Why: the FP4/FP8 article is one of the better-performing recent posts. Turn it into a broader precision guide.

7. \`Apple Silicon Local LLM Inference: Metal, Unified Memory, and M-Series Limits\`  
   Why: Apple Silicon pages get impressions, but current matching includes broad/spec-like queries. Refocus the page around local inference instead of generic Apple hardware.

## Page-Specific Rewrite Briefs

### Qwen 3.6 Architecture Article

- Current problem: very high impressions, low CTR, and the winning queries use "Qwen3.6 architecture details" phrasing.
- Rewrite title target: \`Qwen3.6 Architecture Details for Local Inference\`.
- Add a first-screen summary table: context length, architecture signal, MoE signal, GGUF/local status, ZINC impact.
- Add explicit H2s for "Qwen3.6 architecture details", "Qwen3.6 GGUF availability", and "Qwen3.6 speculative decoding".
- Link outward to MoE, speculative decoding, and Getting Started pages.

### Getting Started Docs

- Current problem: good average position, poor CTR.
- Rewrite title target: \`Run Local LLMs on AMD GPUs with Vulkan or ROCm\`.
- Keep the ZINC brand in the H1/subtitle, but put the search intent first.
- Add a "fast path" code block above longer explanation.
- Link to benchmarks and hardware requirements from the first screen.

### RDNA4 Debug/Performance Posts

- Current problem: several posts rank around positions 4-8 but have zero or low clicks because titles are witty or narrow.
- Add search-intent subtitles and short meta descriptions using: \`AMD RDNA4 LLM inference\`, \`Radeon AI PRO R9700\`, \`Vulkan LLM inference\`, \`llama.cpp comparison\`.
- Create one hub page or guide that links to the individual tuning posts.

### MoE Article

- Current problem: clicks exist, but bounce is high.
- Add a practical "what changes in inference engines" section near the top.
- Add a diagram summary and links to Qwen3.6, Gemma, and router/kernel implementation articles.
- Consider a companion article focused on "MoE inference bottlenecks" rather than model architecture alone.

### Apple Silicon Reference

- Current problem: many impressions appear to come from broad Apple hardware/spec queries.
- Decide whether this is a target. If yes, add a clearer page title for local LLM inference on Apple Silicon. If no, de-emphasize generic MacBook terms and point the page at Metal/ZINC queries.

## Tracking Gaps

- The GA4 export lacks page paths, so page title matching is approximate.
- The GSC CSV export lacks page-query pairs, so high-level query opportunities must be inferred.
- The next tool should call the Search Console API for \`page + query + date + device\`, then join that to GA4 landing-page metrics.
`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.ga4 || !args.gscDir) {
    console.log("Usage: node tools/analyze_seo_exports.mjs --ga4 <ga4-export.csv> --gsc-dir <search-console-export-dir> [--output site/.seo/deep-seo-analysis.md]");
    process.exit(args.help ? 0 : 1);
  }

  const outputPath = resolve(args.output);
  const report = buildReport({
    ga4Path: resolve(args.ga4),
    gscDir: resolve(args.gscDir),
    postsDir: resolve(args.postsDir),
  });

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, report);
  console.log(`Wrote ${outputPath}`);
}

main();
