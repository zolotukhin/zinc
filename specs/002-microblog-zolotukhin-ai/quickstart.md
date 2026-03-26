# Quickstart Validation: Microblog — zolotukhin.ai

## Prerequisites

- Node.js 20+ and npm
- Git (for doc page lastmod dates)

## Validation Scenario 1: Build and Serve Locally (US1)

```bash
cd site
npm install
npm run dev
# Open http://localhost:4321
```

**Pass criteria**:
- Homepage loads with post listing in reverse chronological order
- Clicking a post opens the full post page with title, date, tags, and content
- Navigation links work (home, docs, about, vision)
- Code blocks have syntax highlighting (Zig, GLSL, bash)

## Validation Scenario 2: Production Build and Lighthouse (US2)

```bash
cd site
npm run build
npx serve dist/

# In another terminal:
npx lighthouse http://localhost:3000 --output json --chrome-flags="--headless"
```

**Pass criteria**:
- Build completes in <5 seconds
- `dist/` contains only HTML, CSS, and image files (zero .js files for page content)
- Lighthouse Performance >95
- Lighthouse SEO >95
- Lighthouse Accessibility >95
- `/sitemap.xml` is valid XML with all pages listed
- `/feed.xml` is valid RSS 2.0

## Validation Scenario 3: Documentation Pages (US3)

```bash
# After build, check docs pages exist:
ls dist/docs/spec/index.html
ls dist/docs/turboquant-spec/index.html
ls dist/docs/rdna4-tuning/index.html
ls dist/docs/api/index.html
```

**Pass criteria**:
- All 4 doc pages exist and render correctly
- Doc pages include sidebar navigation between sections
- Code blocks in docs have syntax highlighting
- Changing a file in `docs/` and rebuilding updates the corresponding page
- Doc pages include TechArticle JSON-LD structured data

## Validation Scenario 4: SEO and Structured Data (US2)

```bash
# Check structured data
curl -s http://localhost:3000/posts/hello-world/ | grep -o 'application/ld+json'
curl -s http://localhost:3000/feed.xml | head -5
curl -s http://localhost:3000/sitemap.xml | head -5
```

**Pass criteria**:
- Every post page contains `<script type="application/ld+json">` with BlogPosting schema
- Every doc page contains TechArticle schema
- Every page has `<meta property="og:title">`, `og:description`, `og:url`
- `<link rel="canonical">` present on every page
- RSS feed contains full post content (not just excerpts)
- Sitemap lists all pages with correct lastmod dates

## Validation Scenario 5: Mobile Responsiveness (US1, US2)

```bash
# Lighthouse mobile audit
npx lighthouse http://localhost:3000 --output json --chrome-flags="--headless" --preset=desktop
npx lighthouse http://localhost:3000 --output json --chrome-flags="--headless" --form-factor=mobile
```

**Pass criteria**:
- Both desktop and mobile Lighthouse scores >95
- Content is readable without horizontal scrolling on 320px viewport
- Navigation is usable on touch devices
- Code blocks have horizontal scroll (not overflow)
