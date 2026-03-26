# Implementation Plan: Microblog — zolotukhin.ai

**Branch**: `002-microblog-zolotukhin-ai` | **Date**: 2026-03-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/002-microblog-zolotukhin-ai/spec.md`

## Summary

Build a clean, modern, static microblog at zolotukhin.ai for daily ZINC development posts. The site is optimized for SEO and AI reachability, includes build-time-generated ZINC documentation pages (technical spec, TurboQuant, RDNA4 tuning, API reference), and lives in the ZINC monorepo under `site/`. No JavaScript required for the core reading experience.

## Technical Context

**Language/Version**: Astro 5.x (static site generator) + TypeScript (build scripts)
**Primary Dependencies**: Astro, @astrojs/mdx, @astrojs/sitemap, @astrojs/rss, shiki (syntax highlighting)
**Storage**: Markdown files (posts + docs) — no database
**Testing**: Lighthouse CI (performance/SEO/accessibility), W3C HTML validator, feed validator
**Target Platform**: Static HTML/CSS deployed to Cloudflare Pages (or GitHub Pages/Vercel)
**Project Type**: Static site (build-time only, zero runtime server)
**Performance Goals**: Lighthouse >95 all categories, build <5s for 100 posts, zero JS on page load
**Constraints**: No client-side JS for core experience, mobile-first responsive, must render in all modern browsers
**Scale/Scope**: ~365 posts/year, 4-6 documentation pages, 2-3 static pages

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Performance-First (I)**: Static site with zero JS — maximum performance by design
- [x] **RDNA4-Native (II)**: N/A — this is a website, not GPU code
- [x] **Zig Systems Correctness (III)**: N/A — site is TypeScript/Astro, lives alongside Zig code in monorepo
- [x] **Vulkan-First (IV)**: N/A
- [x] **Production Serving (V)**: Static hosting = production-ready by default (CDN, HTTPS, global edge)
- [x] **Correctness Validation (VI)**: Lighthouse CI + HTML validation + feed validation ensure correctness

## Project Structure

### Documentation (this feature)

```text
specs/002-microblog-zolotukhin-ai/
├── plan.md              # This file
├── research.md          # Tech decisions
├── data-model.md        # Content model
├── quickstart.md        # Validation scenarios
├── contracts/
│   └── feed-spec.md     # RSS/Atom feed contract
└── tasks.md             # Task breakdown
```

### Source Code (repository root)

```text
site/
├── astro.config.mjs         # Astro configuration (site URL, integrations)
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
├── public/
│   ├── favicon.ico
│   ├── og-image.png         # Default Open Graph image
│   └── robots.txt
├── src/
│   ├── layouts/
│   │   ├── BaseLayout.astro     # HTML shell (head, meta, nav, footer)
│   │   ├── PostLayout.astro     # Blog post layout (article, date, tags)
│   │   └── DocLayout.astro      # Documentation page layout (sidebar nav)
│   ├── pages/
│   │   ├── index.astro          # Homepage — post listing
│   │   ├── about.astro          # About ZINC
│   │   ├── vision.astro         # Why ZINC, why now
│   │   ├── posts/
│   │   │   └── [...slug].astro  # Dynamic post routes
│   │   ├── docs/
│   │   │   ├── index.astro      # Docs landing page
│   │   │   └── [...slug].astro  # Dynamic doc routes
│   │   ├── feed.xml.ts          # RSS feed generator
│   │   └── sitemap-index.xml.ts # Sitemap (via @astrojs/sitemap)
│   ├── content/
│   │   ├── posts/               # Blog post markdown files
│   │   │   ├── 2026-03-25-hello-world.md
│   │   │   └── ...
│   │   └── config.ts            # Content collection schemas
│   ├── components/
│   │   ├── PostCard.astro       # Post preview card (title, date, excerpt)
│   │   ├── TagList.astro        # Tag display component
│   │   ├── DocNav.astro         # Documentation sidebar navigation
│   │   ├── SEOHead.astro        # Meta tags, OG, JSON-LD
│   │   └── Header.astro         # Site header/nav
│   ├── styles/
│   │   └── global.css           # Minimal CSS (typography, layout, colors)
│   └── lib/
│       └── docs-loader.ts       # Build-time loader: reads docs/*.md from repo root
└── .github/
    └── workflows/
        └── deploy-site.yml      # CI: build + deploy to Cloudflare Pages
```

**Structure Decision**: Astro project in `site/` subdirectory of the ZINC monorepo. Content collections for posts. Documentation pages are generated at build time by reading from the repo-root `docs/` directory via a custom loader. This keeps docs as single source of truth.

## Complexity Tracking

> No violations. This is a straightforward static site with standard tooling.
