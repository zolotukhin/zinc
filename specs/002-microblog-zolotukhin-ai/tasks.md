# Tasks: Microblog — zolotukhin.ai

**Input**: Design documents from `specs/002-microblog-zolotukhin-ai/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/feed-spec.md, quickstart.md

**Tests**: Lighthouse CI and validation included as checkpoint tasks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize Astro project, base layout, styles, and build pipeline

- [ ] T001 Initialize Astro 5.x project in site/ with package.json, tsconfig.json, astro.config.mjs
- [ ] T002 [P] Configure astro.config.mjs with site URL (https://zolotukhin.ai), MDX integration, sitemap integration, Shiki syntax highlighting for Zig/GLSL/JSON/bash
- [ ] T003 [P] Create global CSS in site/src/styles/global.css — system font stack, typography scale, color scheme (light/dark via prefers-color-scheme), responsive layout, code block styling
- [ ] T004 [P] Add static assets in site/public/ — favicon.ico, og-image.png, robots.txt (allow all, link to sitemap)
- [ ] T005 Create BaseLayout.astro in site/src/layouts/ — HTML shell with <head> (charset, viewport, title slot), nav (home, docs, about, vision), footer, global CSS import
- [ ] T006 [P] Create Header.astro component in site/src/components/ — site name, navigation links, responsive mobile menu (CSS-only, no JS)

**Checkpoint**: `npm run dev` serves a blank site with working navigation, correct typography, and light/dark mode.

---

## Phase 2: Foundational (Content Collections)

**Purpose**: Set up Astro content collections for posts and the docs loader

**CRITICAL**: Blog posts and doc pages depend on this.

- [ ] T007 Define post content collection schema in site/src/content/config.ts — Zod schema for title (string), date (date), tags (string[] optional), excerpt (string), draft (boolean optional)
- [ ] T008 Create build-time docs loader in site/src/lib/docs-loader.ts — reads all .md files from repo-root docs/ directory, extracts title from first heading, computes slug from filename, gets git lastmod date
- [ ] T009 Create sample post in site/src/content/posts/2026-03-25-hello-world.md — valid frontmatter, short content with Zig and GLSL code blocks to test syntax highlighting

**Checkpoint**: Content collections resolve at build time. Sample post and all docs/*.md are available as typed data.

---

## Phase 3: User Story 1 — Daily Dev Blog Posts (Priority: P1) MVP

**Goal**: Visitors can read daily blog posts on a clean, modern microblog.

**Independent Test**: Build the site, verify posts render with dates, titles, tags, listed in reverse chronological order on homepage.

### Layouts and Components

- [ ] T010 [P] [US1] Create PostLayout.astro in site/src/layouts/ — article wrapper with <time>, <h1>, tag list, reading time, content slot, prev/next navigation
- [ ] T011 [P] [US1] Create PostCard.astro in site/src/components/ — post preview card with title, date, excerpt, tag pills, "read more" link
- [ ] T012 [P] [US1] Create TagList.astro in site/src/components/ — renders tag pills with links to tag filter (future) or just display

### Pages

- [ ] T013 [US1] Create homepage in site/src/pages/index.astro — fetch all posts from content collection, sort by date descending, render PostCard for each (depends on T007, T011)
- [ ] T014 [US1] Create dynamic post route in site/src/pages/posts/[...slug].astro — getStaticPaths from content collection, render post with PostLayout (depends on T007, T010)

### Sample Content

- [ ] T015 [P] [US1] Create 3 sample posts in site/src/content/posts/ — varied dates, tags, code blocks (Zig, GLSL, bash) to test the full rendering pipeline
- [ ] T016 [US1] Verify posts build correctly — run `npm run build`, check dist/ for all post pages, verify code block syntax highlighting works

**Checkpoint**: Homepage lists posts in reverse chronological order. Clicking a post shows full content with styled code blocks. MVP is live.

---

## Phase 4: User Story 2 — SEO Optimization and AI Reachability (Priority: P2)

**Goal**: Search engines and AI systems discover and correctly index all content.

**Independent Test**: Lighthouse >95 for Performance/SEO/Accessibility. Valid sitemap, RSS, and structured data.

### SEO Components

- [ ] T017 [P] [US2] Create SEOHead.astro in site/src/components/ — renders <title>, <meta name="description">, canonical URL, og:title, og:description, og:image, og:url, og:type, twitter:card meta tags. Accept props for per-page customization.
- [ ] T018 [P] [US2] Create JSON-LD component — renders <script type="application/ld+json"> with BlogPosting schema on posts, TechArticle on docs, WebSite on homepage. Accept entity props.

### Feed and Sitemap

- [ ] T019 [US2] Create RSS feed generator in site/src/pages/feed.xml.ts — RSS 2.0 format per contracts/feed-spec.md, full post content in <description>, all tags as <category> (depends on T007)
- [ ] T020 [US2] Configure @astrojs/sitemap in astro.config.mjs — ensure all posts, docs, and static pages are included with correct lastmod and changefreq values (depends on T002)

### Integration

- [ ] T021 [US2] Integrate SEOHead.astro into BaseLayout.astro — pass page-specific title, description, and URL to SEO component (depends on T005, T017)
- [ ] T022 [US2] Add JSON-LD to PostLayout.astro and DocLayout.astro — BlogPosting for posts, TechArticle for docs (depends on T010, T018)
- [ ] T023 [US2] Add <link rel="alternate" type="application/rss+xml"> to BaseLayout.astro head (depends on T019)

### Validation

- [ ] T024 [US2] Run Lighthouse CI on built site — verify Performance >95, SEO >95, Accessibility >95 on homepage, a post page, and a doc page
- [ ] T025 [US2] Validate RSS feed — check feed.xml against W3C Feed Validation Service requirements
- [ ] T026 [US2] Validate structured data — check JSON-LD output on post and doc pages against Google Rich Results Test requirements

**Checkpoint**: All pages have proper SEO metadata, structured data, sitemap, and RSS. Lighthouse scores >95.

---

## Phase 5: User Story 3 — ZINC Documentation Pages (Priority: P3)

**Goal**: All ZINC docs (SPEC, TurboQuant, RDNA4 Tuning, API) are available as styled, navigable pages with SEO.

**Independent Test**: Build site, verify /docs lists all 4 doc pages, each renders correctly with navigation and structured data.

### Layout and Components

- [ ] T027 [P] [US3] Create DocLayout.astro in site/src/layouts/ — sidebar navigation between doc pages, content area, breadcrumbs, "Edit on GitHub" link
- [ ] T028 [P] [US3] Create DocNav.astro in site/src/components/ — sidebar navigation listing all documentation pages with active state highlighting

### Pages

- [ ] T029 [US3] Create docs index page in site/src/pages/docs/index.astro — list all doc pages with title, description, and link. Uses docs-loader.ts data. (depends on T008)
- [ ] T030 [US3] Create dynamic doc route in site/src/pages/docs/[...slug].astro — getStaticPaths from docs-loader, render doc content with DocLayout (depends on T008, T027)

### Validation

- [ ] T031 [US3] Verify all 4 doc pages build — check dist/docs/spec/, dist/docs/turboquant-spec/, dist/docs/rdna4-tuning/, dist/docs/api/ exist and render correctly
- [ ] T032 [US3] Verify doc page navigation — sidebar links work, active page is highlighted, all code blocks have syntax highlighting

**Checkpoint**: /docs section is complete with all 4 ZINC documentation pages, working navigation, and correct SEO metadata.

---

## Phase 6: User Story 4 — Project Vision and Narrative Pages (Priority: P4)

**Goal**: Static pages explaining ZINC's mission, the problem, and why now.

**Independent Test**: /about and /vision pages render with content, have SEO metadata, and link from navigation.

- [ ] T033 [P] [US4] Create /about page in site/src/pages/about.astro — what is ZINC, the AMD consumer GPU problem, who is building it, link to GitHub repo
- [ ] T034 [P] [US4] Create /vision page in site/src/pages/vision.astro — why now (RDNA4 maturity, TurboQuant algorithm, Vulkan cooperative matrix), end goal ($500 GPUs for LLM inference), performance numbers, call to action
- [ ] T035 [US4] Verify static pages have correct SEO metadata and are included in sitemap

**Checkpoint**: /about and /vision are live with content, linked from navigation, and indexed.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Deployment, CI, and final polish

- [ ] T036 [P] Create GitHub Actions deploy workflow in site/.github/workflows/deploy-site.yml — build on push to main, deploy to Cloudflare Pages
- [ ] T037 [P] Add Lighthouse CI step to deploy workflow — fail build if any score drops below 90
- [ ] T038 [P] Create 404 page in site/src/pages/404.astro — friendly not-found page with link back to homepage
- [ ] T039 Configure custom domain zolotukhin.ai — DNS records, HTTPS, Cloudflare Pages custom domain setup
- [ ] T040 Write first real blog post in site/src/content/posts/ — "Day 1: Why We're Building ZINC"
- [ ] T041 Run full quickstart.md validation — all 5 scenarios pass

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001-T006)
- **US1 (Phase 3)**: Depends on Foundational (T007-T009) — MVP
- **US2 (Phase 4)**: Depends on US1 (needs layouts to integrate SEO into)
- **US3 (Phase 5)**: Depends on Foundational (T008 docs loader) — can start after Phase 2, parallel with US2
- **US4 (Phase 6)**: Depends on Setup only (T005 BaseLayout) — can start after Phase 1, parallel with everything
- **Polish (Phase 7)**: Depends on US1-US3 completion

### User Story Dependencies

- **US1 (P1)**: Blocked by Foundational. No other story dependencies. **MVP**.
- **US2 (P2)**: Depends on US1 layouts for SEO integration.
- **US3 (P3)**: Only depends on Foundational (T008 docs loader). Can run parallel with US2.
- **US4 (P4)**: Only depends on BaseLayout (T005). Can start earliest of any story.

### Parallel Opportunities

- US3 (docs) and US2 (SEO) can be developed in parallel after Phase 2
- US4 (static pages) can start as soon as BaseLayout exists
- All [P] tasks within a phase can run simultaneously
- Layout and component tasks (T010-T012, T017-T018, T027-T028) are all independent

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (Astro project)
2. Complete Phase 2: Foundational (content collections)
3. Complete Phase 3: User Story 1 (blog posts)
4. **STOP and VALIDATE**: Site builds, posts render, homepage works
5. Deploy — the blog is live with sample posts

### Incremental Delivery

1. Setup + Foundational → Project skeleton ready
2. US1 → Blog posts work → Deploy → MVP live!
3. US2 → SEO + feeds → Lighthouse validation → Discoverable
4. US3 → Docs pages → Full ZINC documentation online
5. US4 → About + Vision → Complete narrative
6. Polish → CI, domain, first real post

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Total tasks: 41
- All tasks produce static output — no runtime dependencies
- Commit after each task or logical group
- Every phase produces a buildable, deployable site
