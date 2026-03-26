# Feature Specification: Microblog — zolotukhin.ai

**Feature Branch**: `002-microblog-zolotukhin-ai`
**Created**: 2026-03-25
**Status**: Draft
**Input**: Daily dev microblog at zolotukhin.ai with SEO optimization, AI reachability, and build-time TurboQuant documentation pages. Lives in the ZINC monorepo.

## User Scenarios & Testing

### User Story 1 - Daily Dev Blog Posts (Priority: P1)

A visitor arrives at zolotukhin.ai and sees a clean, modern microblog with daily posts about ZINC development — what was built, what challenges were overcome, and what's next. Posts are short (1-3 paragraphs), chronologically ordered, and easy to scan. Each post has a date, title, and tags.

**Why this priority**: This is the core value prop — consistent daily content builds SEO authority and tells the ZINC story. Without posts, the site has no reason to exist.

**Independent Test**: Build the site, add 3 sample posts, verify they render correctly with dates, titles, tags, and are listed on the homepage in reverse chronological order.

**Acceptance Scenarios**:

1. **Given** a markdown file in the posts directory, **When** the site is built, **Then** it renders as a styled blog post page with title, date, tags, and content.
2. **Given** multiple posts, **When** a visitor opens the homepage, **Then** they see posts in reverse chronological order with title, date, excerpt, and a "read more" link.
3. **Given** a new markdown file is added, **When** the build runs, **Then** the new post appears on the site without any other changes needed.

---

### User Story 2 - SEO Optimization and AI Reachability (Priority: P2)

Search engines and AI systems (ChatGPT, Perplexity, Claude) discover and surface ZINC content. The site has proper structured data, semantic HTML, fast load times, sitemap, RSS feed, Open Graph tags, and clean URLs. AI crawlers can extract structured information about ZINC, TurboQuant, and RDNA4 inference.

**Why this priority**: The entire purpose of the blog is visibility and adoption. Without SEO and AI reachability, the content exists but nobody finds it.

**Independent Test**: Run Lighthouse audit, verify score >95 for Performance/SEO/Accessibility. Validate structured data with Google Rich Results Test. Confirm sitemap.xml and RSS feed are valid.

**Acceptance Scenarios**:

1. **Given** a published post, **When** crawled by a search engine, **Then** it finds semantic HTML (article, time, heading hierarchy), Open Graph meta tags, JSON-LD structured data, and canonical URL.
2. **Given** the site root, **When** a crawler requests /sitemap.xml, **Then** it receives a valid sitemap listing all posts and documentation pages with lastmod dates.
3. **Given** the site, **When** an AI system queries about "ZINC inference engine" or "TurboQuant KV cache compression", **Then** the structured data and content make it easy to extract factual answers.
4. **Given** any page, **When** loaded on mobile or desktop, **Then** Lighthouse Performance score is >95 (static site, no JS bloat).

---

### User Story 3 - ZINC Documentation Pages (Priority: P3)

The site includes a full documentation section for the ZINC project, generated at build time from the markdown specs in the repo (docs/SPEC.md, docs/TURBOQUANT_SPEC.md, docs/RDNA4_TUNING.md, docs/API.md). These pages cover the architecture, GPU kernels, API reference, RDNA4 tuning guide, and the TurboQuant KV cache compression algorithm. They serve as the canonical public-facing documentation for ZINC.

**Why this priority**: Having ZINC documentation as standalone, SEO-optimized pages makes the project discoverable by developers, researchers, and AI systems. People searching for "Vulkan LLM inference", "RDNA4 GPU tuning", or "TurboQuant KV cache compression" should find these pages.

**Independent Test**: Build the site, verify all documentation pages render from the source markdown in docs/, include proper navigation between sections, and have correct structured data for technical documentation.

**Acceptance Scenarios**:

1. **Given** the docs/ directory in the ZINC repo, **When** the site builds, **Then** it generates documentation pages for each .md file with proper formatting, code blocks, tables, and navigation.
2. **Given** the documentation section, **When** a visitor navigates to /docs, **Then** they see a structured overview with links to: Technical Spec, TurboQuant Spec, RDNA4 Tuning Guide, and API Reference.
3. **Given** a change to any file in docs/, **When** the site rebuilds, **Then** the corresponding documentation pages update automatically.

---

### User Story 4 - Project Vision and Narrative Pages (Priority: P4)

The site has static pages explaining ZINC's mission: why AMD consumer GPUs are underserved, why Vulkan over ROCm, why Zig, why now (TurboQuant timing), and the end goal (making $500 GPUs useful for LLM inference). These pages tell the story and give context to the daily posts.

**Why this priority**: Provides the "why" framing that gives the daily "what" posts meaning. Important for first-time visitors and journalists/bloggers who might write about ZINC.

**Independent Test**: Build the site, verify /about and /vision pages render with correct content, have proper SEO metadata, and link to/from the blog.

**Acceptance Scenarios**:

1. **Given** the /about page, **When** visited, **Then** it explains the ZINC project, the problem (AMD consumer GPUs ignored), and the solution.
2. **Given** the /vision page, **When** visited, **Then** it explains why now (TurboQuant, RDNA4 hardware, Vulkan maturity), the end goal, and includes performance numbers.

---

### Edge Cases

- What happens when a post has no tags? It renders without a tag section; no build error.
- What happens when a post markdown has invalid frontmatter? Build fails with a clear error message pointing to the file and line.
- What happens when docs/TURBOQUANT_SPEC.md has content the site template can't render (e.g., Zig code blocks)? Code blocks render with syntax highlighting; unknown languages fall back to plain text.
- What happens on very old browsers? Site is static HTML+CSS; graceful degradation without JS dependency.

## Requirements

### Functional Requirements

- **FR-001**: Site MUST be a static site generated at build time, producing plain HTML/CSS files deployable to any static host.
- **FR-002**: Blog posts MUST be authored as markdown files with YAML frontmatter (title, date, tags, excerpt).
- **FR-003**: Homepage MUST list posts in reverse chronological order with title, date, excerpt, and link.
- **FR-004**: Site MUST generate /sitemap.xml with all pages and lastmod dates.
- **FR-005**: Site MUST generate /feed.xml (RSS 2.0 or Atom) with full post content.
- **FR-006**: Every page MUST include Open Graph meta tags (og:title, og:description, og:image, og:url, og:type).
- **FR-007**: Every blog post MUST include JSON-LD structured data (BlogPosting schema).
- **FR-008**: Site MUST include a /docs section generated at build time from all markdown files in the repo's docs/ directory (SPEC.md, TURBOQUANT_SPEC.md, RDNA4_TUNING.md, API.md).
- **FR-009**: Site MUST have a clean, modern, minimal design — no JavaScript required for core reading experience.
- **FR-010**: Site MUST be mobile-responsive with Lighthouse scores >95 for Performance, SEO, and Accessibility.
- **FR-011**: Site source MUST live in the ZINC monorepo under a `site/` directory.
- **FR-012**: Site MUST have /about and /vision static pages.
- **FR-013**: Site MUST support syntax highlighting for Zig, GLSL, JSON, and bash code blocks.

### Key Entities

- **Post**: A blog entry with title, date, tags, excerpt, content (markdown), and slug (derived from filename).
- **DocPage**: A documentation page generated from a source markdown file in docs/ (SPEC, TurboQuant, RDNA4 Tuning, API), with navigation links between sections.
- **StaticPage**: A hand-authored page (/about, /vision) with custom content.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Lighthouse Performance score >95 on every page.
- **SC-002**: Lighthouse SEO score >95 on every page.
- **SC-003**: Lighthouse Accessibility score >95 on every page.
- **SC-004**: Site builds in under 5 seconds for 100 posts.
- **SC-005**: All pages are valid HTML5 (W3C validator, zero errors).
- **SC-006**: RSS feed validates against W3C Feed Validation Service.
- **SC-007**: Structured data validates against Google Rich Results Test (zero errors).
- **SC-008**: ZINC documentation pages are indexed by search engines within 2 weeks of deployment.

## Assumptions

- Domain zolotukhin.ai is registered and DNS is under our control.
- Static hosting via GitHub Pages, Cloudflare Pages, or Vercel (all support custom domains + HTTPS).
- No user accounts, comments, or dynamic server-side features — purely static.
- Blog posts are written manually (by humans) as markdown files — not auto-generated.
- The TurboQuant docs are read from the monorepo docs/ directory at build time — single source of truth.
- Design is minimal and typography-focused — no complex layouts or animations.
