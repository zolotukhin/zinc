# Research: Microblog — zolotukhin.ai

## Decision Log

### D1: Astro over Hugo/11ty/Next.js

**Decision**: Use Astro 5.x as the static site generator.
**Rationale**: Zero JS by default (ships pure HTML/CSS), native markdown/MDX support, content collections with type-safe schemas, built-in sitemap and RSS integrations, excellent Lighthouse scores out of the box. Astro's "islands" architecture means we can add interactivity later if needed without changing the foundation.
**Alternatives considered**: Hugo (fast builds but Go templating is awkward for component composition), 11ty (flexible but less opinionated — more setup), Next.js (overkill — ships React runtime, hurts performance for a blog).

### D2: Cloudflare Pages for Hosting

**Decision**: Deploy to Cloudflare Pages with custom domain zolotukhin.ai.
**Rationale**: Free tier supports custom domains + HTTPS, global CDN edge network, automatic builds from git push, no cold starts (static files), generous bandwidth.
**Alternatives considered**: GitHub Pages (limited build customization, no server-side redirects), Vercel (good but Cloudflare's edge network is larger), self-hosted (unnecessary complexity for static files).

### D3: Content Collections for Posts

**Decision**: Use Astro's built-in content collections with Zod schemas for frontmatter validation.
**Rationale**: Type-safe frontmatter validation catches errors at build time (missing title, invalid date format). Automatic slug generation from filename. Built-in pagination support.
**Alternatives considered**: Raw glob imports (no validation), CMS (adds runtime dependency), custom file reader (reinventing the wheel).

### D4: Build-Time Docs Loader

**Decision**: Custom Astro loader that reads `docs/*.md` from the repo root at build time and generates documentation pages.
**Rationale**: Single source of truth — docs are maintained in the ZINC repo's `docs/` directory and automatically appear on the site. No manual copying or syncing.
**Alternatives considered**: Symlinks (fragile across platforms), git submodules (unnecessary — same repo), manual copy step (drift risk).

### D5: No JavaScript for Core Experience

**Decision**: Ship zero client-side JavaScript for the reading experience.
**Rationale**: Maximizes Lighthouse performance score, improves accessibility, reduces attack surface, faster page loads globally. A blog doesn't need interactivity. Syntax highlighting is done at build time via Shiki.
**Alternatives considered**: Minimal JS for search (deferred to later — can add Pagefind without changing architecture), interactive code playgrounds (not needed for this content).

### D6: SEO and AI Reachability Strategy

**Decision**: Semantic HTML5 + JSON-LD structured data + Open Graph + sitemap + RSS + clean URLs.
**Rationale**: Search engines prefer semantic HTML with structured data. AI systems (ChatGPT web browsing, Perplexity, etc.) extract information better from well-structured pages. JSON-LD BlogPosting schema helps search engines understand post metadata.
**Specific tactics**:
- `<article>`, `<time datetime>`, `<h1>`-`<h3>` heading hierarchy
- JSON-LD `BlogPosting` on every post, `TechArticle` on doc pages
- `og:title`, `og:description`, `og:image`, `og:url` on every page
- `/sitemap.xml` with lastmod dates
- `/feed.xml` RSS 2.0 with full content (not just excerpts)
- Clean URLs: `/posts/hello-world` not `/posts/hello-world.html`
- `<link rel="canonical">` on every page

### D7: Syntax Highlighting

**Decision**: Shiki (build-time syntax highlighting) with support for Zig, GLSL, JSON, bash, and TypeScript.
**Rationale**: Shiki produces static HTML with inline styles — no client-side JS needed. Supports all languages we need. Astro has native Shiki integration.
**Alternatives considered**: Prism.js (requires client-side JS), highlight.js (same issue), server-side highlighters with extra runtime dependencies.

### D8: Design Approach

**Decision**: Minimal, typography-first design with system font stack, high contrast, and generous whitespace.
**Rationale**: Fast to implement, fast to load, ages well. Focus is on content readability, not visual complexity. Dark/light mode via CSS `prefers-color-scheme`.
**Alternatives considered**: Tailwind CSS (adds build complexity for a simple blog), CSS framework (unnecessary weight), custom design system (over-engineering).
