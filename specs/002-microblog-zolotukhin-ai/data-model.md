# Data Model: Microblog — zolotukhin.ai

## Content Entities

### Post

A daily blog entry about ZINC development.

**Frontmatter fields** (YAML):
- `title`: string (required) — post title, used in `<h1>` and meta tags
- `date`: date (required) — publication date, ISO 8601 format (YYYY-MM-DD)
- `tags`: string[] (optional) — categorization tags (e.g., "vulkan", "turboquant", "rdna4", "performance")
- `excerpt`: string (required) — 1-2 sentence summary, used in post listing and meta description
- `draft`: boolean (optional, default false) — if true, excluded from production build

**Derived fields**:
- `slug`: derived from filename (e.g., `2026-03-25-hello-world.md` → `hello-world`)
- `url`: `/posts/{slug}`
- `readingTime`: computed from word count at build time

**Validation rules**:
- `title` max 100 characters
- `date` must be valid ISO date, not in the future (for production builds)
- `tags` each tag is lowercase, hyphenated, max 30 characters
- `excerpt` max 300 characters (for meta description compatibility)

**File naming convention**: `YYYY-MM-DD-slug-title.md` in `site/src/content/posts/`

### DocPage

A documentation page generated from the repo's `docs/` directory.

**Source fields** (parsed from markdown):
- `title`: extracted from first `# heading` in the source .md file
- `source_path`: path to source file (e.g., `docs/SPEC.md`)
- `content`: full markdown content

**Derived fields**:
- `slug`: derived from filename (e.g., `SPEC.md` → `spec`, `TURBOQUANT_SPEC.md` → `turboquant-spec`)
- `url`: `/docs/{slug}`
- `lastmod`: git last-modified date of the source file

**Source files → doc pages mapping**:

| Source File | URL | Title |
|-------------|-----|-------|
| docs/SPEC.md | /docs/spec | ZINC Technical Specification |
| docs/TURBOQUANT_SPEC.md | /docs/turboquant-spec | TurboQuant KV Cache Compression |
| docs/RDNA4_TUNING.md | /docs/rdna4-tuning | RDNA4 Tuning Guide |
| docs/API.md | /docs/api | API Reference |

### StaticPage

Hand-authored pages with custom content.

**Pages**:
- `/about` — what is ZINC, who is building it, the problem being solved
- `/vision` — why now, the end goal, TurboQuant timing, performance opportunity

**Fields**: Title, content (Astro component, not markdown).

## Relationships

```
Homepage
├── lists Post[] (reverse chronological, paginated)
├── links to /docs (DocPage index)
├── links to /about (StaticPage)
└── links to /vision (StaticPage)

Post
├── has many Tag[]
└── standalone (no relationships to other posts)

DocPage
├── has navigation links to other DocPage[]
└── links back to /docs index

/feed.xml → contains all Post[] (full content)
/sitemap.xml → contains all Post[] + DocPage[] + StaticPage[]
```
