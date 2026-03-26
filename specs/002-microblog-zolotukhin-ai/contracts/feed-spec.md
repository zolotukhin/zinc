# Feed Contract: RSS and Sitemap

## RSS Feed — /feed.xml

**Format**: RSS 2.0

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>ZINC — Building an LLM Inference Engine</title>
    <description>Daily development log for ZINC, a Zig + Vulkan inference engine for AMD GPUs</description>
    <link>https://zolotukhin.ai</link>
    <atom:link href="https://zolotukhin.ai/feed.xml" rel="self" type="application/rss+xml"/>
    <language>en-us</language>
    <lastBuildDate>RFC 822 date</lastBuildDate>
    <item>
      <title>Post title</title>
      <link>https://zolotukhin.ai/posts/slug</link>
      <guid isPermaLink="true">https://zolotukhin.ai/posts/slug</guid>
      <pubDate>RFC 822 date</pubDate>
      <description>Full HTML content of the post</description>
      <category>tag1</category>
      <category>tag2</category>
    </item>
  </channel>
</rss>
```

**Requirements**:
- Full post content in `<description>` (not just excerpt)
- Valid RFC 822 dates
- `<guid>` uses permalink URL
- Includes all published posts (not drafts)
- Must validate against W3C Feed Validation Service

## Sitemap — /sitemap.xml

**Format**: Sitemap Protocol 0.9

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://zolotukhin.ai/</loc>
    <lastmod>2026-03-25</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://zolotukhin.ai/posts/slug</loc>
    <lastmod>2026-03-25</lastmod>
    <changefreq>never</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://zolotukhin.ai/docs/spec</loc>
    <lastmod>git lastmod date</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.9</priority>
  </url>
</urlset>
```

**Requirements**:
- Includes all pages: homepage, posts, docs, static pages
- `<lastmod>` uses ISO 8601 date format
- Posts use `changefreq: never` (content doesn't change)
- Docs use `changefreq: weekly` (may update with repo)
- Homepage uses `changefreq: daily` (new posts daily)

## JSON-LD Structured Data

### BlogPosting (on every post page)

```json
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "Post title",
  "datePublished": "2026-03-25",
  "author": {
    "@type": "Person",
    "name": "Stepan Zolotukhin",
    "url": "https://zolotukhin.ai/about"
  },
  "publisher": {
    "@type": "Organization",
    "name": "ZINC Project",
    "url": "https://zolotukhin.ai"
  },
  "description": "Post excerpt",
  "url": "https://zolotukhin.ai/posts/slug",
  "keywords": ["tag1", "tag2"]
}
```

### TechArticle (on documentation pages)

```json
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Page title",
  "dateModified": "git lastmod date",
  "author": {
    "@type": "Organization",
    "name": "ZINC Project"
  },
  "description": "Page description",
  "url": "https://zolotukhin.ai/docs/slug"
}
```
