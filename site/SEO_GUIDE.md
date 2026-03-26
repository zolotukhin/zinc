# SEO & AI Reachability Guide — zolotukhin.ai

## Target Search Terms

Every page should naturally cover terms from its relevant cluster. Don't stuff — write content that answers the query someone would type.

### Primary clusters

| Cluster | Target queries | Pages |
|---------|---------------|-------|
| **AMD GPU inference** | "LLM inference AMD GPU", "AMD consumer GPU AI", "RDNA4 LLM", "RDNA3 inference", "RX 9070 XT LLM", "Radeon AI PRO R9700 inference" | /zinc, /blog posts |
| **ROCm alternatives** | "ROCm alternative consumer GPU", "vLLM without ROCm", "llama.cpp alternative AMD", "Vulkan LLM inference" | /zinc |
| **TurboQuant** | "TurboQuant KV cache compression", "KV cache quantization LLM", "Lloyd-Max quantization GPU", "QJL residual correction" | /zinc/docs/turboquant-spec |
| **RDNA4 tuning** | "RDNA4 tuning LLM", "RADV cooperative matrix", "AMD GPU ECC disable", "SPIR-V RADV performance" | /zinc/docs/rdna4-tuning |
| **Local AI serving** | "local LLM server", "OpenAI compatible local inference", "self-hosted LLM API", "continuous batching consumer GPU" | /zinc, /blog posts |
| **Zig + Vulkan** | "Zig inference engine", "Vulkan compute shaders LLM", "GLSL compute shader inference" | /zinc/docs/spec |

### Competitor terms (mention naturally in comparisons)

- vLLM, llama.cpp, ROCm, HIP, CUDA, TensorRT-LLM, Ollama, LM Studio
- NVIDIA, MI300X, H100, A100
- GGUF, GGML, safetensors

## Rules

### Do

1. **Title tags**: Include primary keyword + brand. Max 60 chars. Format: `{Topic} — zolotukhin.ai`
2. **Meta descriptions**: Include 2-3 target keywords naturally. Max 155 chars. Write as a compelling snippet someone would click.
3. **H1**: One per page, includes primary keyword. Match closely to what someone would search.
4. **H2/H3**: Use target keywords in subheadings where natural. These become anchor links and FAQ candidates.
5. **First paragraph**: Front-load the primary keyword in the first 100 words.
6. **Comparison content**: Mention competitors by name in genuine comparisons (e.g., "Unlike vLLM which requires ROCm..."). This captures "X vs Y" searches.
7. **Structured data**: JSON-LD on every page. Use `FAQPage` schema on pages with natural Q&A content. Use `SoftwareApplication` on /zinc.
8. **Alt text**: Every image gets descriptive alt text with a keyword where natural.
9. **Internal links**: Link between pages with keyword-rich anchor text (not "click here").
10. **Clean URLs**: Hyphens, lowercase, descriptive slugs. No underscores, no IDs.
11. **Canonical URLs**: Every page has `<link rel="canonical">`.
12. **RSS**: Full content in feed (not excerpts). Helps AI systems index content.

### Don't

1. **No hidden text**: No `display: none`, no text matching background color, no font-size: 0. Google penalizes this.
2. **No keyword stuffing**: If a sentence sounds unnatural with a keyword, rewrite or remove it.
3. **No duplicate content**: Each page has unique title and description.
4. **No thin pages**: Every page must have substantial content. If it's just a link list, add context.
5. **No orphan pages**: Every page must be reachable from at least one other page via a link.

## Structured Data Patterns

### SoftwareApplication (on /zinc)
```json
{
  "@type": "SoftwareApplication",
  "name": "ZINC",
  "description": "...",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Linux",
  "url": "https://zolotukhin.ai/zinc",
  "codeRepository": "https://github.com/zolotukhin/zinc"
}
```

### FAQPage (on pages with comparison/explanation content)
```json
{
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "Does ZINC work on AMD consumer GPUs?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes. ZINC targets RDNA3 and RDNA4..."
      }
    }
  ]
}
```

### TechArticle (on /zinc/docs/* pages)
Already implemented. Ensure `headline` and `description` are keyword-rich.

### BlogPosting (on /blog/* posts)
Already implemented. Ensure `keywords` array covers target terms.

## Blog Post SEO Checklist

For every new post:
- [ ] Title includes a primary keyword
- [ ] Excerpt (meta description) includes 2-3 target terms
- [ ] Tags map to target clusters
- [ ] First paragraph mentions the primary topic
- [ ] At least one internal link to /zinc or a doc page
- [ ] At least one mention of a competitor/alternative for comparison queries
- [ ] Code blocks have language specified (for Shiki highlighting)
- [ ] No orphan — linked from homepage or another post

## Measuring Success

- Google Search Console: impressions, clicks, position for target queries
- Check `site:zolotukhin.ai` in Google after 2 weeks
- Test key queries in ChatGPT/Perplexity/Claude to see if ZINC content surfaces
- Lighthouse SEO score >95 on every page
