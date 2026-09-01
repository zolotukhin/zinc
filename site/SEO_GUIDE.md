# SEO & AI Reachability Guide — zolotukhin.ai

## Target Search Terms

Every page should naturally cover terms from its relevant cluster. Don't stuff — write content that answers the query someone would type.

## Local Analytics Workflow

The detailed GA4 and Search Console analysis is generated locally under `site/.seo/`, which is gitignored. Do not commit raw analytics exports or generated private reports.

Use:

```bash
npm run seo:analyze -- --ga4 <ga4-export.csv> --gsc-dir <search-console-export-dir>
```

The public repo should keep only durable SEO policy, tooling, and content strategy. Page-level traffic, query counts, CTRs, bounce rates, and raw export filenames belong in local ignored reports.

### Review Checklist

Use the local report to decide which pages to edit first:

- Pages with high impressions, low CTR, and average position 10 or better need better `seoTitle`, `seoDescription`, and first-screen answers.
- Pages with search clicks but poor engagement need a stronger opening, clearer internal next steps, and tighter intent matching.
- Pages with both trailing-slash and non-trailing-slash variants need canonical redirect work.
- 404 traffic needs Cloudflare or GA4 page-path investigation before content changes.
- Query clusters with growing impressions should become hubs, not isolated posts.

### Durable Blog Topics

Future posts should deepen proven clusters instead of broadening randomly:

1. **Qwen3.6 architecture/local inference**
   - `Qwen3.6 Architecture Details: Hybrid Attention, Sparse MoE, and Local Inference`
   - `Qwen3.6 GGUF and Local Inference: What Needs To Exist Before It Runs Locally`
2. **Speculative decoding and MTP**
   - `Why Speculative Decoding Fails on Qwen3.6-Style MoE/SSM Models`
   - `MTP vs Draft Models for Local Qwen Inference`
3. **MoE inference**
   - `MoE Inference on GPUs: Router Top-K, Shared Experts, and Why It Bottlenecks`
   - `Qwen and Gemma MoE Inference: What Sparse Routing Costs on Consumer GPUs`
4. **Gemma local inference**
   - `Gemma 4 Local Inference: Sliding-Window Attention, Asymmetric GQA, and GPU Memory`
   - `Gemma 4 on AMD RDNA4: Prefill, Vulkan Command Buffers, and Kernel Assumptions`
   - `Gemma 4 vs Qwen3.6 for Local Inference: MoE, SSM, SWA, and Memory Tradeoffs`
5. **RDNA4 / Radeon inference**
   - `AMD RDNA4 LLM Inference Guide: R9700, RX 9070 XT, Vulkan, and llama.cpp`
   - `Radeon AI PRO R9700 vs RX 9070 XT for Local LLM Inference`
6. **Quantization and precision**
   - `FP4 vs FP8 for Local LLM Inference on RDNA4`
   - `KV Cache Quantization for 128K Context on 32GB GPUs`
7. **Apple Silicon local inference**
   - `Apple Silicon Local LLM Inference: Metal, Unified Memory, and M-Series Limits`

### Metadata policy update

Blog posts now support dedicated search metadata:

```yaml
seoTitle: "Qwen3.6 Architecture Details for Local Inference"
seoDescription: "A concise technical guide to Qwen3.6 architecture signals, GGUF status, MoE routing, and what local inference engines need."
```

Use `title` for the article headline. Use `seoTitle` and `seoDescription` for Google results. High-potential pages should receive these fields first.

### Primary clusters

| Cluster | Target queries | Pages |
|---------|---------------|-------|
| **AMD GPU inference** | "LLM inference AMD GPU", "AMD consumer GPU AI", "RDNA4 LLM", "RDNA3 inference", "RX 9070 XT LLM", "Radeon AI PRO R9700 inference" | /zinc, /blog posts |
| **AMD backends** | "ROCm consumer GPU inference", "Vulkan or ROCm LLM inference", "llama.cpp alternative AMD", "Vulkan LLM inference" | /zinc |
| **TurboQuant** | "TurboQuant KV cache compression", "KV cache quantization LLM", "Lloyd-Max quantization GPU", "QJL residual correction" | /zinc/docs/turboquant-spec |
| **Gemma local inference** | "Gemma 4 local inference", "Gemma 4 AMD GPU", "Gemma 4 RDNA4", "Gemma flash attention", "Gemma MoE inference" | /blog posts, /zinc |
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
- [ ] `seoTitle` is 45-60 characters after the brand suffix is included
- [ ] `seoDescription` is 120-155 characters and includes 2-3 target terms
- [ ] Excerpt is reader-facing card copy, not necessarily the meta description
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
