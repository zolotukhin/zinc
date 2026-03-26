# zolotukhin.ai blog writing guide

This guide defines how AI should write blog posts for `zolotukhin.ai`.

The target outcome is simple: every post should feel sharp, current, technical, and human. It should be easy to read from top to bottom, but still useful to engineers who care about systems, AI, hardware, inference, compilers, and performance.

## What a strong post feels like

A strong post opens with a real hook, explains why the topic matters now, gets technical without turning into a wall of jargon, and lands with a clear takeaway. It should feel like an engineer explaining something important, not like a content machine filling a template.

The writing should sound informed, calm, and specific. The best reference points in `writing/sample_*` are the narrative flow, concrete examples, and technical honesty. The parts to improve are pacing, visual structure, and SEO discipline.

## Hard rules

- Use simple, direct language.
- Keep the writing technical, but clean.
- Make the post flow in prose from top to bottom.
- Start with an engaging opening in the first 2 to 3 paragraphs.
- Keep paragraphs short. Most should be 2 to 4 sentences.
- Use bullet points rarely. Only use them when they genuinely improve clarity.
- Do not rely on long separator-style titles such as `AI-HW-SEO-STACK` or `Why-Low-Latency-LLM-Inference-Matters`.
- Do not overuse em dashes. Prefer commas, periods, or a new sentence.
- Do not use all-caps words for emphasis.
- Do not sound like generic AI writing. Avoid phrases such as `in today's rapidly evolving landscape`, `game changer`, `revolutionary`, `unlocking the power of`, or `it is worth noting`.
- Every important claim should have support: a metric, benchmark, code detail, architecture fact, tradeoff, or direct observation.

## Voice and tone

- Write like a strong technical founder or systems engineer.
- Be opinionated when there is a real reason.
- Be specific about tradeoffs and limitations.
- Prefer plain English over buzzwords.
- Use first person when the post is about work we actually did.
- Define terms the first time they appear if they are not obvious.
- If a concept is hard, explain it in clean language first, then go deeper.

## Default post structure

Most posts should follow this shape:

1. A title that is specific, current, and easy to search.
2. A short excerpt that clearly states the topic and why it matters.
3. A hook in the opening that creates curiosity fast.
4. A section that frames the problem or opportunity.
5. A section that explains the core technical idea.
6. A section with proof: code, numbers, benchmarks, architecture, or failure modes.
7. A visual break: table, diagram, or schema.
8. A closing section that says what changed, what matters, or what comes next.

The post should feel like a guided sequence, not a bag of fragments.

## Titles and headings

Titles must be catchy, but still grounded in real AI and hardware trends. They should contain the concrete thing the reader is searching for.

Good title patterns:

- `Why RDNA4 matters for local LLM inference`
- `How TurboQuant cuts KV cache bandwidth`
- `What changed in AMD consumer GPU inference this month`
- `Why Vulkan is the right bet for local AI on AMD`
- `The bottleneck that still limits local 35B inference`

Avoid titles like:

- `THE FUTURE OF AI HARDWARE`
- `Everything you need to know about local AI`
- `AI-HW-SYSTEMS-SEO BREAKDOWN`
- `A deep dive into the world of modern inference optimization`

Heading rules:

- Use sentence case, not title case.
- Make headings descriptive and searchable.
- Headings should move the story forward.
- Avoid vague headers like `Overview`, `Thoughts`, or `More details`.

## SEO and AI reachability

Posts should be easy for both humans and machines to parse.

- Put the main topic or keyword in the title.
- Use the same topic naturally in the excerpt and early in the introduction.
- Use one or two close variants throughout the post.
- Prefer exact nouns over vague references. For example, repeat `RDNA4`, `KV cache compression`, `local LLM inference`, `AMD consumer GPUs`, or `Vulkan inference` where useful.
- Answer likely reader questions directly in normal prose.
- Explain acronyms on first use.
- Keep heading hierarchy clean.
- Use tables when comparing options, speeds, tradeoffs, or hardware.
- Use diagrams when describing architecture, flow, memory movement, or before/after changes.
- Every image, chart, diagram, or schema should have a caption and a one-sentence interpretation in the text.

SEO should not feel bolted on. If the writing starts sounding like keyword stuffing, rewrite it.

## Visual structure rules

Every technical post should include at least one visual aid unless the topic is very short.

Preferred visual types:

- A simple architecture or flow diagram
- A benchmark table
- A before/after comparison table
- A memory layout or pipeline schema
- A short code block with commentary

Preferred generation methods:

- Mermaid for flow and architecture diagrams
- SVG generated from code for polished diagrams
- Markdown tables generated from benchmark data
- Small code snippets pulled from the real implementation

Visual rules:

- Use 1 to 3 visuals per post.
- Each visual must clarify a point, not decorate the page.
- Place the first visual before the article turns into a long wall of text.
- Follow every visual with a short paragraph that explains what the reader should notice.

## Formatting rules

- Short intro paragraphs.
- Mostly prose.
- Bullet lists only when the reader truly benefits from scanning.
- Keep quoted text short.
- Use code blocks only when the code helps the explanation.
- If a section gets too dense, add a table, diagram, or example.
- Do not stack multiple giant sections without a visual or concrete example between them.

## Content rules for technical credibility

- Be precise about what was built, measured, changed, or observed.
- If a claim is time-sensitive, anchor it to a real date, release, benchmark run, or hardware generation.
- If performance is discussed, name the hardware, model, workload, and constraint.
- If a decision was made, state the alternative and why it lost.
- If something is uncertain, say so clearly.
- Do not oversell.

## Anti-patterns to avoid

- Generic grand openings with no real hook
- Long uninterrupted blocks of abstract explanation
- Bullet-heavy writing that reads like notes
- Repetitive sentence rhythm
- Empty claims about the future of AI
- Corporate marketing language
- Forced hype
- Strange capitalization or separator-heavy wording

## Good opening formula

Use one of these patterns in the first paragraph:

- Start with a surprising result
- Start with a concrete bottleneck
- Start with a decision and why it was necessary
- Start with a benchmark that changes the framing
- Start with a bug, failure, or constraint that forced a better design

Then quickly answer: why should the reader care right now?

## Recommended length

- Standard post: 900 to 1,600 words
- Deep technical post: 1,400 to 2,200 words

If the post goes long, it must earn that length with diagrams, examples, benchmarks, or code.

## Final checklist

Before publishing, verify:

- The title is specific, current, and searchable.
- The first 100 words contain the main topic naturally.
- The opening creates curiosity immediately.
- The post reads smoothly in prose.
- The post does not depend on bullet points for structure.
- The language sounds human and technically grounded.
- There is at least one visual aid.
- The visuals are explained, not just inserted.
- Claims are supported by real details.
- Headings are clear and descriptive.
- There are no awkward all-caps phrases or separator-heavy titles.
- Em dashes are rare or absent.

## Short instruction for AI

Write like a technical operator, not a marketer. Lead with a hook, stay concrete, keep the prose moving, use current AI and hardware language naturally, and break up dense sections with a diagram or table before the reader gets lost.
