# zolotukhin.ai

The Astro website for ZINC, its documentation, benchmarks, and engineering blog.
Production runs on Cloudflare Pages. Deployments are manual Wrangler uploads;
pushing to `main` runs CI without publishing the site.

## Develop

Use **Bun 1.4.4 or newer**, pinned by the repository’s `.bun-version`, and
Node **24.20.0**, pinned in `.node-version`. The supported minimum for the
current Astro release is Node 22.23.4. Run these commands from `site/`:

```bash
bun install --frozen-lockfile
bun run dev
```

| Command                    | Purpose                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `bun run dev`              | Start the development server                                                          |
| `bun run build`            | Build all pages into `dist/`                                                          |
| `bun run preview`          | Preview the production build locally                                                  |
| `bun run check:posts`      | Verify article images and wide tables on phones/desktops, with and without JavaScript |
| `bun run check:benchmarks` | Verify the built dashboard, numeric data, links, controls, and mobile layouts         |

`bun run build` adds Cloudflare's exclusion markers around literal code after
Astro compresses the HTML. This final pass uses Bun's HTML parser and verifies
that all other HTML bytes remain unchanged. Publish the full command's output.

After `bun run build`, run `bun run check:benchmarks`. Install the test browser
with `bunx playwright install chromium` if needed. The check serves `dist/` on a
temporary loopback port and stops it afterward. Use `BENCHMARK_BASE_URL` to
check a running dev server instead, or `BENCHMARK_SCREENSHOTS=/tmp/benchmarks`
to save desktop/mobile screenshots in both themes. The check includes axe-core
accessibility scans with evidence expanded. CI runs the same browser checks,
plus the benchmark calculation tests. These are site development tools; they do
not ship in the dashboard's client bundle.

`bun run check:posts` checks the edited hardware/photo articles after a production
build. It rejects missing images, empty galleries, and page-wide mobile overflow;
wide tables must keep their content reachable. Use `POST_SCREENSHOTS=/tmp/posts`
to save the verified layouts.

The site uses Astro 7 and the unified Markdown processor so existing
remark/rehype plugins keep working. KaTeX styles and fonts are bundled from
the installed package, matching the renderer. Mermaid renders diagrams in the
browser.

## Content

| Location                         | Content                                        |
| -------------------------------- | ---------------------------------------------- |
| `src/pages/`                     | Landing pages, benchmark dashboard, and routes |
| `src/content/posts/`             | Blog posts                                     |
| `../docs/`                       | Technical documentation                        |
| `../src/**/*.zig`                | Source for the generated Zig API reference     |
| `src/data/zinc-performance.json` | Published benchmark measurements               |

Use the [blog guide](../writing/ZOLOTUKHIN_AI_BLOG_GUIDE.md) and
[post template](../writing/ZOLOTUKHIN_AI_POST_TEMPLATE.md) for new posts. Published
content lives in `src/content/posts/`; old implementation plans and duplicate
drafts are available in Git history.

The build needs the full repository, the pinned Zig toolchain, and the headers
used by the native API layout probe. CI installs `libvulkan-dev` on Linux.
Site build checks run when `site/`, `docs/`, or `src/` changes.

## Publish

After the build and browser checks pass, upload the production bundle:

```bash
bun run build
bunx wrangler pages deploy dist --project-name zolotukhin-ai --branch main
```

Verify `/`, `/zinc/`, `/zinc/docs/`, `/zinc/benchmarks/`, `/blog/`, `/feed.xml`,
and `/sitemap-index.xml` on `https://zolotukhin.ai` after deployment. For a
Cloudflare-hosted build, use `site` as the root, `bun run build` as the build
command, `dist` as the output directory, and the Node version in `.node-version`.
