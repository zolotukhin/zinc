# zolotukhin.ai

Static Astro site for `zolotukhin.ai`. The production deploy target is Cloudflare Pages.

## Local development

From the repo root:

```bash
cd site
npm ci
npm run dev
```

Useful commands:

| Command | What it does |
| --- | --- |
| `npm run dev` | Start the local Astro dev server |
| `npm run build` | Build the production site into `dist/` |
| `npm run preview` | Preview the built site locally |

## Important repo layout

- The Astro app lives in `site/`
- Technical documentation source files live in repo-root `docs/`
- Blog posts live in `site/src/content/posts/`

Because docs are loaded from `../docs`, the Cloudflare Pages project must watch both `site/*` and `docs/*`.

## First Cloudflare Pages deploy

This repo is set up for Cloudflare Pages Git integration. That is the recommended path for the first deploy because it gives you automatic production deploys on `main` and preview deploys for pull requests.

### 1. Push the repo to GitHub

The Pages project should point at:

- Repository: `zolotukhin/zinc`
- Production branch: `main`

### 2. Create the Pages project

In Cloudflare:

1. Open `Workers & Pages`
2. Select `Create application`
3. Select `Pages`
4. Select `Connect to Git`
5. Authorize GitHub if Cloudflare asks for it
6. Choose the `zolotukhin/zinc` repository

### 3. Use these build settings

Use these exact values:

- Project name: `zolotukhin-ai` or `zinc-site`
- Framework preset: `Astro`
- Root directory: `site`
- Build command: `npm run build`
- Build output directory: `dist`

Cloudflare Pages supports pinning Node with a file in the project root. This repo includes `.node-version` set to `22.16.0`, which matches the current Pages v3 build image default.

If Cloudflare does not pick that up for any reason, add this environment variable in the Pages dashboard:

- `NODE_VERSION=22.16.0`

### 4. Save and deploy

Cloudflare will install dependencies, run the Astro build, and publish the `dist/` directory to a `*.pages.dev` URL.

After the first successful deploy, verify these routes:

- `/`
- `/about`
- `/blog`
- `/zinc`
- `/zinc/docs`
- `/feed.xml`
- `/sitemap-index.xml`
- `/robots.txt`

### 5. Configure build watch paths

Open the Pages project and go to:

`Settings` -> `Build` -> `Build watch paths`

Use:

- Include paths: `site/*, docs/*`
- Exclude paths: leave empty unless you later want to skip other directories

This matters because the docs pages are built from the repo-root `docs/` directory, not only from files inside `site/`.

### 6. Attach the production domain

After the `*.pages.dev` deployment looks correct:

1. Open the Pages project
2. Go to `Custom domains`
3. Add `zolotukhin.ai`
4. Optionally add `www.zolotukhin.ai`

If you use both domains, redirect `www` to the apex with a Cloudflare Redirect Rule. Do not rely on `_redirects` for that hostname-level redirect.

## Optional manual deploy after the project exists

Once the Pages project already exists, you can also deploy a local build manually:

```bash
cd site
npm run build
npx wrangler pages deploy dist --project-name <your-pages-project-name>
```

Use this only if you intentionally want a manual upload flow. Do not mix ad hoc manual deploys with Git-triggered production deploys unless you understand which one should be the source of truth.

## Current deployment behavior

- Production deploys should come from pushes to `main`
- Pull requests get preview deployments from Cloudflare Pages
- GitHub Actions in `.github/workflows/site-check.yml` only validate the build; they do not deploy the site
