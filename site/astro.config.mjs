// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
export default defineConfig({
  site: 'https://zolotukhin.ai',
  integrations: [mdx(), sitemap()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      defaultColor: 'light',
      themes: {
        light: 'github-light',
        dark: 'vitesse-dark',
      },
      langs: ['zig', 'glsl', 'json', 'bash', 'typescript', 'c'],
    },
  },
});
