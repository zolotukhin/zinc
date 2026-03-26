import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).optional().default([]),
    excerpt: z.string(),
    draft: z.boolean().optional().default(false),
  }),
});

const docs = defineCollection({
  loader: glob({ pattern: '*.md', base: '../docs' }),
});

export const collections = { posts, docs };
