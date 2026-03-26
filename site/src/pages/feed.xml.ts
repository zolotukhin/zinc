import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = (await getCollection('posts', ({ data }) => !data.draft))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());

  return rss({
    title: 'Stepan Zolotukhin — zolotukhin.ai',
    description: 'Writing about ZINC, local LLM inference on AMD GPUs, Vulkan compute, Zig, and systems engineering.',
    site: context.site!.toString(),
    items: posts.map(post => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.excerpt,
      link: `/blog/${post.id}`,
      categories: post.data.tags,
    })),
  });
}
