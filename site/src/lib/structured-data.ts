export interface BreadcrumbItem {
  name: string;
  href?: string;
}

export function buildBreadcrumbList(siteUrl: string, items: BreadcrumbItem[]) {
  return {
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => {
      const listItem: Record<string, unknown> = {
        '@type': 'ListItem',
        position: index + 1,
        name: item.name,
      };

      if (item.href) {
        listItem.item = new URL(item.href, siteUrl).toString();
      }

      return listItem;
    }),
  };
}
