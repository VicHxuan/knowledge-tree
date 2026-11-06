import { getCollection } from 'astro:content';
import { defineRouteMiddleware, type StarlightRouteData } from '@astrojs/starlight/route-data';

type Entry = StarlightRouteData['sidebar'][number];
type Link = Extract<Entry, { type: 'link' }>;

/** Starlight names automatic groups after folders. Use the maintained index titles and order. */
export const onRequest = defineRouteMiddleware(async ({ locals }) => {
  const pages = await getCollection('docs');
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const clean = (href: string) => href.replace(/\/$/, '');
  const metadata = new Map(pages.map((page) => [
    `${base}/${page.id.replace(/\/index$/, '')}`,
    { title: page.data.title, order: page.data.sidebar.order ?? 999,
      index: /\/index\.mdx?$/.test(page.filePath ?? '') },
  ]));
  const groupInfo = (entry: Entry) => entry.type === 'group'
    ? entry.entries.filter((child): child is Link => child.type === 'link')
      .map((child) => ({ href: clean(child.href), meta: metadata.get(clean(child.href)) }))
      .filter((child) => child.meta?.index)
      .sort((a, b) => a.href.length - b.href.length)[0]
    : undefined;
  const order = (entry: Entry) => entry.type === 'link'
    ? metadata.get(clean(entry.href))?.order ?? 999
    : groupInfo(entry)?.meta?.order ?? 999;
  const decorate = (entries: Entry[]) => {
    for (const entry of entries) {
      if (entry.type !== 'group') continue;
      const overview = groupInfo(entry);
      if (entry.autogenerate && overview?.meta) entry.label = overview.meta.title;
      decorate(entry.entries);
      if (entry.entries.some((child) => child.autogenerate)) {
        entry.entries.sort((a, b) => {
          if (a.type === 'link' && clean(a.href) === overview?.href) return -1;
          if (b.type === 'link' && clean(b.href) === overview?.href) return 1;
          return order(a) - order(b);
        });
      }
    }
  };
  const route = locals.starlightRoute;
  decorate(route.sidebar);
  const flatten = (entries: Entry[]): Link[] => entries.flatMap((entry) =>
    entry.type === 'group' ? flatten(entry.entries) : [entry]);
  const links = flatten(route.sidebar).filter((entry) => entry.href.startsWith(`${base}/`));
  const current = links.findIndex((entry) => entry.isCurrent);
  if (current >= 0) {
    // Respect explicit per-page pagination choices while matching the visible sidebar order.
    if (route.entry.data.prev === undefined || route.entry.data.prev === true) route.pagination.prev = links[current - 1];
    if (route.entry.data.next === undefined || route.entry.data.next === true) route.pagination.next = links[current + 1];
  }
});
