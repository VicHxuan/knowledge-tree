import { posix } from 'node:path';

export const SITE_BASE = '/knowledge-tree';

export function parseArgs(args, allowed) {
  const flags = {};
  const positional = [];
  for (let i = 0; i < args.length; i++) {
    if (!args[i].startsWith('--')) { positional.push(args[i]); continue; }
    const name = args[i].slice(2);
    if (!allowed.includes(name)) throw new Error(`未知参数 --${name}`);
    if (!args[i + 1] || args[i + 1].startsWith('--')) throw new Error(`--${name} 缺少值`);
    flags[name] = args[++i];
  }
  return { flags, positional };
}

export function sourceDomain(source, domains) {
  if (!/^tech_stack\/.+\.md$/.test(source) || source.includes('\\') || source.split('/').some((p) => !p || p === '.' || p === '..')) {
    throw new Error(`无效原件路径: ${source}`);
  }
  const domain = domains.find((d) => d.tsDir === source.split('/')[1]);
  if (!domain) throw new Error(`原件不在已注册领域内: ${source}`);
  return domain;
}

export function validateSlug(slug, domains) {
  if (!slug || !/^[a-z0-9][a-z0-9/-]*$/.test(slug) || slug.split('/').some((p) => !p || p === 'index') || !domains.some((d) => d.id === slug.split('/')[0])) {
    throw new Error(`无效 slug（使用已注册域和小写 kebab-case 路径）: ${slug}`);
  }
  return slug;
}

export function deriveSlug(source, domains) {
  const domain = sourceDomain(source, domains);
  const parts = source.split('/').slice(2).map((p) => p.replace(/\.md$/, '').replace(/^\d+[_-]/, '').replace(/_/g, '-').toLowerCase());
  return validateSlug([domain.id, ...parts].join('/'), domains);
}

export function splitTarget(target) {
  const i = target.search(/[?#]/);
  return i < 0 ? { pathname: target, suffix: '' } : { pathname: target.slice(0, i), suffix: target.slice(i) };
}

export function rewriteSourceTarget(target, sourceDir, bySource, external = {}) {
  if (/^(?:[a-z][a-z\d+.-]*:|\/|#|\?)/i.test(target)) return target;
  const { pathname, suffix } = splitTarget(target);
  const path = posix.normalize(posix.join(sourceDir, pathname));
  if (path.startsWith('../')) return target;
  if (external[path]) return external[path] + suffix;
  const entry = bySource.get(path);
  return entry ? `${SITE_BASE}/${entry.slug}/${suffix}` : target;
}

/** Canonicalize keys and collapse every redirect to its final route. */
export function flattenRedirects(redirects) {
  const canonical = (s) => '/' + s.replace(/^\/+|\/+$/g, '');
  const table = Object.fromEntries(Object.entries(redirects).map(([a, b]) => [canonical(a), canonical(b)]));
  for (const from of Object.keys(table)) {
    const seen = new Set([from]);
    let to = table[from];
    while (table[to]) {
      if (seen.has(to)) throw new Error(`重定向环: ${[...seen, to].join(' → ')}`);
      seen.add(to);
      to = table[to];
    }
    if (to === from) throw new Error(`重定向自环: ${from}`);
    table[from] = to;
  }
  return table;
}

/** Keep source moves independent of public URLs; a route change is explicit. */
export function renameRegistration(map, relations, redirects, { from, source, slug = from, title, label }, domains) {
  sourceDomain(source, domains);
  validateSlug(slug, domains);
  const entry = map.entries.find((e) => e.slug === from);
  if (!entry) throw new Error(`旧 slug 未注册: ${from}`);
  if (map.entries.some((e) => e !== entry && (e.slug === slug || e.source === source))) throw new Error(`source 或 slug 已被占用: ${source} / ${slug}`);
  if (slug !== from && Object.keys(redirects).some((key) => key.replace(/^\/+|\/+$/g, '') === slug)) throw new Error(`新 slug 被已有重定向占用: ${slug}`);
  Object.assign(entry, { source, slug, output: `src/content/docs/${slug}.md`, ...(title ? { title } : {}), ...(label ? { label } : {}) });
  if (slug !== from) {
    for (const edge of relations.edges) {
      if (edge.from === from) edge.from = slug;
      if (edge.to === from) edge.to = slug;
    }
    // Published routes and routes with incoming historical aliases remain reachable.
    if (entry.status === 'published' || Object.values(redirects).some((v) => v.replace(/^\/+|\/+$/g, '') === from)) redirects[`/${from}`] = `/${slug}`;
  }
  return { map, relations, redirects: flattenRedirects(redirects), entry };
}
