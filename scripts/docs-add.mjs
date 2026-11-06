#!/usr/bin/env node
/** Register an existing source; --slug overrides the nested-path default. */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve, posix } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DOMAINS } from '../src/data/domains.mjs';
import { parseArgs, sourceDomain, deriveSlug, validateSlug } from './lib/docs-model.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
try {
  const { flags, positional } = parseArgs(process.argv.slice(2), ['title', 'label', 'status', 'order', 'description', 'slug']);
  if (positional.length !== 1) throw new Error('用法: npm run docs:add -- tech_stack/<路径>.md [--slug <稳定slug>] [--title …] [--status draft|published]');
  const source = positional[0];
  const domain = sourceDomain(source, DOMAINS);
  if (!existsSync(join(ROOT, source))) throw new Error(`原件不存在: ${source}`);
  const slug = validateSlug(flags.slug ?? deriveSlug(source, DOMAINS), DOMAINS);
  const mapPath = join(ROOT, 'src/data/content-map.json');
  const map = JSON.parse(readFileSync(mapPath, 'utf8'));
  if (map.entries.some((e) => e.source === source || e.slug === slug)) throw new Error(`source 或 slug 已注册: ${source} / ${slug}`);
  if (existsSync(join(ROOT, `src/content/docs/${slug}.md`)) || existsSync(join(ROOT, `src/content/docs/${slug}.mdx`)) || existsSync(join(ROOT, `src/content/docs/${slug}/index.md`)) || existsSync(join(ROOT, `src/content/docs/${slug}/index.mdx`))) throw new Error(`slug 与已有页面冲突: ${slug}`);
  const redirects = JSON.parse(readFileSync(join(ROOT, 'src/data/redirects.json'), 'utf8'));
  if (Object.keys(redirects).some((key) => key.replace(/^\/+|\/+$/g, '') === slug)) throw new Error(`slug 被已有重定向占用: ${slug}`);
  const status = flags.status ?? 'draft';
  if (!['draft', 'published'].includes(status)) throw new Error(`非法 status: ${status}`);
  const title = flags.title ?? readFileSync(join(ROOT, source), 'utf8').match(/^#\s+(.+)$/m)?.[1]?.trim();
  if (!title) throw new Error('原件需要一个 H1，或显式指定 --title');
  const section = posix.dirname(slug);
  const sameSection = map.entries.filter((e) => posix.dirname(e.slug) === section);
  const order = flags.order === undefined ? Math.max(0, ...sameSection.map((e) => e.order)) + 10 : Number(flags.order);
  if (!Number.isFinite(order) || order < 0) throw new Error('--order 必须是非负数字');
  const entry = { source, output: `src/content/docs/${slug}.md`, slug, title, label: flags.label ?? title, order, status, description: flags.description ?? `${title}——${domain.label}板块文章。` };
  const peers = sameSection.length ? sameSection : map.entries.filter((e) => e.slug.split('/')[0] === slug.split('/')[0]);
  const after = peers.length ? map.entries.indexOf(peers.at(-1)) + 1 : map.entries.length;
  map.entries.splice(after, 0, entry);
  writeFileSync(mapPath, JSON.stringify(map, null, 2) + '\n');
  console.log(`已注册 ${source}\n  → ${slug} (${status})\n节点展示信息自动派生；按教学关系补充 knowledge-relations.json 的 edges，然后运行 docs:sync / docs:check。`);
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
