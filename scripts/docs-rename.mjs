#!/usr/bin/env node
/** Move the source first; URLs stay stable unless --new-slug is supplied. */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DOMAINS } from '../src/data/domains.mjs';
import { parseArgs, renameRegistration } from './lib/docs-model.mjs';
import { renameHeadingAliasPage } from './lib/heading-aliases.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
try {
  const { flags, positional } = parseArgs(process.argv.slice(2), ['from', 'to', 'new-slug', 'new-title', 'new-label']);
  if (positional.length || !flags.from || !flags.to) throw new Error('用法: npm run docs:rename -- --from <旧slug> --to tech_stack/<新路径>.md [--new-slug <新slug>] [--new-title …] [--new-label …]');
  if (!existsSync(join(ROOT, flags.to))) throw new Error(`新文件不存在，请先移动原件: ${flags.to}`);
  const names = ['content-map', 'knowledge-relations', 'redirects', 'heading-aliases'];
  const paths = names.map((name) => join(ROOT, `src/data/${name}.json`));
  const [map, relations, redirects, headingAliases] = paths.map((path) => JSON.parse(readFileSync(path, 'utf8')));
  const newSlug = flags['new-slug'] ?? flags.from;
  if (newSlug !== flags.from && ['.md', '.mdx', '/index.md', '/index.mdx'].some((suffix) => existsSync(join(ROOT, `src/content/docs/${newSlug}${suffix}`)))) throw new Error(`新 slug 与已有页面冲突: ${newSlug}`);
  const result = renameRegistration(map, relations, redirects, { from: flags.from, source: flags.to, slug: newSlug, title: flags['new-title'], label: flags['new-label'] }, DOMAINS);
  result.headingAliases = renameHeadingAliasPage(headingAliases, flags.from, newSlug);
  // Validate all changes before any write, including route and heading-alias collisions.
  for (const [i, key] of ['map', 'relations', 'redirects', 'headingAliases'].entries()) writeFileSync(paths[i], JSON.stringify(result[key], null, 2) + '\n');
  console.log(`已更新原件: ${flags.to}\n公开 URL: ${result.entry.slug}${newSlug === flags.from ? '（保持不变）' : '（旧路由及历史别名已合并到新目标）'}\n检查原件之间的相对链接，再运行 docs:sync / docs:check。`);
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
