#!/usr/bin/env node
/**
 * validate-docs.mjs — integrity checks for the V2 publishing pipeline.
 *
 * 1. internal links: every /knowledge-tree/... link in any site page must
 *    resolve to a known page (published chapter, hand page); links to draft
 *    pages are errors (they 404 in production builds).
 * 2. relative markdown links are forbidden — the site convention is absolute
 *    /knowledge-tree/ paths (Starlight does not rewrite relative links).
 * 3. redirects.json: sources must not shadow real pages, targets must exist.
 */
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join, relative, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { posix } from 'node:path';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DOCS = join(ROOT, 'src/content/docs');
const SITE_BASE = '/knowledge-tree';
const REDIRECTS_PATH = join(ROOT, 'src/data/redirects.json');

function walkDocs(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walkDocs(p, out);
    else if (/\.(md|mdx)$/.test(e.name)) out.push(p);
  }
  return out;
}

function slugOf(file) {
  let s = posix.normalize(relative(DOCS, file).split('\\').join('/'));
  s = s.replace(/\.(md|mdx)$/, '');
  if (s === 'index') return '';
  if (s.endsWith('/index')) s = s.slice(0, -'/index'.length);
  return s;
}

// ---------- known slugs ----------
const map = JSON.parse(readFileSync(join(ROOT, 'src/data/content-map.json'), 'utf8'));
const published = new Set(map.entries.filter((e) => e.status === 'published').map((e) => e.slug));
const drafts = new Set(map.entries.filter((e) => e.status === 'draft').map((e) => e.slug));
const files = walkDocs(DOCS);
const known = new Set([...published]);
for (const f of files) {
  const s = slugOf(f);
  if (s === '404') continue;
  known.add(s);
}

// ---------- link scan ----------
let errors = 0;
for (const f of files) {
  const rel = relative(ROOT, f);
  const lines = readFileSync(f, 'utf8').split(/\r?\n/);
  const inFence = new Array(lines.length).fill(false);
  let open = null;
  lines.forEach((line, i) => {
    const m = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (open) {
      inFence[i] = true;
      if (m && m[1][0] === open.ch && m[1].length >= open.len && !/\S/.test(line.slice(line.indexOf(m[1]) + m[1].length))) open = null;
    } else if (m) { open = { ch: m[1][0], len: m[1].length }; inFence[i] = true; }
  });
  lines.forEach((line, i) => {
    if (inFence[i]) return;
    for (const m of line.matchAll(/\]\(([^)\s]*)\)/g)) {
      const t = m[1];
      if (/^(https?:|mailto:|#)/.test(t)) continue;
      if (t.startsWith('./') || t.startsWith('../')) {
        console.error(`${rel}:${i + 1}: relative link forbidden (use ${SITE_BASE}/... absolute): ${t}`);
        errors++;
        continue;
      }
      if (!t.startsWith(SITE_BASE + '/')) continue;
      const slug = t.slice(SITE_BASE.length).replace(/^\/+/, '').replace(/\/+$/, '');
      if (drafts.has(slug)) {
        console.error(`${rel}:${i + 1}: link to draft page (excluded from prod): ${t}`);
        errors++;
      } else if (!known.has(slug)) {
        console.error(`${rel}:${i + 1}: broken internal link: ${t}`);
        errors++;
      }
    }
  });
}

// ---------- redirects ----------
if (existsSync(REDIRECTS_PATH)) {
  const redirects = JSON.parse(readFileSync(REDIRECTS_PATH, 'utf8'));
  for (const [from, to] of Object.entries(redirects)) {
    const f = from.replace(/^\/+/, '').replace(/\/+$/, '');
    const t = to.replace(/^\/+/, '').replace(/\/+$/, '');
    if (known.has(f)) { console.error(`redirect source shadows a real page: ${from}`); errors++; }
    if (drafts.has(t)) { console.error(`redirect target is a draft page: ${to}`); errors++; }
    else if (!known.has(t)) { console.error(`redirect target does not exist: ${to}`); errors++; }
  }
}

if (errors) { console.error(`validation FAILED (${errors} problem(s))`); process.exit(1); }
console.log(`validation OK: ${files.length} pages, ${known.size} known slugs, links & redirects consistent`);
