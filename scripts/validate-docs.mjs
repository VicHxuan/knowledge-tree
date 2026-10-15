#!/usr/bin/env node
/**
 * validate-docs.mjs — integrity checks for the V2 publishing pipeline.
 *
 * 链接与重定向：
 * 1. internal links: every /knowledge-tree/... link in any site page must
 *    resolve to a known page (published chapter, hand page); links to draft
 *    pages are errors (they 404 in production builds).
 * 2. relative markdown links are forbidden — the site convention is absolute
 *    /knowledge-tree/ paths (Starlight does not rewrite relative links).
 * 3. redirects.json: sources must not shadow real pages, targets must exist,
 *    chains must not cycle.
 *
 * 六模块结构（P0）：
 * 4. published 文章骨架 = 1 个 H1 + 按序出现的六个标准 `##` 模块。模块标题
 *    从 tech_stack/TECHNICAL_ARTICLE_TEMPLATE.md 的骨架围栏中提取（不自行
 *    编造），仅检查、不自动修复；草稿与手写页豁免。
 *
 * 数据一致性（P0）：
 * 5. content-map：source/slug/output 唯一、status 合法、source 落盘、output
 *    遵循 `src/content/docs/<slug>.md` 约定、slug 顶层目录是已注册知识域。
 * 6. knowledge-relations：节点 id 唯一且对应映射条目、边端点存在、无重复边、
 *    无自环、prerequisite 无环（DAG）、展示字段与 content-map 漂移时告警。
 */
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join, relative, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { posix } from 'node:path';
import { DOMAINS } from '../src/data/domains.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DOCS = join(ROOT, 'src/content/docs');
const TS = join(ROOT, 'tech_stack');
const SITE_BASE = '/knowledge-tree';
const REDIRECTS_PATH = join(ROOT, 'src/data/redirects.json');
const TEMPLATE_PATH = join(TS, 'TECHNICAL_ARTICLE_TEMPLATE.md');
const RELATIONS_PATH = join(ROOT, 'src/data/knowledge-relations.json');

let errors = 0;
let warnings = 0;
const err = (msg) => { console.error(`  ✗ ${msg}`); errors++; };
const warn = (msg) => { console.warn(`  ⚠ ${msg}`); warnings++; };

function walkDocs(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walkDocs(p, out);
    else if (/\.(md|mdx)$/.test(e.name)) out.push(p);
  }
  return out;
}

/** fence-aware 每行围栏标记（与 sync-docs 同规则） */
function markFences(lines) {
  const inFence = new Array(lines.length).fill(false);
  let open = null;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^\s{0,3}(`{3,}|~{3,})/);
    if (open) {
      if (m && m[1][0] === open.ch && m[1].length >= open.len && !/\S/.test(lines[i].slice(lines[i].indexOf(m[1]) + m[1].length))) {
        open = null;
      }
      inFence[i] = true;
    } else if (m) {
      open = { ch: m[1][0], len: m[1].length };
      inFence[i] = true;
    }
  }
  return inFence;
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
const mapBySlug = new Map(map.entries.map((e) => [e.slug, e]));
const published = new Set(map.entries.filter((e) => e.status === 'published').map((e) => e.slug));
const drafts = new Set(map.entries.filter((e) => e.status === 'draft').map((e) => e.slug));
const files = walkDocs(DOCS);
const known = new Set([...published]);
for (const f of files) {
  const s = slugOf(f);
  if (s === '404') continue;
  known.add(s);
}

// ---------- 1. link scan ----------
console.log('[links] 扫描站内链接…');
for (const f of files) {
  const rel = relative(ROOT, f);
  const lines = readFileSync(f, 'utf8').split(/\r?\n/);
  const inFence = markFences(lines);
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

// ---------- 2. redirects ----------
if (existsSync(REDIRECTS_PATH)) {
  console.log('[redirects] 校验重定向表…');
  const redirects = JSON.parse(readFileSync(REDIRECTS_PATH, 'utf8'));
  for (const [from, to] of Object.entries(redirects)) {
    const f = from.replace(/^\/+/, '').replace(/\/+$/, '');
    const t = to.replace(/^\/+/, '').replace(/\/+$/, '');
    if (known.has(f)) err(`redirect source shadows a real page: ${from}`);
    if (drafts.has(t)) err(`redirect target is a draft page: ${to}`);
    else if (!known.has(t)) err(`redirect target does not exist: ${to}`);
  }
  // 链式重定向环检测：A→B→…→A
  for (const from of Object.keys(redirects)) {
    const seen = new Set([from.replace(/^\/+|\/+$/g, '')]);
    let cur = from.replace(/^\/+|\/+$/g, '');
    while (redirects[cur] != null) {
      cur = redirects[cur].replace(/^\/+|\/+$/g, '');
      if (seen.has(cur)) { err(`redirect cycle detected starting at: ${from}`); break; }
      seen.add(cur);
    }
  }
}

// ---------- 3. 六模块结构 ----------
console.log('[structure] 校验已发布文章的六模块结构…');
function templateModules() {
  const lines = readFileSync(TEMPLATE_PATH, 'utf8').split(/\r?\n/);
  const inFence = markFences(lines);
  const out = [];
  lines.forEach((line, i) => {
    if (!inFence[i]) return;
    if (/^##\s+\d+\.\s\S/.test(line)) out.push(line.replace(/^##\s+/, '').replace(/\s+$/, ''));
  });
  if (out.length !== 6) throw new Error(`template skeleton expected 6 module headings, got ${out.length}`);
  return out;
}
const MODULES = templateModules();

let structChecked = 0;
for (const e of map.entries) {
  if (e.status !== 'published') continue;
  const src = join(ROOT, e.source);
  if (!existsSync(src)) continue; // source 缺失在一致性检查里报
  const rel = relative(ROOT, src);
  const lines = readFileSync(src, 'utf8').split(/\r?\n/);
  const inFence = markFences(lines);
  const h1 = [];
  const h2 = [];
  let prevLevel = 0;
  const levelSkips = [];
  lines.forEach((line, i) => {
    if (inFence[i]) return;
    const m = line.match(/^(#{1,6})\s+(.+)/);
    if (!m) return;
    const level = m[1].length;
    if (level === 1) h1.push({ line: i + 1, text: m[2].trim() });
    if (level === 2) h2.push({ line: i + 1, text: m[2].trim() });
    if (prevLevel && level > prevLevel + 1) levelSkips.push({ line: i + 1, text: m[2].trim(), from: prevLevel, to: level });
    if (level > 1) prevLevel = level;
  });
  structChecked++;

  if (h1.length !== 1) {
    err(`${rel}: expected exactly 1 H1, found ${h1.length}${h1.map((h) => ` (L${h.line})`).join('')}`);
  }
  const actual = h2.map((h) => h.text);
  const missing = MODULES.filter((m) => !actual.includes(m));
  const extra = actual.filter((t) => !MODULES.includes(t));
  const inOrder = MODULES.every((m, idx) => actual[idx] === m);
  if (actual.length !== MODULES.length || !inOrder) {
    err(`${rel}: H2 module sequence mismatch (expected 6 standard modules in order)`);
    if (missing.length) console.error(`      缺失模块: ${missing.map((m) => m.replace(/^##\s*/, '')).join('；')}`);
    if (extra.length) console.error(`      非标准 H2: ${extra.join('；')}`);
    if (!missing.length && !extra.length) console.error(`      模块齐全但顺序或措辞不同（首个差异：L${h2[MODULES.findIndex((m, idx) => actual[idx] !== m)]?.line ?? '?'}）`);
  }
  for (const skip of levelSkips) {
    warn(`${rel}:${skip.line}: heading level skip ${'#'.repeat(skip.from)} → ${'#'.repeat(skip.to)}: ${skip.text}`);
  }
}
console.log(`  六模块标准：${MODULES.join(' / ')}`);
console.log(`  checked ${structChecked} published article(s)`);

// ---------- 4. data consistency ----------
console.log('[consistency] 校验 content-map / knowledge-relations…');
const DOMAIN_IDS = new Set(DOMAINS.map((d) => d.id));
{
  const bySource = new Set();
  const bySlug = new Set();
  const byOutput = new Set();
  for (const e of map.entries) {
    if (!['published', 'draft'].includes(e.status)) err(`content-map ${e.slug}: invalid status "${e.status}"`);
    if (bySlug.has(e.slug)) err(`content-map duplicate slug: ${e.slug}`);
    if (bySource.has(e.source)) err(`content-map duplicate source: ${e.source}`);
    if (byOutput.has(e.output)) err(`content-map duplicate output: ${e.output}`);
    bySlug.add(e.slug); bySource.add(e.source); byOutput.add(e.output);
    if (!existsSync(join(ROOT, e.source))) err(`content-map source missing on disk: ${e.source}`);
    if (e.output !== `src/content/docs/${e.slug}.md`) err(`content-map ${e.slug}: output must be src/content/docs/<slug>.md, got ${e.output}`);
    const top = e.slug.split('/')[0];
    if (!DOMAIN_IDS.has(top)) err(`content-map ${e.slug}: top-level dir "${top}" is not a registered domain (src/data/domains.mjs)`);
  }
}
{
  const relations = JSON.parse(readFileSync(RELATIONS_PATH, 'utf8'));
  const nodeIds = new Set();
  for (const n of relations.nodes) {
    if (nodeIds.has(n.id)) { err(`relations duplicate node id: ${n.id}`); continue; }
    nodeIds.add(n.id);
    const m = mapBySlug.get(n.id);
    if (!m) {
      err(`relations node not present in content-map: ${n.id}`);
    } else {
      if (n.domain !== n.id.split('/')[0]) warn(`relations node ${n.id}: domain "${n.domain}" != slug domain "${n.id.split('/')[0]}"`);
      if (m.status !== n.status) warn(`relations node ${n.id}: status "${n.status}" drifted from content-map "${m.status}" (display uses content-map)`);
      if (m.title !== n.title) warn(`relations node ${n.id}: title drifted from content-map (display uses content-map)`);
    }
  }
  const edgeKeys = new Set();
  for (const e of relations.edges) {
    if (!['prerequisite', 'related'].includes(e.type)) { err(`relations edge with unknown type "${e.type}": ${e.from} -> ${e.to}`); continue; }
    if (!nodeIds.has(e.from) || !nodeIds.has(e.to)) { err(`relations edge references unknown node: ${e.from} -> ${e.to}`); continue; }
    if (e.from === e.to) { err(`relations self-loop: ${e.from}`); continue; }
    const key = `${e.type}|${e.from}|${e.to}`;
    if (edgeKeys.has(key)) err(`relations duplicate edge: ${e.from} -[${e.type}]-> ${e.to}`);
    edgeKeys.add(key);
    if (e.type === 'related' && edgeKeys.has(`related|${e.to}|${e.from}`)) warn(`relations duplicate reversed related edge: ${e.from} <-> ${e.to}`);
  }
  // prerequisite 无环检测（DFS 三色标记）
  const pres = new Map();
  for (const e of relations.edges) {
    if (e.type !== 'prerequisite') continue;
    (pres.get(e.from) ?? pres.set(e.from, []).get(e.from)).push(e.to);
  }
  const color = new Map(); // 0=白 1=灰 2=黑
  const stack = [];
  const visit = (id) => {
    if (color.get(id) === 1) {
      const cycleStart = stack.indexOf(id);
      err(`prerequisite cycle detected: ${[...stack.slice(cycleStart), id].join(' -> ')}`);
      return;
    }
    if (color.get(id) === 2) return;
    color.set(id, 1);
    stack.push(id);
    for (const next of pres.get(id) ?? []) visit(next);
    stack.pop();
    color.set(id, 2);
  };
  for (const n of relations.nodes) visit(n.id);
}

if (errors) { console.error(`validation FAILED (${errors} error(s), ${warnings} warning(s))`); process.exit(1); }
console.log(`validation OK: ${files.length} pages, ${known.size} known slugs, ${structChecked} article skeletons, links/redirects/structure/data consistent (${warnings} warning(s))`);
