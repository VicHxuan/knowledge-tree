#!/usr/bin/env node
/**
 * privacy-check.mjs — fail if personal data reaches anything publishable.
 *
 * Sensitive patterns are NOT hardcoded here (this file is public). They are
 * loaded at runtime, in priority order:
 *   1. $PRIVACY_PATTERNS  — JSON string, e.g. '[{"fatal":["..."]},{"warn":[...]}]'
 *   2. scripts/.privacy-patterns.json — local-only file (gitignored)
 * If neither is present the check prints a notice and passes (e.g. fresh CI
 * clone, which contains only already-public data by construction).
 *
 * Scans text files under the repo, skipping local-only dirs (tech_stack/personal,
 * tech_stack/references), build caches, and node_modules.
 */
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { join, relative, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const LOCAL_PATTERNS = join(ROOT, 'scripts/.privacy-patterns.json');
const SKIP_DIRS = new Set(['.git', '.astro', 'node_modules', 'dist', 'tech_stack/personal', 'tech_stack/references']);
const SKIP_FILES = new Set(['scripts/.privacy-patterns.json']); // local-only pattern source, gitignored

function loadPatterns() {
  if (process.env.PRIVACY_PATTERNS) {
    try { return JSON.parse(process.env.PRIVACY_PATTERNS); } catch { throw new Error('PRIVACY_PATTERNS is not valid JSON'); }
  }
  if (existsSync(LOCAL_PATTERNS)) return JSON.parse(readFileSync(LOCAL_PATTERNS, 'utf8'));
  return null;
}

const patterns = loadPatterns();
if (!patterns) {
  console.log('privacy check SKIPPED: no patterns configured (set PRIVACY_PATTERNS or create scripts/.privacy-patterns.json locally)');
  process.exit(0);
}
const FATAL = patterns.fatal ?? [];
const WARN = patterns.warn ?? []; // [{ p, allow: [paths] }]

function walk(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    const rel = relative(ROOT, p).split('\\').join('/');
    if (e.isDirectory()) {
      if (SKIP_DIRS.has(rel)) continue;
      walk(p, out);
    } else if (/\.(md|mdx|html|json|ya?ml|txt|mjs|astro)$/.test(e.name)) {
      if (!SKIP_FILES.has(rel)) out.push(p);
    }
  }
  return out;
}

let bad = 0;
for (const f of walk(ROOT)) {
  const rel = relative(ROOT, f).split('\\').join('/');
  const text = readFileSync(f, 'utf8');
  for (const pat of FATAL) {
    if (text.includes(pat)) { console.error(`FATAL ${rel}: contains a forbidden pattern`); bad++; }
  }
  for (const { p, allow } of WARN) {
    if (text.includes(p) && !(allow ?? []).includes(rel)) { console.error(`FATAL ${rel}: contains a restricted pattern (allowed only in ${allow.join(', ')})`); bad++; }
  }
}
if (bad) { console.error(`privacy check FAILED (${bad} hit(s))`); process.exit(1); }
console.log('privacy check OK: no personal data in publishable files');
