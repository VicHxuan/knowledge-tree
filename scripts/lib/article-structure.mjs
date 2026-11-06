import { readFileSync } from 'node:fs';

/** Mark fenced code lines so examples cannot be mistaken for article headings. */
export function markFences(lines) {
  const inFence = new Array(lines.length).fill(false);
  let open = null;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^\s{0,3}(`{3,}|~{3,})/);
    if (open) {
      if (m && m[1][0] === open.ch && m[1].length >= open.len && !/\S/.test(lines[i].slice(lines[i].indexOf(m[1]) + m[1].length))) open = null;
      inFence[i] = true;
    } else if (m) {
      open = { ch: m[1][0], len: m[1].length };
      inFence[i] = true;
    }
  }
  return inFence;
}

/** The writing template is the single source for the six standard modules. */
export function readArticleModules(template = new URL('../../tech_stack/TECHNICAL_ARTICLE_TEMPLATE.md', import.meta.url)) {
  const lines = readFileSync(template, 'utf8').split(/\r?\n/);
  const inFence = markFences(lines);
  const modules = lines.filter((line, i) => inFence[i] && /^##\s+\d+\.\s\S/.test(line))
    .map((line) => line.replace(/^##\s+/, '').trimEnd());
  if (modules.length !== 6) throw new Error(`template skeleton expected 6 module headings, got ${modules.length}`);
  return modules;
}
