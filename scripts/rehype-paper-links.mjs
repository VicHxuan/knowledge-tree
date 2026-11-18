import { readFileSync } from 'node:fs';
import { relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { referenceIndex, normalizePaperURL } from './lib/papers.mjs';
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const index = referenceIndex(read('../src/data/papers.json').papers);
const edges = read('../src/data/paper-relations.json').edges;
const docs = fileURLToPath(new URL('../src/content/docs/', import.meta.url));
export function annotatePaperLinks(tree, index, allowed) {
  function walk(node) {
    if (!node.children || node.tagName === 'a') return;
    const out = [];
    for (const child of node.children) {
      walk(child);
      out.push(child);
      if (child.type !== 'element' || child.tagName !== 'a') continue;
      const paper = index.get(normalizePaperURL(child.properties?.href));
      if (!paper || !allowed.has(paper.id)) continue;
      out.push({ type: 'text', value: ' ' }, {
        type: 'element', tagName: 'a',
        properties: { href: `/knowledge-tree/papers/#${paper.id}`, className: ['kt-paper-catalog-link'],
          'aria-label': `查看文献记录：${paper.title}` },
        children: [{ type: 'text', value: '文献' }],
      });
    }
    node.children = out;
  }
  walk(tree);
}
export default function rehypePaperLinks() {
  return (tree, file) => {
    if (!file.path) return;
    const slug = relative(docs, file.path).replace(/\\/g, '/').replace(/\.mdx?$/, '');
    const allowed = new Set(edges.filter(e => e.article === slug).map(e => e.paper));
    if (allowed.size) annotatePaperLinks(tree, index, allowed);
  };
}
