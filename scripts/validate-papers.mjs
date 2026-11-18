import { readFileSync } from 'node:fs';
import { validatePapers } from './lib/papers.mjs';
import { DOMAINS } from '../src/data/domains.mjs';
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const papers = read('../src/data/papers.json');
const relations = read('../src/data/paper-relations.json');
validatePapers(papers, relations, read('../src/data/content-map.json').entries, DOMAINS.map(d => d.id));
console.log(`Papers OK: ${papers.papers.length} public records, ${relations.edges.length} article relationships`);
