import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePaperURL, referenceIndex, validatePapers } from '../scripts/lib/papers.mjs';
import { annotatePaperLinks } from '../scripts/rehype-paper-links.mjs';
const paper = { id: 'ppo', title: 'PPO', authors: ['John Schulman'], year: 2017, venue: 'arXiv', doi: '', url: 'https://arxiv.org/abs/1707.06347', sources: [], topics: ['rl'] };
const data = () => ({version:1,papers:[structuredClone(paper)]});
const relation = {version:1,edges:[{paper:'ppo',article:'rl/ppo',note:'原始方法'}]};
const articles = [{slug:'rl/ppo'}];
test('PDF and versioned arXiv URLs resolve to one record; DOI is case-insensitive', () => {
  const index = referenceIndex([paper]);
  assert.equal(index.get(normalizePaperURL('http://arxiv.org/pdf/1707.06347v2.pdf')).id, 'ppo');
  assert.equal(normalizePaperURL('https://doi.org/10.1109/ABC'), normalizePaperURL('http://dx.doi.org/10.1109/abc'));
  assert.equal(normalizePaperURL('file:///home/user/paper.pdf'), null);
});
test('conflicting identifiers and dangling article relationships fail validation', () => {
  assert.throws(() => referenceIndex([paper,{...paper,id:'duplicate'}]), /Ambiguous/);
  assert.throws(() => validatePapers(data(),relation,[],['rl']), /Unknown article/);
  assert.throws(() => validatePapers(data(),{version:1,edges:[...relation.edges,...relation.edges]},articles,['rl']), /Repeated/);
});
test('public export rejects private fields and local service links', () => {
  const d = data(); d.papers[0].note = 'private';
  assert.throws(() => validatePapers(d,relation,articles,['rl']), /Private/);
  delete d.papers[0].note; d.papers[0].url = 'http://127.0.0.1:23119/api/';
  assert.throws(() => validatePapers(d,relation,articles,['rl']), /Non-public/);
});
test('citation decoration preserves source link and only decorates selected relationships', () => {
  const a = {type:'element',tagName:'a',properties:{href:paper.url},children:[{type:'text',value:'source'}]};
  const tree = {type:'root',children:[{type:'element',tagName:'p',children:[a]}]};
  annotatePaperLinks(tree, referenceIndex([paper]), new Set());
  assert.equal(tree.children[0].children.length, 1);
  annotatePaperLinks(tree, referenceIndex([paper]), new Set(['ppo']));
  assert.equal(tree.children[0].children[0], a);
  assert.equal(tree.children[0].children[2].properties.href, '/knowledge-tree/papers/#ppo');
});
