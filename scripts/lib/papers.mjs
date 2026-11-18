/** Public bibliography validation and reference matching shared by build and tests. */
export function normalizePaperURL(value) {
  try {
    const u = new URL(value);
    if (!['http:', 'https:'].includes(u.protocol)) return null;
    let host = u.hostname.toLowerCase().replace(/^www\./, '');
    let path = decodeURIComponent(u.pathname).replace(/\/$/, '');
    if (host === 'arxiv.org') path = path.replace(/^\/pdf\//, '/abs/').replace(/(?:v\d+)?(?:\.pdf)?$/, '');
    if (host === 'doi.org' || host === 'dx.doi.org') return `doi:${path.slice(1).toLowerCase()}`;
    if (host === 'proceedings.neurips.cc' || host === 'papers.nips.cc') {
      host = 'proceedings.neurips.cc';
      path = path.replace('/paper_files/paper/', '/paper/');
    }
    return `${host}${path}${u.search}`;
  } catch { return null; }
}

export function referenceIndex(papers) {
  const index = new Map();
  for (const paper of papers) {
    for (const url of [paper.url, ...(paper.sources ?? [])]) {
      const key = normalizePaperURL(url);
      if (!key) continue;
      if (index.has(key) && index.get(key).id !== paper.id) throw new Error(`Ambiguous paper URL: ${url}`);
      index.set(key, paper);
    }
  }
  return index;
}

export function validatePapers(data, relations, articles, domains) {
  if (data.version !== 1 || relations.version !== 1) throw new Error('Unsupported papers schema');
  const ids = new Set();
  for (const p of data.papers) {
    if (!/^[a-z0-9][a-z0-9-]*$/.test(p.id) || ids.has(p.id)) throw new Error(`Invalid/duplicate paper ID: ${p.id}`);
    ids.add(p.id);
    if (!p.title || !Array.isArray(p.authors) || !p.authors.length) throw new Error(`Missing bibliography: ${p.id}`);
    for (const topic of p.topics) if (!domains.includes(topic)) throw new Error(`Unknown topic: ${topic}`);
    for (const url of [p.url, p.openAccessUrl, p.codeUrl, ...p.sources].filter(Boolean)) {
      if (!normalizePaperURL(url) || /(?:localhost|127\.0\.0\.1)/i.test(new URL(url).hostname)) throw new Error(`Non-public link: ${url}`);
    }
    const allowed = new Set(['id','title','authors','year','venue','doi','url','topics','sources','openAccessUrl','codeUrl']);
    for (const field of Object.keys(p)) if (!allowed.has(field)) throw new Error(`Private/unexpected field: ${field}`);
  }
  const seen = new Set();
  for (const edge of relations.edges) {
    if (!ids.has(edge.paper)) throw new Error(`Unknown paper: ${edge.paper}`);
    if (!articles.some(e => e.slug === edge.article)) throw new Error(`Unknown article: ${edge.article}`);
    const pair = `${edge.paper}\0${edge.article}`;
    if (seen.has(pair)) throw new Error(`Repeated paper relationship: ${pair}`);
    seen.add(pair);
    if (typeof edge.note !== 'string' || !edge.note) throw new Error(`Missing relationship description: ${pair}`);
  }
  referenceIndex(data.papers);
}
