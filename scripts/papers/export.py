#!/usr/bin/env python3
"""Export an explicit, local selection of Zotero items to public website metadata."""
import argparse
import json
from pathlib import Path
import re
from zotero import REPO, save


def export(paper):
    private = paper / 'maintenance'
    selection = json.loads((private / 'site-map.json').read_text())
    items = {x['key']: x for x in json.loads((private / 'zotero-items.json').read_text())}
    papers = []
    for entry in selection['papers']:
        obj = items.get(entry['itemKey'])
        if not obj:
            raise ValueError(f'Missing selected item {entry["id"]}; reconcile merges before exporting')
        d = obj['data']
        authors = [x.get('name') or ' '.join(filter(None, [x.get('firstName'), x.get('lastName')]))
                   for x in d.get('creators', []) if x['creatorType'] == 'author']
        date = d.get('date', '')
        match = re.search(r'\b(?:19|20)\d{2}\b', date)
        doi = d.get('DOI', '')
        url = 'https://doi.org/' + doi if doi else d.get('url', '')
        if not url.startswith(('https://', 'http://')):
            raise ValueError(f'No public source for {entry["id"]}')
        sources = sorted(set(entry.get('sources', []) + [u for u in [d.get('url'), url] if u]))
        record = {'id': entry['id'], 'title': d['title'], 'authors': authors,
                  'year': int(match[0]) if match else None,
                  'venue': d.get('publicationTitle') or d.get('proceedingsTitle') or d.get('institution') or d.get('publisher') or d.get('repository') or '',
                  'doi': doi, 'url': url, 'topics': entry['topics'], 'sources': sources}
        # Resource links are a curated public allowlist, not attachment paths or notes.
        for field in ['openAccessUrl', 'codeUrl']:
            if entry.get(field):
                record[field] = entry[field]
        papers.append(record)
    save(REPO / 'src/data/papers.json', {'version': 1, 'papers': sorted(papers, key=lambda p: p['id'])})
    print(f'Exported {len(papers)} selected records. Private notes, paths and credentials excluded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=REPO.parent / 'paper')
    args = parser.parse_args()
    export(args.paper)
