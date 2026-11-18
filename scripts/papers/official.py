#!/usr/bin/env python3
"""Import reviewed official editions and produce a version-aware local checklist.

The manifest and receipts are private. Downloads are reviewed before this command;
credentials, browser sessions and subscription URLs never enter the public export.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, parse_qs

from zotero import Zotero, PREFIX, digest, save, refresh, write_csv


VERSIONS = {'published', 'technical-report', 'supplement', 'author-book'}


def inspect_pdf(path):
    with path.open('rb') as stream:
        if stream.read(5) != b'%PDF-':
            raise ValueError('Expected a PDF, not an HTML download page')
    info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True,
                          timeout=30, check=True).stdout
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    text = subprocess.run(['pdftotext', '-f', '1', '-l', '2', str(path), '-'],
                          capture_output=True, text=True, timeout=30, check=True).stdout
    return {'pages': pages, 'text': text,
            'arxivStamp': bool(re.search(r'arXiv:\s*\d{4}\.\d{4,5}', text, re.I)),
            'sha256': digest(path)}


def validate_entry(entry, paper):
    if not entry.get('reviewed') or not entry.get('evidence'):
        raise ValueError('A title and edition review is required')
    if entry['version'] not in VERSIONS:
        raise ValueError('Unsupported edition')
    url = urlsplit(entry['sourceUrl'])
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
        raise ValueError('Use a public HTTPS source without credentials')
    if url.hostname == 'arxiv.org' or url.hostname.endswith('.arxiv.org'):
        raise ValueError('arXiv downloads are excluded by the collection policy')
    if any(re.search(r'token|signature|credential|password|session|^x-amz-', key, re.I)
           for key in parse_qs(url.query)):
        raise ValueError('Use the canonical source, not a signed download URL')
    path = Path(entry['file'])
    if not path.is_absolute():
        path = paper / path
    info = inspect_pdf(path)
    if info['sha256'] != entry['sha256']:
        raise ValueError('PDF changed after review')
    if info['arxivStamp']:
        raise ValueError('PDF carries an arXiv version stamp')
    return path, info


def import_editions(z, manifest):
    receipt_file = z.private / 'official-imports.json'
    receipts = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
    # Validate the whole input before the first library write.
    checked = [(entry, *validate_entry(entry, z.paper)) for entry in manifest['editions']]
    for entry, path, info in checked:
        parent = entry['itemKey']
        identity = parent + ':' + info['sha256']
        receipt = receipts.get(identity, {})
        parent_obj = z.request(f'{PREFIX}/items/{parent}')
        if parent_obj['data']['itemType'] in ('note', 'attachment', 'annotation'):
            raise ValueError('A bibliographic parent is required')
        child = receipt.get('attachmentKey') or entry.get('existingAttachment')
        if child:
            obj = z.request(f'{PREFIX}/items/{child}')
            if obj['data'].get('parentItem') != parent:
                raise ValueError('Attachment belongs to a different parent')
        else:
            for obj in z.request(f'{PREFIX}/items/{parent}/children'):
                if obj['data']['itemType'] != 'attachment':
                    continue
                current = z.file_path(obj['key'])
                if current and current.is_file() and digest(current) == info['sha256']:
                    child = obj['key']
                    break
        if not child:
            title = {'published': '正式发表版本', 'technical-report': '官方技术报告',
                     'supplement': '官方补充材料', 'author-book': '作者官方书稿'}[entry['version']]
            child = z.create('items', {
                'itemType': 'attachment', 'parentItem': parent,
                'linkMode': 'imported_url', 'title': title,
                'url': entry['sourceUrl'], 'contentType': 'application/pdf',
                'filename': path.name, 'tags': [], 'relations': {},
            })
        receipt = {**receipt, 'itemKey': parent, 'attachmentKey': child,
                   'sha256': info['sha256'], 'sourceUrl': entry['sourceUrl'],
                   'version': entry['version'], 'evidence': entry['evidence'],
                   'origin': entry.get('origin', 'download'), 'pages': info['pages']}
        receipts[identity] = receipt
        save(receipt_file, receipts)  # Persist before upload for resumable retries.
        current = z.file_path(child)
        if not current or not current.is_file():
            z.upload(child, path)
            current = z.file_path(child)
        if not current or not current.is_file() or digest(current) != info['sha256']:
            raise ValueError('Stored attachment failed SHA-256 verification')
        receipt['verified'] = True
        receipt['storedPath'] = str(current)
        save(receipt_file, receipts)
        print(parent, child, entry['version'], 'verified')
    refresh(z)
    checklist(z.paper)


def checklist(paper):
    private = paper / 'maintenance'
    def read(name, default):
        path = private / name
        return json.loads(path.read_text()) if path.exists() else default
    items = read('zotero-items.json', [])
    receipts = read('official-imports.json', {})
    links = read('generated-links.json', {})
    # Zotero may rename files after a bibliographic metadata update.
    for receipt in receipts.values():
        suffix = '[' + receipt['attachmentKey'] + '].pdf'
        candidates = [target for rel, target in links.items()
                      if rel.startswith('library/' + receipt['itemKey'] + '/') and rel.endswith(suffix)]
        if len(candidates) == 1:
            receipt['storedPath'] = candidates[0]
    if receipts:
        save(private / 'official-imports.json', receipts)
    decisions = read('official-review.json', {})
    sources = read('official-sources.json', {})
    rows = []
    for obj in items:
        d, key = obj['data'], obj['key']
        if d['itemType'] in ('attachment', 'note', 'annotation'):
            continue
        editions = [r for r in receipts.values() if r['itemKey'] == key and r.get('verified')
                    and Path(r.get('storedPath', '')).is_file()
                    and digest(Path(r['storedPath'])) == r['sha256']]
        accepted = [r for r in editions if r['version'] in ('published', 'technical-report')
                    or (r['version'] == 'supplement' and re.search(r'supplement|补充材料', d['title'], re.I))]
        decision = decisions.get(key, {})
        row = {'key': key, 'title': d['title'],
               'status': '已核验正式全文' if accepted else decision.get('status', '待核验正式版本'),
               'reason': '' if accepted else decision.get('reason', '现有附件或来源尚未核验'),
               'source': sources.get(key, {}).get('landing', d.get('url', '')),
               'attachments': ';'.join(r['attachmentKey'] for r in editions),
               'versions': ';'.join(sorted(set(r['version'] for r in editions)))}
        rows.append(row)
    fields = ['key', 'title', 'status', 'reason', 'source', 'attachments', 'versions']
    write_csv(paper / 'catalog/official-fulltext.csv', fields, rows)
    write_csv(paper / 'catalog/missing-official-fulltext.csv', fields,
              [r for r in rows if r['status'] != '已核验正式全文'])
    print(f'Official editions: {sum(r["status"] == "已核验正式全文" for r in rows)}/{len(rows)} records')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['import', 'checklist'])
    p.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[3] / 'paper')
    p.add_argument('--manifest', type=Path)
    args = p.parse_args()
    if args.command == 'import':
        if not args.manifest:
            p.error('--manifest is required for import')
        import_editions(Zotero(args.paper), json.loads(args.manifest.read_text()))
    else:
        checklist(args.paper)
