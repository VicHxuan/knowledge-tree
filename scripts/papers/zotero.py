#!/usr/bin/env python3
"""Local Zotero maintenance. Writes use the authorized API, never SQLite.

Private configuration and receipts live under paper/maintenance, outside the site.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[2]
API = 'http://127.0.0.1:23119/api'
PREFIX = '/users/0'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class Zotero:
    def __init__(self, paper):
        self.paper = paper
        self.private = paper / 'maintenance'
        self.auth_path = self.private / '.zotero-auth.json'
        with urllib.request.urlopen(API + '/', timeout=10) as response:
            self.server = response.headers['Zotero-Server-ID']
        self.auth = json.loads(self.auth_path.read_text()) if self.auth_path.exists() else None
        if self.auth and self.auth['serverID'] != self.server:
            raise RuntimeError('Zotero instance changed. Reconcile the library and authorize again.')

    def request(self, route, method='GET', data=None, headers=None, raw=False):
        url = route if route.startswith('http') else API + route
        parts = urllib.parse.urlsplit(url)
        if parts.hostname not in ('localhost', '127.0.0.1') or parts.port != 23119:
            raise RuntimeError('Refusing to send Zotero credentials outside the local API')
        h = {'Zotero-API-Version': '3', 'Zotero-Server-ID': self.server}
        if method != 'GET':
            if not self.auth:
                raise RuntimeError('Run authorize and approve the Zotero dialog first')
            h['Zotero-API-Key'] = self.auth['key']
        if data is not None and not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
            h['Content-Type'] = 'application/json'
        h.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
                if raw:
                    return body.decode()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            message = e.read().decode(errors='replace')
            raise RuntimeError(f'Zotero {method} {parts.path}: {e.code} {message}') from None

    def all(self, kind):
        # Explicit pagination also handles older servers with default limits.
        out, start = [], 0
        while True:
            page = self.request(f'{PREFIX}/{kind}?limit=100&start={start}')
            out.extend(page)
            if len(page) < 100:
                return out
            start += len(page)

    def create(self, kind, data):
        result = self.request(f'{PREFIX}/{kind}', 'POST', [data],
                              {'Zotero-Write-Token': uuid.uuid4().hex})
        if result.get('failed'):
            raise RuntimeError(json.dumps(result['failed'], ensure_ascii=False))
        key = result.get('success', {}).get('0')
        if not key:
            key = result.get('successful', {}).get('0', {}).get('key')
        if not key:
            raise RuntimeError('No created key returned')
        return key

    def patch(self, kind, key, patch):
        obj = self.request(f'{PREFIX}/{kind}/{key}')
        def equivalent(key, value):
            current = obj['data'].get(key)
            if key == 'collections':
                return set(current or []) == set(value)
            return current == value
        changed = {k: v for k, v in patch.items() if not equivalent(k, v)}
        if changed:
            self.request(f'{PREFIX}/{kind}/{key}', 'PATCH', changed,
                         {'If-Unmodified-Since-Version': str(obj['version'])})
        return bool(changed)

    def file_path(self, key):
        try:
            url = self.request(f'{PREFIX}/items/{key}/file/view/url', raw=True).strip()
            if url.startswith('file://'):
                return Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path))
        except RuntimeError as e:
            if ': 404 ' not in str(e):
                raise
        return None

    def upload(self, key, path, old_md5=None):
        route = f'{PREFIX}/items/{key}/file'
        match = {'If-Match': old_md5} if old_md5 else {'If-None-Match': '*'}
        params = {'md5': digest(path, 'md5'), 'filename': path.name,
                  'filesize': path.stat().st_size, 'mtime': int(path.stat().st_mtime * 1000)}
        headers = {'Content-Type': 'application/x-www-form-urlencoded', **match}
        result = self.request(route, 'POST', urllib.parse.urlencode(params).encode(), headers)
        if result.get('exists'):
            return
        payload = result['prefix'].encode() + path.read_bytes() + result['suffix'].encode()
        self.request(result['url'], 'POST', payload, {'Content-Type': result['contentType']}, raw=True)
        self.request(route, 'POST', urllib.parse.urlencode({'upload': result['uploadKey']}).encode(), headers)

    def collections(self, paths):
        existing = self.all('collections')
        mapping = {}
        for path in sorted(paths, key=lambda s: (s.count('/'), s)):
            parent, _, name = path.rpartition('/')
            parent_key = mapping.get(parent, False)
            matches = [x for x in existing if x['data']['name'] == name
                       and x['data'].get('parentCollection', False) == parent_key]
            if len(matches) > 1:
                raise RuntimeError(f'Ambiguous collection: {path}')
            if matches:
                key = matches[0]['key']
            else:
                key = self.create('collections', {'name': name, 'parentCollection': parent_key})
                existing.append({'key': key, 'data': {'name': name, 'parentCollection': parent_key}})
            mapping[path] = key
        save(self.private / 'collection-map.json', mapping)
        return mapping


def normalize_title(value):
    return re.sub(r'[^\w]', '', value.casefold())


def arxiv_id(value):
    match = re.search(r'(?:arxiv[.:/]|arxiv\.org/(?:abs|pdf)/)\s*(\d{4}\.\d{4,5})', value, re.I)
    return match.group(1) if match else None


def match_item(data, items):
    doi = data.get('DOI', '').lower().strip()
    arxiv = arxiv_id(data.get('url', '') + ' ' + data.get('DOI', '') + ' ' + data.get('extra', ''))
    title = normalize_title(data['title'])
    matches = []
    for obj in items:
        d = obj['data']
        if d['itemType'] in ('attachment', 'note', 'annotation'):
            continue
        other_arxiv = arxiv_id(d.get('url', '') + ' ' + d.get('DOI', '') + ' ' + d.get('extra', ''))
        if ((doi and doi == d.get('DOI', '').lower().strip()) or
            (arxiv and arxiv == other_arxiv) or title == normalize_title(d.get('title', ''))):
            matches.append(obj['key'])
    if len(matches) > 1:
        raise RuntimeError(f'Multiple candidates for {data["title"]}: {matches}')
    return matches[0] if matches else None


def apply(z, manifest):
    paths = set()
    for path in manifest['collections']:
        parts = path.split('/')
        paths.update('/'.join(parts[:n]) for n in range(1, len(parts) + 1))
    cmap = z.collections(paths)
    receipt_path = z.private / 'import-history.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    items = z.all('items')
    for entry in manifest.get('updates', []):
        obj = z.request(f'{PREFIX}/items/{entry["key"]}')
        patch = dict(entry.get('fields', {}))
        if entry.get('collections'):
            patch['collections'] = sorted(set(obj['data'].get('collections', []) +
                                              [cmap[x] for x in entry['collections']]))
        z.patch('items', entry['key'], patch)
    for entry in manifest.get('imports', []):
        data = dict(entry['data'])
        identity = entry['id']
        prior = receipts.get(identity, {})
        key = prior.get('itemKey') or entry.get('existingKey') or match_item(data, items)
        if not key:
            data['collections'] = [cmap[x] for x in entry['collections']]
            key = z.create('items', data)
            items.append(z.request(f'{PREFIX}/items/{key}'))
        else:
            obj = z.request(f'{PREFIX}/items/{key}')
            cols = sorted(set(obj['data'].get('collections', []) + [cmap[x] for x in entry['collections']]))
            z.patch('items', key, {'collections': cols})
        receipt = {'itemKey': key, **{k: v for k, v in prior.items() if k != 'itemKey'}}
        receipts[identity] = receipt
        save(receipt_path, receipts)
        if entry.get('attachExisting'):
            child = z.request(f'{PREFIX}/items/{entry["attachExisting"]}')
            parent = z.request(f'{PREFIX}/items/{key}')
            z.patch('items', key, {'collections': sorted(set(parent['data'].get('collections', []) + child['data'].get('collections', [])))})
            z.patch('items', child['key'], {'parentItem': key, 'collections': []})
            receipt['attachmentKey'] = child['key']
            save(receipt_path, receipts)
        if entry.get('file'):
            path = Path(entry['file'])
            if not path.is_absolute():
                path = z.paper / path
            if not path.exists() and receipt.get('verified'):
                stored = z.file_path(receipt['attachmentKey'])
                if not stored or not stored.is_file() or digest(stored) != receipt['sha256']:
                    raise RuntimeError(f'Archived source needs recovery: {identity}')
                print(identity, 'already imported; source archived and attachment verified')
                continue
            sha = digest(path)
            child = receipt.get('attachmentKey')
            if not child:
                children = z.request(f'{PREFIX}/items/{key}/children')
                for a in children:
                    if a['data']['itemType'] != 'attachment':
                        continue
                    p = z.file_path(a['key'])
                    if p and p.exists() and digest(p) == sha:
                        child = a['key']
                        break
            if not child:
                child = z.create('items', {'itemType': 'attachment', 'parentItem': key,
                                           'linkMode': 'imported_file', 'title': entry.get('attachmentTitle', '全文'),
                                           'contentType': 'application/pdf', 'filename': path.name,
                                           'tags': [], 'relations': {}})
            receipt.update(attachmentKey=child, source=str(path), sha256=sha)
            save(receipt_path, receipts)
            present = z.file_path(child)
            if not present or not present.exists():
                z.upload(child, path)
                present = z.file_path(child)
            if not present or not present.exists() or digest(present) != sha:
                raise RuntimeError(f'Attachment verification failed: {identity}')
            receipt['verified'] = True
            save(receipt_path, receipts)
        print(identity, key, 'verified')
    # Archive existing collections only after all imports have succeeded.
    for key in manifest.get('archiveRoots', []):
        z.patch('collections', key, {'parentCollection': cmap['99_旧分类归档']})
    refresh(z)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refresh(z):
    items, cols = z.all('items'), z.all('collections')
    save(z.private / 'zotero-items.json', items)
    save(z.private / 'zotero-collections.json', cols)
    bykey = {x['key']: x for x in items}
    colmap = {x['key']: x['data'] for x in cols}
    def colpath(key):
        c = colmap[key]
        return (colpath(c['parentCollection']) + '/' if c.get('parentCollection') else '') + c['name']
    rows, missing, unresolved, links = [], [], [], {}
    for obj in items:
        d, key = obj['data'], obj['key']
        if d['itemType'] in ('attachment', 'note', 'annotation'):
            if d['itemType'] == 'attachment' and not d.get('parentItem'):
                unresolved.append({'key': key, 'title': d.get('title', ''), 'reason': '独立附件'})
            continue
        pdfs = [a for a in items if a['data'].get('parentItem') == key and
                a['data'].get('contentType') == 'application/pdf']
        count = 0
        for a in pdfs:
            p = z.file_path(a['key'])
            if p and p.is_file():
                count += 1
                year = re.search(r'\b(?:19|20)\d{2}\b', d.get('date', ''))
                title = re.sub(r'[\x00-\x1f/\\:*?"<>|]', '-', d.get('title', 'paper')).strip()[:100]
                name = f'{year[0] if year else "undated"} - {title} [{a["key"]}].pdf'
                rel = f'library/{key}/{name}'
                link = z.paper / rel
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() and link.resolve() != p.resolve():
                    link.unlink()
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(p)
                if not link.is_symlink():
                    raise RuntimeError(f'User file occupies generated link: {link}')
                links[rel] = str(p)
            else:
                missing.append({'itemKey': key, 'attachmentKey': a['key'], 'title': d.get('title', ''),
                                'url': d.get('url', ''), 'status': '附件文件本地不可用'})
        if not pdfs:
            missing.append({'itemKey': key, 'attachmentKey': '', 'title': d.get('title', ''),
                            'url': d.get('url', ''), 'status': '尚无 PDF 附件'})
        if not (d.get('DOI') or d.get('url')):
            unresolved.append({'key': key, 'title': d.get('title', ''), 'reason': '缺少公开来源标识'})
        rows.append({'itemKey': key, 'title': d.get('title', ''), 'date': d.get('date', ''),
                     'DOI': d.get('DOI', ''), 'url': d.get('url', ''),
                     'collections': ' | '.join(colpath(k) for k in d.get('collections', [])),
                     'pdfRecords': len(pdfs), 'localPDFs': count,
                     'zotero': f'zotero://select/library/items/{key}'})
    # Remove only links owned by the previous run, never original user files.
    owned_path = z.private / 'generated-links.json'
    old = json.loads(owned_path.read_text()) if owned_path.exists() else {}
    for rel, target in old.items():
        p = z.paper / rel
        if rel not in links and p.is_symlink() and str(p.readlink()) == target:
            p.unlink()
    save(owned_path, links)
    write_csv(z.paper / 'catalog/papers.csv', ['itemKey','title','date','DOI','url','collections','pdfRecords','localPDFs','zotero'], rows)
    write_csv(z.paper / 'catalog/missing-fulltext.csv', ['itemKey','attachmentKey','title','url','status'], missing)
    write_csv(z.paper / 'catalog/unresolved.csv', ['key','title','reason'], unresolved)
    print(f'Catalog: {len(rows)} records, {len(links)} local PDFs, {len(missing)} fulltext tasks, {len(unresolved)} metadata tasks')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['authorize', 'refresh', 'apply'])
    p.add_argument('--paper', type=Path, default=REPO.parent / 'paper')
    p.add_argument('--manifest', type=Path)
    args = p.parse_args()
    z = Zotero(args.paper.resolve())
    if args.command == 'authorize':
        req = urllib.request.Request(API + '/local/authorize', data=json.dumps({'appName':'Knowledge Tree 文献维护'}).encode(),
                                     headers={'Content-Type':'application/json','Zotero-Server-ID':z.server}, method='POST')
        with urllib.request.urlopen(req, timeout=300) as r:
            auth = json.load(r)
        z.private.mkdir(parents=True, exist_ok=True)
        fd = os.open(z.auth_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump({'serverID':z.server, **auth}, f)
        print('Authorized. Credentials stored privately.')
    elif args.command == 'apply':
        if not args.manifest:
            p.error('apply requires --manifest with reviewed changes')
        apply(z, json.loads(args.manifest.read_text()))
    else:
        refresh(z)


if __name__ == '__main__':
    main()
