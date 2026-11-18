"""Regression checks for imports: identity ambiguity and interrupted upload recovery."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('kt_zotero', Path(__file__).resolve().parents[1] / 'scripts/papers/zotero.py')
z = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(z)


class FakeZotero:
    def __init__(self, paper):
        self.paper = paper
        self.private = paper / 'maintenance'
        self.objects = {}
        self.created = []
        self.stored = {}
        self.fail_upload = True

    def collections(self, paths):
        return {p:'COLLECTION' for p in paths}

    def all(self, kind):
        return list(self.objects.values())

    def create(self, kind, data):
        key = f'KEY{len(self.created)}'
        self.created.append(key)
        self.objects[key] = {'key':key,'data':copy.deepcopy(data)}
        return key

    def request(self, route):
        if route.endswith('/children'):
            key = route.split('/')[-2]
            return [x for x in self.objects.values() if x['data'].get('parentItem') == key]
        return self.objects[route.rsplit('/',1)[-1]]

    def patch(self, kind, key, patch):
        self.objects[key]['data'].update(patch)

    def file_path(self, key):
        return self.stored.get(key)

    def upload(self, key, path):
        if self.fail_upload:
            self.fail_upload = False
            raise RuntimeError('simulated connection loss')
        self.stored[key] = path


class WorkflowTests(unittest.TestCase):
    def test_versioned_arxiv_matches_and_ambiguous_title_stops(self):
        data = {'title':'A paper','url':'https://arxiv.org/abs/1707.06347v2'}
        one = {'key':'A','data':{'itemType':'preprint','title':'A paper','DOI':'10.48550/arXiv.1707.06347'}}
        self.assertEqual(z.match_item(data,[one]),'A')
        with self.assertRaisesRegex(RuntimeError,'Multiple candidates'):
            z.match_item(data,[one,{**one,'key':'B'}])

    def test_collection_order_alone_does_not_write(self):
        api = object.__new__(z.Zotero)
        calls = []
        def request(route, *args):
            calls.append((route,args))
            return {'data':{'collections':['B','A']},'version':1}
        api.request = request
        self.assertFalse(api.patch('items','PAPER',{'collections':['A','B']}))
        self.assertEqual(len(calls),1)

    def test_resume_failed_upload_reuses_parent_and_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            pdf = paper / 'source.pdf'
            pdf.write_bytes(b'%PDF-fixture')
            fake = FakeZotero(paper)
            plan = {'collections':['Topic'], 'imports':[{'id':'paper-1','data':{'itemType':'preprint','title':'Paper'},'collections':['Topic'],'file':'source.pdf'}]}
            old_refresh = z.refresh
            z.refresh = lambda _:None
            try:
                with self.assertRaisesRegex(RuntimeError,'connection loss'):
                    z.apply(fake,plan)
                self.assertEqual(len(fake.created),2)
                z.apply(fake,plan)
                z.apply(fake,plan)
                self.assertEqual(len(fake.created),2)
                record = json.loads((paper/'maintenance/import-history.json').read_text())['paper-1']
                self.assertTrue(record['verified'])
                self.assertEqual(record['sha256'],z.digest(pdf))
                self.assertTrue(pdf.exists())
            finally:
                z.refresh = old_refresh


if __name__ == '__main__':
    unittest.main()
