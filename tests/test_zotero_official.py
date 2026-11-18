"""Official editions: reject unsafe sources and recover interrupted uploads."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/papers'))
import official
from test_zotero_workflow import FakeZotero


class OfficialEditionTests(unittest.TestCase):
    def entry(self):
        return dict(itemKey='PARENT', file='source.pdf', sourceUrl='https://publisher.example/paper',
                    version='published', sha256='reviewed', reviewed=True, evidence='Title and publisher imprint checked')

    def test_rejects_arxiv_signed_links_html_and_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for url in ['https://arxiv.org/pdf/1234.56789', 'https://publisher.example/pdf?X-Amz-Signature=secret']:
                with self.assertRaises(ValueError):
                    official.validate_entry({**self.entry(), 'sourceUrl':url}, root)
            p = root/'source.pdf'
            p.write_text('<html>login required</html>')
            with self.assertRaisesRegex(ValueError,'Expected a PDF'):
                official.inspect_pdf(p)
            with patch.object(official,'inspect_pdf',return_value={'sha256':'different','arxivStamp':False}):
                with self.assertRaisesRegex(ValueError,'changed after review'):
                    official.validate_entry(self.entry(),root)
            with patch.object(official,'inspect_pdf',return_value={'sha256':'reviewed','arxivStamp':True}):
                with self.assertRaisesRegex(ValueError,'arXiv version stamp'):
                    official.validate_entry(self.entry(),root)

    def test_interrupted_upload_reuses_attachment_and_verifies_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source.pdf'; source.write_bytes(b'%PDF-fixture')
            fake=FakeZotero(root)
            fake.objects['PARENT']={'key':'PARENT','data':{'itemType':'conferencePaper'}}
            entry={**self.entry(),'sha256':official.digest(source)}
            info={'sha256':entry['sha256'],'arxivStamp':False,'pages':1}
            with patch.object(official,'inspect_pdf',return_value=info),patch.object(official,'refresh'),patch.object(official,'checklist'):
                with self.assertRaisesRegex(RuntimeError,'connection loss'):
                    official.import_editions(fake,{'editions':[entry]})
                official.import_editions(fake,{'editions':[entry]})
                official.import_editions(fake,{'editions':[entry]})
            self.assertEqual(len(fake.created),1)
            receipt=next(iter(json.loads((fake.private/'official-imports.json').read_text()).values()))
            self.assertTrue(receipt['verified'])
            self.assertEqual(receipt['storedPath'],str(source))

    def test_missing_file_and_supplement_do_not_complete_main_paper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);private=root/'maintenance';private.mkdir()
            pdf=root/'supplement.pdf';pdf.write_bytes(b'%PDF-fixture')
            official.save(private/'zotero-items.json',[{'key':k,'data':{'itemType':'journalArticle','title':k}} for k in ['MAIN','MISSING']])
            official.save(private/'official-imports.json',{
                'a':dict(itemKey='MAIN',attachmentKey='A',verified=True,version='supplement',storedPath=str(pdf),sha256=official.digest(pdf)),
                'b':dict(itemKey='MISSING',attachmentKey='B',verified=True,version='published',storedPath=str(root/'gone.pdf'),sha256='x')})
            official.checklist(root)
            with (root/'catalog/missing-official-fulltext.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))),2)

    def test_zotero_renamed_attachment_is_found_after_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);private=root/'maintenance';private.mkdir()
            pdf=root/'renamed.pdf';pdf.write_bytes(b'%PDF-fixture')
            official.save(private/'zotero-items.json',[{'key':'P','data':{'itemType':'report','title':'Report'}}])
            official.save(private/'official-imports.json',{'r':dict(itemKey='P',attachmentKey='A',verified=True,
                version='technical-report',storedPath=str(root/'old.pdf'),sha256=official.digest(pdf))})
            official.save(private/'generated-links.json',{'library/P/2023 - Report [A].pdf':str(pdf)})
            official.checklist(root)
            with (root/'catalog/missing-official-fulltext.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(list(csv.DictReader(f)),[])


if __name__=='__main__':unittest.main()
