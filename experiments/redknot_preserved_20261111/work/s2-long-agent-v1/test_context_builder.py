"""CPU input/provenance contracts. Fixtures are never benchmark input evidence."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace as NS
import unittest

import context_builder as cb


class BytesTokenizer:
    def encode(self,text,add_special_tokens=False):
        assert add_special_tokens is False
        return NS(ids=list(text.encode()))


class FixtureCodec:
    def render(self,payload):
        ids=[0];docs=[]
        for mi,m in enumerate(payload['messages']):
            c=m['content'];c=[dict(type='text',text=c)] if isinstance(c,str) else c
            for bi,b in enumerate(c):
                ids += [300];start=len(ids);ids+=list(b['text'].encode());end=len(ids)
                docs.append(dict(input_start=start,input_end=end,tokens=end-start,
                    document_id=cb.digest(ids[start:end]),origin=dict(message_index=mi,content_block_index=bi,
                    source_kind='user_text_block')))
        return dict(input_ids=ids,input_ids_sha256=cb.digest(ids),documents=docs)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def task_selection(self):
        rows=[dict(instance_id=i,repo=cb.REPO,base_commit=cb.BASE,problem_statement='Original issue '+i,
                   environment_setup_commit='a'*40,version='4.1') for i in cb.TASKS]
        task=self.root/'tasks.jsonl';task.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        v=cb.seal(dict(schema='s2-long-same-base-selection-v1',repo=cb.REPO,base_commit=cb.BASE,
            selected_instance_ids=list(cb.TASKS),tasks_file='tasks.jsonl',tasks_file_sha256=cb.file_sha(task),
            problem_statement_utf8_sha256={r['instance_id']:cb.sha(r['problem_statement'].encode()) for r in rows}))
        path=self.root/'selection.json';cb.write_json(path,v);return path,rows

    def archive(self,entries=None):
        archive=self.root/'source.tar.gz'
        entries=entries or {'django/db/migrations/optimizer.py':b'# unchanged\r\n'+b'print(1)\n'*70,
                            'django/db/backends/sqlite3/schema.py':b'# other source\n'+b'pass\n'*140,
                            'tests/test_schema.py':b'not selected', 'README.rst':b'not selected'}
        with tarfile.open(archive,'w:gz') as tar:
            for name,raw in entries.items():
                m=tarfile.TarInfo('django-'+cb.BASE+'/'+name);m.size=len(raw);m.mode=0o644
                tar.addfile(m,io.BytesIO(raw))
        receipt=dict(repo=cb.REPO,base_commit=cb.BASE,source_url=f'https://github.com/{cb.REPO}/archive/{cb.BASE}.tar.gz',
                     archive_sha256=cb.file_sha(archive),archive_bytes=archive.stat().st_size)
        p=self.root/'source-receipt.json';cb.write_json(p,receipt);return archive,p

    def bundle(self):
        selection,rows=self.task_selection();archive,receipt=self.archive()
        manifest=cb.build(archive,receipt,selection,BytesTokenizer(),self.root/'bundle',
                          minimum=500,target=1200,maximum=2200,min_block_tokens=256)
        return self.root/'bundle/context-manifest.json',selection,rows,manifest

    def test_whole_files_preserve_line_endings_and_exclude_test_files(self):
        path,selection,rows,manifest=self.bundle();blocks=cb.load_content_blocks(path)
        self.assertEqual(manifest['task_ids'],list(cb.TASKS));self.assertGreaterEqual(len(blocks),1)
        self.assertIn('\r\n',blocks[0]['text']);self.assertFalse(any('tests/' in r['path'] for r in manifest['files']))
        self.assertEqual(manifest['files_truncated'],0);self.assertTrue(manifest['actual_chat_render_pending'])
        self.assertEqual(manifest['shared_block_content_tokens'],sum(len(b['text'].encode()) for b in blocks))

    def test_context_tampering_and_actual_workspace_mismatch_fail(self):
        path,selection,rows,m=self.bundle();workspace=self.root/'repo'
        blocks=cb.load_content_blocks(path)
        for item,b in zip(m['files'],blocks):
            p=workspace/item['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b['text'].split('\n\n',1)[1].encode())
        self.assertTrue(cb.verify_workspace(path,workspace)['exact_raw_bytes'])
        (workspace/m['files'][0]['path']).write_text('modified')
        with self.assertRaisesRegex(ValueError,'workspace source differs'):cb.verify_workspace(path,workspace)
        (path.parent/m['files'][0]['block_file']).write_text('modified')
        with self.assertRaisesRegex(ValueError,'context block changed'):cb.load_content_blocks(path)

    def test_archive_scope_path_and_bytes_are_checked_before_build(self):
        archive,p=self.archive({'../escape.py':b'bad'});receipt=json.loads(p.read_text())
        with self.assertRaisesRegex(ValueError,'unsafe'):cb.archive_files(archive,receipt)
        receipt['archive_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'archive bytes'):cb.archive_files(archive,receipt)

    def test_padding_or_truncation_never_fills_insufficient_context(self):
        archive,p=self.archive();files,inventory=cb.archive_files(archive,json.loads(p.read_text()))
        with self.assertRaisesRegex(ValueError,'no padding'):cb.select_blocks(files,BytesTokenizer(),minimum=9999,target=9999,maximum=10000)
        blocks,omitted,n=cb.select_blocks(files,BytesTokenizer(),minimum=500,target=500,maximum=950)
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0]['bytes'],len(files[blocks[0]['path']][1].encode()))

    def test_real_request_audit_keeps_task_first_and_measures_nonprefix_shared_ids(self):
        manifest,selection,rows,m=self.bundle();blocks=cb.load_content_blocks(manifest)
        payloads={r['instance_id']:dict(messages=[dict(role='user',content=[dict(type='text',text=r['problem_statement']),*blocks])]) for r in rows}
        report=cb.audit_rendered_requests(payloads,FixtureCodec(),manifest,selection,min_total=1,max_total=10000)
        self.assertTrue(report['same_shared_document_token_ids'])
        self.assertTrue(report['task_specific_prefix_changed'])
        for r in report['actual_request_records']:
            self.assertEqual(r['shared_tokens_after_pair_lcp'],r['shared_eligible_tokens'])
        self.assertFalse(report['online_agent_executed'])
        bad=copy.deepcopy(payloads);bad[cb.TASKS[0]]['messages'][0]['content'].append(blocks[0])
        with self.assertRaisesRegex(ValueError,'exactly once'):cb.audit_rendered_requests(bad,FixtureCodec(),manifest,selection,min_total=1,max_total=10000)

    def test_actual_task_problem_cannot_be_replaced_or_moved_after_material(self):
        manifest,selection,rows,m=self.bundle();blocks=cb.load_content_blocks(manifest)
        payloads={r['instance_id']:dict(messages=[dict(role='user',content=[*blocks,dict(type='text',text=r['problem_statement'])])]) for r in rows}
        with self.assertRaisesRegex(ValueError,'must precede'):cb.audit_rendered_requests(payloads,FixtureCodec(),manifest,selection,min_total=1,max_total=10000)

    def test_selection_file_has_no_hidden_grader_fields(self):
        path,rows=self.task_selection();task=self.root/'tasks.jsonl';rows[0]['patch']='answer'
        task.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        v=json.loads(path.read_text());v['tasks_file_sha256']=cb.file_sha(task);v.pop('sha256');path.write_text(json.dumps(cb.seal(v)))
        with self.assertRaisesRegex(ValueError,'forbidden'):cb.validate_selection(path)

    def test_eligible_roots_reject_test_patch_and_unrelated_material(self):
        for path in ('tests/a.py','django/db/migrations/test_foo.py','django/db/migrations/solution.py',
                     'django/db/migrations/fix.patch','docs/releases/9.0.txt','grader_only/answer.json'):
            self.assertFalse(cb.eligible(path),path)

if __name__=='__main__':unittest.main()
