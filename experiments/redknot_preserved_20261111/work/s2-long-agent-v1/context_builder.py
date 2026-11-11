"""CPU-only, whole-file supplemental context for two same-base Verified tasks.

No task code, repository setup, tests or network commands are executed here.
The original issue and evaluator remain unchanged; adding this material is an
explicit experimental input extension, not the official default prompt.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

REPO = 'django/django'
BASE = '0ab58c120939093fea90822f376e1866fc714d1f'
TASKS = ('django__django-15268', 'django__django-15278')
TOKENIZER_SHA = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
ALLOWED_FIELDS = {'base_commit','created_at','environment_setup_commit','image',
                  'instance_id','problem_statement','repo','version'}
# Chosen from the original public issue topics (migration optimizer and SQLite
# nullable OneToOneField), before new model execution; no gold patch was read.
PRIORITY_PATHS = (
    'django/db/migrations/optimizer.py',
    'django/db/migrations/operations/models.py',
    'django/db/migrations/operations/fields.py',
    'django/db/backends/base/schema.py',
    'django/db/backends/sqlite3/schema.py',
    'django/db/models/fields/related.py',
    'docs/topics/migrations.txt',
    'docs/ref/migration-operations.txt',
    'docs/ref/models/fields.txt',
    'django/db/migrations/state.py',
    'django/db/migrations/autodetector.py',
    'django/db/models/fields/__init__.py',
)
EXTRA_ROOTS = ('django/db/migrations/', 'django/db/models/fields/',
               'django/db/backends/base/', 'django/db/backends/sqlite3/',
               'docs/topics/db/', 'docs/ref/models/')
EXTRA_SUFFIXES = {'.py','.txt','.rst','.md'}


def require(ok, message):
    if not ok: raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def seal(value): return dict(value,sha256=digest(value))
def sha(raw): return hashlib.sha256(raw).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read_sealed(path):
    value=json.loads(Path(path).read_text())
    require(value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'invalid manifest seal')
    return value

def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def freeze_selection(candidates, original_selection, output):
    source=read_sealed(original_selection)
    require(file_sha(candidates)==source['files']['agent_visible/candidates500.jsonl'],
            'agent-visible candidate population changed')
    rows=[json.loads(line) for line in Path(candidates).read_text().splitlines()]
    require(len(rows)==500 and len({r['instance_id'] for r in rows})==500,'Verified500 population differs')
    require(all(set(r)<=ALLOWED_FIELDS for r in rows),'forbidden agent-visible field')
    grouped=defaultdict(list)
    for row in rows:grouped[row['repo'],row['base_commit']].append(row)
    groups=[(key,value) for key,value in grouped.items() if len(value)>1]
    require(len(groups)==1 and groups[0][0]==(REPO,BASE)
            and set(r['instance_id'] for r in groups[0][1])==set(TASKS),
            'unique same-repository same-base selection no longer holds')
    picked=sorted(groups[0][1],key=lambda r:TASKS.index(r['instance_id']))
    require(len({r['environment_setup_commit'] for r in picked})==1,'environment setup commits differ')
    require(not set(TASKS)&set(source['selected_instance_ids']),'long cohort must be separately registered')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tasks_path=output/'tasks.jsonl'
    with tasks_path.open('x') as f:
        for row in picked:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    value=seal(dict(schema='s2-long-same-base-selection-v1',status='frozen_before_long_model_execution',
        dataset=source['dataset'],dataset_revision=source['revision'],population=500,
        population_file_sha256=file_sha(candidates),original20_selection_sha256=source['sha256'],
        selection_rule='all unique (repo,base_commit) groups with at least two rows; exactly one group; task IDs ascending',
        model_outcomes_used=False,gold_patch_accessed=False,test_patch_accessed=False,
        original20_member=False,new_cohort=True,selected_instance_ids=list(TASKS),repo=REPO,base_commit=BASE,
        environment_setup_commit=picked[0]['environment_setup_commit'],
        tasks_file='tasks.jsonl',tasks_file_sha256=file_sha(tasks_path),
        problem_statement_utf8_sha256={r['instance_id']:sha(r['problem_statement'].encode()) for r in picked},
        input_protocol='official_task_instruction_then_shared_whole_file_TextContent_blocks',
        official_default_prompt_unchanged=False,original_issue_unchanged=True,official_evaluator_unchanged=True,
        session_design=['A cold fresh workspace','B warm bank fresh workspace','A repeat fresh workspace'],
        independent_task_count=2,third_session_is_repeated_task=True,
        proposed_shared_content_tokens=dict(minimum=80000,target=82000,maximum=95000),
        actual_full_request_tokens_required=dict(minimum=95000,maximum=110000),
        source_policy=dict(priority_paths=list(PRIORITY_PATHS),extra_roots=list(EXTRA_ROOTS),
            order='priority paths then remaining eligible paths lexicographically',
            max_files=48,minimum_block_content_tokens=256,whole_files_only=True,
            raw_duplicate_files_omitted=True,no_padding=True,no_problem_or_answer_text_in_shared_blocks=True)))
    write_json(output/'selection-manifest.json',value)
    return value


def validate_selection(path):
    p=Path(path);v=read_sealed(p)
    require(v.get('schema')=='s2-long-same-base-selection-v1' and v.get('repo')==REPO
        and v.get('base_commit')==BASE and v.get('selected_instance_ids')==list(TASKS),'foreign long cohort')
    task_path=p.parent/v['tasks_file']
    require(file_sha(task_path)==v['tasks_file_sha256'],'long task rows changed')
    rows=[json.loads(x) for x in task_path.read_text().splitlines()]
    require([r['instance_id'] for r in rows]==list(TASKS) and all(set(r)<=ALLOWED_FIELDS for r in rows),
            'long tasks contain foreign/forbidden fields')
    for r in rows:
        require(r['repo']==REPO and r['base_commit']==BASE
            and sha(r['problem_statement'].encode())==v['problem_statement_utf8_sha256'][r['instance_id']],
            'original task statement or revision changed')
    return v,rows


def eligible(path):
    p=PurePosixPath(path)
    if any(part.lower() in ('tests','test','grader_only','.git','.github') for part in p.parts):return False
    if p.name.lower().startswith(('test_', 'test-', 'gold', 'solution')) or p.suffix in ('.patch','.diff'):return False
    return path in PRIORITY_PATHS or (path.startswith(EXTRA_ROOTS) and p.suffix in EXTRA_SUFFIXES)


def archive_files(archive, receipt):
    """Validate names without extracting links or executing repository code."""
    require(receipt.get('repo')==REPO and receipt.get('base_commit')==BASE,'archive scope differs')
    require(receipt.get('source_url') in (
        f'https://github.com/{REPO}/archive/{BASE}.tar.gz',
        f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'), 'unregistered upstream archive URL')
    require(file_sha(archive)==receipt.get('archive_sha256') and Path(archive).stat().st_size==receipt.get('archive_bytes'),
            'actual upstream archive bytes differ')
    root='django-'+BASE;inventory=[];files={};seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            path=PurePosixPath(m.name)
            require(not path.is_absolute() and '..' not in path.parts and path.parts and path.parts[0]==root,
                    'unsafe or foreign archive path')
            name='/'.join(path.parts[1:])
            if m.isdir():continue
            require(name and name not in seen,'duplicate archive member');seen.add(name)
            if not m.isfile():
                require(not eligible(name),'eligible source cannot be a symlink/hardlink/device')
                inventory.append(dict(path=name,type='nonregular',link=m.linkname));continue
            require(0<=m.size<=32*1024*1024,'unexpected source member size')
            f=tar.extractfile(m);require(f is not None,'missing regular source body');raw=f.read()
            require(len(raw)==m.size,'short source body')
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            row=dict(path=name,bytes=len(raw),sha256=sha(raw),git_blob_sha1=blob,mode=oct(m.mode))
            inventory.append(row)
            if eligible(name):
                try:text=raw.decode('utf-8')
                except UnicodeDecodeError:continue
                require(text.encode()==raw,'UTF-8 roundtrip altered source')
                files[name]=(row,text)
    require(files,'archive contains no eligible whole source files')
    return files,sorted(inventory,key=lambda x:x['path'])


def block_text(path, raw_text, blob):
    return (f'Repository source: {REPO}\nBase commit: {BASE}\nPath: {path}\n'
            f'Git blob SHA-1: {blob}\n\n'+raw_text)

def token_ids(tokenizer,text):return list(tokenizer.encode(text,add_special_tokens=False).ids)


def select_blocks(files,tokenizer,*,minimum=80000,target=82000,maximum=95000,max_files=48,min_block_tokens=256):
    require(0<minimum<=target<=maximum and 0<max_files<=48,'invalid shared context budget')
    order=[p for p in PRIORITY_PATHS if p in files]+sorted(set(files)-set(PRIORITY_PATHS))
    selected=[];omitted=[];total=0;raw_seen=set()
    for path in order:
        row,text=files[path]
        if row['sha256'] in raw_seen:
            omitted.append(dict(path=path,reason='duplicate_complete_raw_file'));continue
        block=block_text(path,text,row['git_blob_sha1']);ids=token_ids(tokenizer,block)
        if len(ids)<min_block_tokens:
            omitted.append(dict(path=path,reason='short_whole_file',tokens=len(ids)));continue
        if total+len(ids)>maximum:
            omitted.append(dict(path=path,reason='would_exceed_upper_bound_without_truncation',tokens=len(ids)));continue
        if len(selected)>=max_files:break
        raw_seen.add(row['sha256']);total+=len(ids)
        selected.append(dict(**row,text=block,block_sha256=sha(block.encode()),
                             block_content_tokens=len(ids),block_content_token_ids_sha256=digest(ids),
                             raw_file_content_tokens=len(token_ids(tokenizer,text))))
        if total>=target:break
    require(total>=minimum,f'insufficient whole-file context within {max_files} blocks: {total} content tokens; no padding permitted')
    return selected,omitted,total


def build(archive,source_receipt,selection,tokenizer,output,*,tokenizer_sha256=TOKENIZER_SHA,**budgets):
    selected_tasks,_=validate_selection(selection)
    receipt=json.loads(Path(source_receipt).read_text())
    if 'sha256' in receipt:require(receipt['sha256']==digest({k:v for k,v in receipt.items() if k!='sha256'}),'source receipt seal differs')
    files,inventory=archive_files(archive,receipt)
    blocks,omitted,total=select_blocks(files,tokenizer,**budgets)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'blocks').mkdir()
    entries=[]
    for i,b in enumerate(blocks):
        name=f'blocks/{i:03d}-{b["sha256"][:16]}.txt'
        (out/name).write_bytes(b['text'].encode())
        entries.append({k:v for k,v in b.items() if k!='text'}|dict(block_file=name,ordinal=i))
    manifest=seal(dict(schema='s2-long-whole-file-context-v1',status='content_ready_actual_sdk_render_pending',
        repo=REPO,base_commit=BASE,task_ids=list(TASKS),selection_sha256=selected_tasks['sha256'],
        source_receipt=receipt,source_receipt_file_sha256=file_sha(source_receipt),
        archive_sha256=file_sha(archive),archive_inventory_sha256=digest(inventory),
        archive_regular_files=sum('sha256' in row for row in inventory),
        tokenizer_json_sha256=tokenizer_sha256,add_special_tokens=False,
        count_interpretation='standalone TextContent token count; not complete chat-template request length',
        shared_block_content_tokens=total,raw_file_content_tokens=sum(b['raw_file_content_tokens'] for b in entries),
        block_count=len(entries),files=entries,omitted=omitted,
        whole_files=True,files_truncated=0,duplicate_raw_files=0,artificial_padding_tokens=0,
        source_code_executed=False,gold_or_test_patch_read=False,old_model_output_used=False,
        approximate_selected_tail_rows=len(entries)*128,
        actual_group_selected_rows_pending=True,actual_chat_render_pending=True,
        official_default_prompt_unchanged=False,original_issue_unchanged=True,official_evaluator_unchanged=True))
    write_json(out/'archive-inventory.json',seal(dict(schema='s2-long-archive-inventory-v1',files=inventory)))
    write_json(out/'context-manifest.json',manifest)
    return manifest


def load_content_blocks(manifest_path):
    path=Path(manifest_path);manifest=read_sealed(path)
    require(manifest.get('schema')=='s2-long-whole-file-context-v1' and manifest['repo']==REPO
        and manifest['base_commit']==BASE,'foreign context manifest')
    result=[]
    for item in manifest['files']:
        p=PurePosixPath(item['block_file'])
        require(not p.is_absolute() and '..' not in p.parts,'unsafe block reference')
        f=path.parent.joinpath(*p.parts)
        require(not f.is_symlink() and f.is_file() and file_sha(f)==item['block_sha256'],'context block changed')
        text=f.read_bytes().decode('utf-8');header,body=text.split('\n\n',1)
        require(text==block_text(item['path'],body,item['git_blob_sha1'])
            and sha(body.encode())==item['sha256'] and len(body.encode())==item['bytes'],
            'whole file body/header no longer matches source')
        result.append(dict(type='text',text=text))
    require(len(result)==manifest['block_count']<=48,'context block inventory differs')
    return result


def verify_workspace(manifest_path,repo_root):
    manifest=read_sealed(manifest_path);load_content_blocks(manifest_path);root=Path(repo_root).resolve()
    for item in manifest['files']:
        p=root/item['path']
        require(not p.is_symlink() and p.resolve().is_relative_to(root)
            and p.is_file() and file_sha(p)==item['sha256'],'workspace source differs: '+item['path'])
    return seal(dict(schema='s2-long-workspace-source-match-v1',context_sha256=manifest['sha256'],
        files_checked=manifest['block_count'],exact_raw_bytes=True,repo_root=str(root),
        task_code_executed=False,git_head_claim=False))


def audit_rendered_requests(payloads,codec,manifest_path,selection_path,*,min_total=95000,max_total=110000):
    """Two actual SDK initial task payloads, preserving all original blocks.

    codec must be the real pinned CheckpointCodec. The supplied messages are
    consumed without modifications. LCP is an ideal exact-token prefix bound,
    not a measured cache hit; document hashes are actual full-prompt spans.
    """
    manifest=read_sealed(manifest_path);blocks=load_content_blocks(manifest_path)
    selection,rows=validate_selection(selection_path)
    require(manifest['selection_sha256']==selection['sha256'],'context belongs to another selection')
    require(set(payloads)==set(TASKS),'two registered actual task payloads required')
    rendered=[];records=[];shared_docs=[]
    for row in rows:
        payload=payloads[row['instance_id']]
        occurrences={b['text']:[] for b in blocks}
        for mi,m in enumerate(payload['messages']):
            content=m.get('content')
            if isinstance(content,list):
                for bi,b in enumerate(content):
                    if b.get('type')=='text' and b.get('text') in occurrences:
                        require(m['role']=='user','supplemental source must be user text')
                        occurrences[b['text']].append((mi,bi))
        require(all(len(v)==1 for v in occurrences.values()),'each complete source must appear exactly once as its own TextContent')
        first=min(v[0] for v in occurrences.values())
        prefix=[]
        for mi,m in enumerate(payload['messages']):
            c=m.get('content')
            if isinstance(c,str) and mi<first[0]:prefix.append(c)
            elif isinstance(c,list):
                prefix += [b.get('text','') for bi,b in enumerate(c) if (mi,bi)<first]
        require(row['problem_statement'] in '\n'.join(prefix),'original issue must precede shared source unchanged')
        r=codec.render(payload);require(min_total<=len(r['input_ids'])<=max_total,'actual initial request outside declared total-token band')
        mapping={(d['origin'].get('message_index'),d['origin'].get('content_block_index')):d for d in r['documents']
                 if d['origin'].get('source_kind')=='user_text_block'}
        docs=[]
        for b in blocks:
            key=occurrences[b['text']][0]
            require(key in mapping,'real checkpoint mapping lost a shared block')
            d=mapping[key];require(d['tokens']>=256,'shared block lost eligible span')
            docs.append(dict(input_start=d['input_start'],input_end=d['input_end'],tokens=d['tokens'],
                             document_id=d['document_id']))
        rendered.append(r);shared_docs.append(docs)
        records.append(dict(instance_id=row['instance_id'],request_payload_sha256=digest(payload),
            input_tokens=len(r['input_ids']),input_ids_sha256=r['input_ids_sha256'],
            shared_eligible_tokens=sum(d['tokens'] for d in docs),shared_documents=docs,
            shared_eligible_fraction=sum(d['tokens'] for d in docs)/len(r['input_ids'])))
    left,right=rendered;lcp=0
    for a,b in zip(left['input_ids'],right['input_ids']):
        if a!=b:break
        lcp+=1
    require(lcp<min(len(left['input_ids']),len(right['input_ids'])),'task prefixes did not change')
    require([d['document_id'] for d in shared_docs[0]]==[d['document_id'] for d in shared_docs[1]],
            'actual document token IDs differ despite shared source text')
    for record,docs in zip(records,shared_docs):
        record['shared_tokens_after_pair_lcp']=sum(max(0,d['input_end']-max(lcp,d['input_start'])) for d in docs)
    return seal(dict(schema='s2-long-actual-sdk-input-pair-v1',context_sha256=manifest['sha256'],
        selection_sha256=selection['sha256'],actual_request_records=records,
        exact_common_prefix_tokens=lcp,prefix_value_is_opportunity_not_measured_hit=True,
        same_shared_document_token_ids=True,task_specific_prefix_changed=True,
        actual_group_selected_rows_must_be_admitted_separately=True,
        online_agent_executed=False,performance_claim=False))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('select');s.add_argument('--candidates',required=True);s.add_argument('--original-selection',required=True);s.add_argument('--output',required=True)
    b=sub.add_parser('build');b.add_argument('--archive',required=True);b.add_argument('--source-receipt',required=True);b.add_argument('--selection',required=True);b.add_argument('--tokenizer',required=True);b.add_argument('--tokenizers-dir');b.add_argument('--output',required=True)
    v=sub.add_parser('verify-workspace');v.add_argument('--manifest',required=True);v.add_argument('--repo-root',required=True)
    a=p.parse_args()
    if a.command=='select':value=freeze_selection(a.candidates,a.original_selection,a.output)
    elif a.command=='verify-workspace':value=verify_workspace(a.manifest,a.repo_root)
    else:
        require(file_sha(a.tokenizer)==TOKENIZER_SHA,'actual checkpoint tokenizer differs')
        if a.tokenizers_dir:sys.path.insert(0,str(Path(a.tokenizers_dir).resolve()))
        from tokenizers import Tokenizer
        value=build(a.archive,a.source_receipt,a.selection,Tokenizer.from_file(a.tokenizer),a.output)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
