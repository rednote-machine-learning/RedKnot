"""Offline LongBench v1 Full/S1.6 inputs. No model, network, or tokenizer is used.

prepare_data(assets_dir, cache) returns the native data directory. Its sibling
sources/ contains the original contexts and templates. Python >=3.9, POSIX.
"""
import array
import collections
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zipfile


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(x):
    return sha(canonical(x))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def render_prompt(rec, sources, prompts):
    """Return user text and fully rendered checkpoint prompt from the real source order."""
    ds = rec['dataset']
    template = prompts['templates'][ds]
    ids = rec['input_scope_document_ids']
    target = rec['source_id']
    check(ids.count(target) == 1, 'Original context missing or duplicated')
    scope = rec['composition_scope']
    original = template.format(context=sources[target], input=rec['question'])
    if scope == 'official_unchanged':
        check(ids == [target], 'Unexpected backgrounds on unmodified official input')
        content = original
    elif scope in ('additional_same_dataset_passages','additional_native_demonstration_contexts_before_original_examples'):
        if scope.startswith('additional_native'):
            check(ids[-1] == target, 'Original example must be last')
        content = template.format(context=prompts['context_separator'].join(sources[i] for i in ids), input=rec['question'])
    elif scope == 'complete_official_target_after_separate_background':
        check(ids[-1] == target, 'Target must follow all background documents')
        zh = ds in prompts['chinese_datasets']
        layout = prompts['background_layout']['zh' if zh else 'en']
        backgrounds = ''.join(layout['item'].format(i=n, context=sources[s]) for n,s in enumerate(ids[:-1],1))
        content = layout['intro'] + backgrounds + layout['original'] + original
    else:
        raise ValueError(f'Unknown composition scope: {scope}')
    return content, prompts['chat_prefix'] + content + prompts['chat_suffix']


def decode_pool(payload):
    out = []
    view = memoryview(payload)
    pos = 0
    while pos < len(view):
        n = struct.unpack_from('<I',view,pos)[0]
        pos += 4
        check(n > 0 and pos + n*4 <= len(view), 'Invalid frozen token block length')
        out.append(view[pos:pos+n*4])
        pos += n*4
    return out


def restore_ids(pool, order):
    packed = array.array('I')
    check(packed.itemsize == 4, 'This Python must have 32-bit unsigned int arrays')
    for block in order:
        check(isinstance(block,int) and 0<=block<len(pool),'Invalid token block index')
        packed.frombytes(pool[block])
    if sys.byteorder != 'little':
        packed.byteswap()
    return packed.tolist()


def records(bundle, deep=False):
    pool = decode_pool(bundle.read('tokens.bin'))
    for line in bundle.read('recipes.jsonl').splitlines():
        recipe = json.loads(line)
        record = recipe['record']
        ids = restore_ids(pool, recipe['blocks'])
        check(len(ids)==record['input_tokens'] and digest(ids)==record['input_ids_sha256'],
              f"Token mismatch for case {record['index']}")
        record['input_ids'] = ids
        if deep:
            for span in record['source_documents']:
                actual = ids[span['input_start']:span['input_end']]
                check(digest(actual)==span['token_ids_sha256'], 'Document token hash mismatch')
        yield recipe['index'],record


def paths(assets):
    pkg = Path(assets) / 'frozen'
    manifest = pkg / 'cohort.json'
    check(file_sha(manifest)==COHORT_SHA256,'Frozen cohort manifest differs')
    return pkg,read(manifest)


def validate_package(pkg, m):
    for name,expected in m['assets'].items():
        p = pkg / name
        check(p.is_file() and not p.is_symlink() and file_sha(p)==expected, f'Frozen asset changed: {name}')


def write_json(path, obj):
    Path(path).write_bytes(canonical(obj))


def materialize(root, pkg, manifest):
    data = root/'data'
    for sub in ['index','groups','records']:
        (data/sub).mkdir(parents=True,exist_ok=True)
    sources = root/'sources'
    sources.mkdir()
    shutil.copyfile(pkg/'prompts.json', sources/'prompts.json')
    bindings = {}
    for ds in manifest['datasets']:
        with zipfile.ZipFile(pkg/f'{ds}.zip') as z:
            check(z.namelist()==['sources.jsonl','tokens.bin','recipes.jsonl'],'Unexpected shard members')
            (sources/f'{ds}.jsonl').write_bytes(z.read('sources.jsonl'))
            for idx,rec in records(z):
                i=rec['index']
                path=f'records/{i}.json.gz'
                raw=gzip.compress(canonical(rec),compresslevel=1,mtime=0)
                (data/path).write_bytes(raw)
                ptr = dict(index=i,path=path,file_sha256=sha(raw))
                write_json(data/f'index/{i}.json',dict(idx,**ptr))
                bindings[i]=ptr
    for group in read(pkg/'groups.json'):
        g=dict(group,cases=[bindings[i] for i in group['indices']])
        write_json(data/'groups'/f"{group['group_id']}.json",g)
    write_json(data/'complete.json',manifest['complete'])
    all_files = sorted(p for p in root.rglob('*') if p.is_file())
    write_json(root/'cache-files.json',{p.relative_to(root).as_posix():file_sha(p) for p in all_files})


def check_cache(root):
    root=Path(root)
    inventory=read(root/'cache-files.json')
    found={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(found==set(inventory)|{'cache-files.json'},'Cache file inventory differs')
    for rel,pin in inventory.items():
        p=root/rel
        check(not p.is_symlink() and p.is_file() and file_sha(p)==pin, f'Cached file changed: {rel}')


def prepare_data(assets_dir, cache):
    """Verify shipped shards, then atomically materialize ordinary native B4 data."""
    pkg,m=paths(assets_dir)
    validate_package(pkg,m)
    cache=Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    name='longbench21-'+COHORT_SHA256[:16]
    ready=cache/name
    lock_fd=os.open(cache/f'.{name}.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(lock_fd,'r+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if ready.exists():
            check_cache(ready)
        else:
            with tempfile.TemporaryDirectory(dir=cache,prefix='.'+name+'-') as tmp:
                root=Path(tmp)/'release'
                materialize(root,pkg,m)
                root.rename(ready)
    return ready/'data'


def verify_dataset(data):
    """Validate materialized bytes, original logical IDs, flags, dimensions and B4 plan."""
    data=Path(data)
    check_cache(data.parent)
    indices = {r['index']:r for p in (data/'index').glob('*.json') for r in [read(p)]}
    groups = sorted((data/'groups').glob('*.json'))
    counts=collections.Counter()
    logical=[]
    companions=[]
    for p in groups:
        g=read(p)
        ids,targets=g['indices'],g['target_indices']
        check(len(ids)==len(set(ids))==4 and len(set(targets))==len(targets) and set(targets)<=set(ids),'Invalid B4 membership')
        check(g['group_id']==p.stem and g['native_chunk_counts']==[indices[i]['frames'] for i in ids] and len(set(g['native_chunk_counts']))==1,'Invalid B4 frame counts')
        check(g['cases']==[{k:indices[i][k] for k in ('index','path','file_sha256')} for i in ids],'B4 case binding differs')
        counts.update(targets)
        companions.extend(i for i in ids if i not in targets)
        logical.append([g['group_id'],ids,targets,g['native_chunk_counts']])
    check(len(groups)==527 and len(companions)==8 and set(counts)==set(indices) and set(counts.values())=={1},'Wrong target/companion accounting')
    check(digest(logical)==GROUPS_SHA256,'B4 plan differs')
    input_digest=digest([[i,indices[i]['source_id'],indices[i]['input_ids_sha256'],indices[i]['rendered_prompt_sha256']] for i in sorted(indices)])
    check(input_digest==INPUTS_SHA256,'Original input hashes differ')
    by=collections.defaultdict(list)
    for r in indices.values():
        by[r['dataset']].append(r)
    by_ds={}
    for ds,rows in by.items():
        check(len(rows)==100 and len({(r['source_id'],r['source_index']) for r in rows})==100,'Wrong per-subset target inventory')
        by_ds[ds]={'targets':len(rows),'input_tokens':sum(x['input_tokens'] for x in rows),'length_bands':dict(collections.Counter(x['length_band'] for x in rows)),
                   'composition_modes':dict(collections.Counter(x['composition_mode'] for x in rows))}
    values=indices.values()
    return dict(target_count=len(indices),group_count=len(groups),batch_size=4,num_companion_slots=len(companions),companion_indices=companions,
                by_dataset=by_ds,dataset_counts={k:v['targets'] for k,v in by_ds.items()},total_input_tokens=sum(x['input_tokens'] for x in values),
                length_bands=dict(collections.Counter(x['length_band'] for x in values)),composition_modes=dict(collections.Counter(x['composition_mode'] for x in values)),
                input_tokens_range=[min(x['input_tokens'] for x in values),max(x['input_tokens'] for x in values)],
                logical_sha256=input_digest,group_membership_sha256=GROUPS_SHA256)
# GENERATED ORIGINAL DATA PINS
COHORT_SHA256 = "89307f14cb2b82541720d5f5f573775ca49a4070415800f27180f846a0cabf30"
GROUPS_SHA256 = "bcb1559ffde66ab95ec6b7b13d7889b90bce272cae6b4b53a71f6dfb99a0cebe"
INPUTS_SHA256 = "45977c584f678bda370f176ae35e7365b0f2ec0cc2e3f7df993dca5df419bd1c"
