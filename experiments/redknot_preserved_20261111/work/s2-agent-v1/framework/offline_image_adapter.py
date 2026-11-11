"""Content-verified offline source adapter for the unchanged official builder."""
from __future__ import annotations
import hashlib,json,os,pathlib,stat,tarfile

def sha_file(path):
    with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_path(path):
    path=pathlib.Path(path).resolve()
    if any(c.isspace() for c in str(path)) or ':' in str(path):
        raise ValueError('Definition source path must have no whitespace/colon')
    return path

def archive_provenance(archive,archive_sha,registry_manifest,registry_digest,base_image):
    archive=safe_path(archive);manifest=pathlib.Path(registry_manifest)
    if sha_file(archive)!=archive_sha:raise ValueError('Docker archive SHA mismatch')
    raw=manifest.read_bytes()
    if 'sha256:'+hashlib.sha256(raw).hexdigest()!=registry_digest:raise ValueError('Registry manifest digest mismatch')
    registry=json.loads(raw)
    if registry.get('schemaVersion')!=2 or 'config' not in registry or 'layers' not in registry:raise ValueError('Expected resolved single-platform image manifest')
    with tarfile.open(archive,'r:*') as tar:
        members={}
        for member in tar.getmembers():
            name=pathlib.PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.name in members:raise ValueError('Unsafe/duplicate archive member')
            if not (member.isfile() or member.isdir()):raise ValueError('Outer Docker archive must contain only regular files/directories')
            members[member.name]=member
        def read(name):
            if name not in members or not members[name].isfile():raise ValueError('Missing regular archive member: '+name)
            return tar.extractfile(members[name])
        with read('manifest.json') as f: saved=json.load(f)
        if not isinstance(saved,list) or len(saved)!=1:raise ValueError('Expected exactly one saved image')
        item=saved[0]
        expected=base_image.removeprefix('docker.io/')
        if expected not in [x.removeprefix('docker.io/') for x in item.get('RepoTags',[])]:raise ValueError('Saved image tag differs from selected official image')
        with read(item['Config']) as f:config_raw=f.read()
        if 'sha256:'+hashlib.sha256(config_raw).hexdigest()!=registry['config']['digest']:raise ValueError('Config blob differs from official registry manifest')
        config=json.loads(config_raw)
        if (config.get('os'),config.get('architecture'))!=('linux','amd64'):raise ValueError('Expected official linux/amd64 image')
        diffs=config.get('rootfs',{}).get('diff_ids',[])
        if len(diffs)!=len(item['Layers']) or len(diffs)!=len(registry['layers']):raise ValueError('Layer counts differ')
        layers=[]
        for index,(name,diff) in enumerate(zip(item['Layers'],diffs)):
            with read(name) as f:actual='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=diff:raise ValueError('Uncompressed Docker layer differs from config diff_id: '+name)
            layers.append({'index':index,'archive_member':name,'uncompressed_digest':actual,'registry_compressed_descriptor':registry['layers'][index]})
    return {'archive_path':str(archive),'archive_sha256':archive_sha,'registry_manifest_path':str(manifest.resolve()),'registry_manifest_digest':registry_digest,'config_digest':registry['config']['digest'],'platform':'linux/amd64','official_tag':base_image,'layers':layers,'verification':'Manifest raw SHA -> exact config blob -> every ordered uncompressed layer diff_id; no layer rewriting.'}

def python_tree_inventory(directory,expected=None):
    root=safe_path(directory)
    if not root.is_dir():raise ValueError('Managed Python directory missing')
    installs=[p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and p.name.startswith('cpython-3.13.')]
    if len(installs)!=1 or not (installs[0]/'bin/python3.13').is_file():raise ValueError('Expected exactly one uv-managed CPython3.13 Linux installation')
    entries=[]
    for p in sorted(root.rglob('*')):
        rel=str(p.relative_to(root));mode=stat.S_IMODE(p.lstat().st_mode)
        if p.is_symlink():
            target=os.readlink(p)
            if not p.resolve().is_relative_to(root):raise ValueError('Python cache symlink escapes cache root')
            entries.append({'path':rel,'kind':'symlink','target':target,'mode':mode})
        elif p.is_file():entries.append({'path':rel,'kind':'file','sha256':sha_file(p),'bytes':p.stat().st_size,'mode':mode})
        elif p.is_dir():entries.append({'path':rel,'kind':'directory','mode':mode})
        else:raise ValueError('Unsupported Python cache filesystem entry')
    digest=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if expected is not None and digest!=expected:raise ValueError('Managed Python tree SHA mismatch')
    name=installs[0].name
    version_entries=[{'path':'.','kind':'directory','mode':stat.S_IMODE(installs[0].stat().st_mode)}]
    for entry in entries:
        if entry['path'].startswith(name+'/'):
            version_entries.append({**entry,'path':entry['path'][len(name)+1:]})
    version_digest=hashlib.sha256(json.dumps(version_entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'host_path':str(root),'container_path':'/agent-server/uv-managed-python','installation_name':name,'tree_sha256':digest,'entries':entries,'installation_host_path':str(installs[0]),'installation_container_path':'/agent-server/uv-managed-python/'+name,'installation_tree_sha256':version_digest,'installation_entries':version_entries,'excluded_cache_root_entries':sorted(p.name for p in root.iterdir() if p.name!=name),'transport':'Copy only the exact versioned installation directory. Cache aliases and .lock are omitted; UV creates its own links.'}

def managed_python_files_line(cache):
    """Do not pass the cache root to Apptainer: directory symlinks dereference."""
    name=cache['installation_name']
    if pathlib.Path(name).name!=name or not name.startswith('cpython-3.13.'):
        raise ValueError('Expected exact versioned managed Python directory name')
    source=safe_path(cache['installation_host_path'])
    if source!=safe_path(pathlib.Path(cache['host_path'])/name):
        raise ValueError('Python installation path differs from its inventory')
    destination='/agent-server/uv-managed-python/'+name
    if cache['installation_container_path']!=destination:
        raise ValueError('Python installation destination differs from inventory')
    return f'    {source} {destination}\n'

def adapt_definition(original,base_image,archive,managed_python=None):
    header=f'Bootstrap: docker\nFrom: {base_image}\n'
    if not original.startswith(header):raise ValueError('Official definition header changed')
    replacement=f'Bootstrap: docker-archive\nFrom: {safe_path(archive)}\n'
    adapted=replacement+original[len(header):]
    addition=None
    if managed_python is not None:
        if adapted.count('%files\n')!=1:raise ValueError('Official files section changed')
        addition=managed_python_files_line(managed_python)
        adapted=adapted.replace('%files\n','%files\n'+addition,1)
    reverted=adapted.replace(replacement,header,1)
    if addition:reverted=reverted.replace('%files\n'+addition,'%files\n',1)
    if reverted!=original:raise AssertionError('Unexpected recipe mutation')
    return adapted,{'original_definition_sha256':hashlib.sha256(original.encode()).hexdigest(),'adapted_definition_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'changed_only_bootstrap_from_and_optional_python_cache_file':True,'optional_files_line':addition,'post_environment_runscript_byte_identical':True}
