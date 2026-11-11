"""Select already-verified SIFs without registry lookups or builds during inference."""
import hashlib,json,pathlib

def file_sha(path):
    with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_images(provenance_paths,selected_rows,sdk_commit):
    by_id={}
    for path in provenance_paths:
        value=json.loads(pathlib.Path(path).read_text())
        if value.get('schema')!='s2-official-image-preparation-v2' or value.get('status')!='image_built_or_existing':raise ValueError('Unadmitted image provenance schema/status')
        image=pathlib.Path(value['sif_path']).resolve()
        if not image.is_file() or file_sha(image)!=value['sif_sha256']:raise ValueError('Admitted SIF bytes changed')
        if value.get('sdk_commit')!=sdk_commit or value.get('target')!='source-minimal':raise ValueError('Admitted SIF SDK/target mismatch')
        iid=value['instance_id']
        if iid in by_id:raise ValueError('Duplicate image task provenance')
        by_id[iid]={**value,'provenance_path':str(pathlib.Path(path).resolve()),'provenance_file_sha256':file_sha(path)}
    selected={r['instance_id']:r for r in selected_rows}
    if set(by_id)!=set(selected):raise ValueError('Image admission must exactly cover selected execution tasks')
    for iid,row in selected.items():
        value=by_id[iid]
        if value['base_commit']!=row['base_commit'] or value['base_image_tag']!='docker.io/'+row['image_tag_reference']:raise ValueError('Admitted image task/base identity differs')
    return by_id

def install(module,images):
    originals=(module.remote_image_exists,module.ensure_apptainer_agent_image)
    by_base={v['base_image_tag']:v for v in images.values()}
    def local_only(*,base_image,custom_tag,target,wrap_swebench_deps):
        value=by_base.get(base_image)
        if value is None or target!='source-minimal' or bool(wrap_swebench_deps)!=value['sphinx_dependency_wrap']:raise ValueError('Unadmitted image request')
        return pathlib.Path(value['sif_path'])
    module.remote_image_exists=lambda _:False
    module.ensure_apptainer_agent_image=local_only
    return originals

def restore(module,originals):
    module.remote_image_exists,module.ensure_apptainer_agent_image=originals
