"""Prepare one selected official agent image; no model or GPU calls.

Optional offline inputs adapt the official generated definition outside upstream:
Docker Bootstrap/From -> verified docker-archive; optional %files copy seeds uv's
managed Python cache. A separately verified dependency bundle can replace only
package-install commands with an exact offline replay. Official SDK sources,
lock, runtime environment and runscript stay unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,sys
B=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from run_official import verify_inputs
from offline_image_adapter import archive_provenance,python_tree_inventory,adapt_definition,sha_file,managed_python_files_line
from sdk_bundle_adapter import validate_bundle,adapt_dependency_install

def main():
 p=argparse.ArgumentParser();p.add_argument('--ordinal',type=int,default=0);p.add_argument('--build-root',required=True);p.add_argument('--run',action='store_true');p.add_argument('--force',action='store_true');p.add_argument('--docker-archive');p.add_argument('--archive-sha256');p.add_argument('--registry-manifest');p.add_argument('--registry-manifest-digest');p.add_argument('--managed-python-dir');p.add_argument('--managed-python-tree-sha256');p.add_argument('--sdk-bundle-receipt');a=p.parse_args();m,s=verify_inputs()
 if not 0<=a.ordinal<20:p.error('ordinal must be in fixed20')
 offline=(a.docker_archive,a.archive_sha256,a.registry_manifest,a.registry_manifest_digest)
 if any(offline) and not all(offline):p.error('All four Docker archive/registry verification arguments are required together')
 if a.managed_python_tree_sha256 and not a.managed_python_dir:p.error('Python tree digest requires --managed-python-dir')
 if a.sdk_bundle_receipt and not a.managed_python_dir:p.error('Offline SDK bundle requires the admitted managed Python cache')
 row=s['cases'][a.ordinal];base='docker.io/'+row['image_tag_reference'];plan={'schema':'s2-official-image-preparation-v2','instance_id':row['instance_id'],'base_commit':row['base_commit'],'base_image_tag':base,'sdk_commit':m['sources']['sdk']['commit'],'target':'source-minimal','sphinx_dependency_wrap':row['repo'].startswith('sphinx-doc/'),'build_root':str(pathlib.Path(a.build_root).resolve()),'mode':'execute_official_cpu_image_builder' if a.run else 'plan_only','model_calls':0,'gpu':False}
 archive=None;python_cache=None;bundle=None
 if a.docker_archive:archive=archive_provenance(*offline,base);plan['offline_archive']=archive
 if a.managed_python_dir:python_cache=python_tree_inventory(a.managed_python_dir,a.managed_python_tree_sha256);plan['managed_python_cache']=python_cache
 if a.sdk_bundle_receipt:bundle=validate_bundle(a.sdk_bundle_receipt,m['sources']['sdk']['commit'],sha_file(B/'benchmarks/vendor/software-agent-sdk/uv.lock'));plan['sdk_dependency_bundle']=bundle
 if not a.run:print(json.dumps(plan,indent=2));return
 os.environ['OPENHANDS_APPTAINER_BUILD_ROOT']=plan['build_root']
 if a.force:os.environ['OPENHANDS_APPTAINER_FORCE_BUILD']='1'
 elif os.getenv('OPENHANDS_APPTAINER_FORCE_BUILD','').lower() in {'1','true','yes'}:raise ValueError('Inherited force-build flag requires explicit --force')
 sys.path.insert(0,str(B/'benchmarks'))
 from benchmarks.swebench import apptainer_build as builder
 from benchmarks.swebench.build_images import extract_custom_tag,get_official_docker_image,should_wrap_instance_id
 actual=get_official_docker_image(row['instance_id'])
 if actual!=base:raise RuntimeError('Official base image differs from pinned task metadata')
 image=builder.apptainer_agent_image_path(extract_custom_tag(actual),'source-minimal')
 receipt=image.with_suffix('.s2-offline-provenance.json')
 identity={'archive_sha256':archive['archive_sha256'] if archive else None,'registry_manifest_digest':archive['registry_manifest_digest'] if archive else None,'managed_python_tree_sha256':python_cache['tree_sha256'] if python_cache else None,'managed_python_installation_tree_sha256':python_cache['installation_tree_sha256'] if python_cache else None,'managed_python_transport':'versioned_directory_only_v1' if python_cache else None,'sdk_commit':plan['sdk_commit'],'base_image_tag':base,'sdk_bundle_receipt_sha256':bundle['receipt_file_sha256'] if bundle else None}
 if image.exists() and not a.force and (archive or python_cache):
  if not receipt.is_file():raise ValueError('Existing SIF lacks offline provenance; use an independent build root')
  prior=json.loads(receipt.read_text())
  if prior['input_identity']!=identity or prior['sif_sha256']!=sha_file(image):raise ValueError('Existing SIF provenance differs')
 original=builder._definition_file_content
 mutation={}
 def offline_definition(*args,**kwargs):
  text=original(*args,**kwargs)
  if archive:
   adapted,proof=adapt_definition(text,base,archive['archive_path'],python_cache)
  elif python_cache:
   # No transport adaptation requested: preserve the original docker header.
   addition=managed_python_files_line(python_cache)
   if text.count('%files\n')!=1:raise ValueError('Official files section changed')
   adapted=text.replace('%files\n','%files\n'+addition,1)
   if adapted.replace('%files\n'+addition,'%files\n',1)!=text:raise AssertionError('Unexpected recipe change')
   proof={'original_definition_sha256':hashlib.sha256(text.encode()).hexdigest(),'adapted_definition_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'optional_files_line':addition,'post_environment_runscript_byte_identical':True}
  else:return text
  mutation.update(proof)
  if bundle:
   adapted,dependency_proof=adapt_dependency_install(adapted,bundle)
   mutation['dependency_install_adapter']=dependency_proof
   mutation['changed_only_bootstrap_from_and_optional_python_cache_file']=False
   mutation['post_environment_runscript_byte_identical']=False
   mutation['runtime_environment_and_runscript_byte_identical']=True
  return adapted
 if archive or python_cache:builder._definition_file_content=offline_definition
 try:
  image=pathlib.Path(builder.ensure_apptainer_agent_image(base_image=actual,custom_tag=extract_custom_tag(actual),target='source-minimal',wrap_swebench_deps=should_wrap_instance_id(row['instance_id'])))
 finally:builder._definition_file_content=original
 plan.update(status='image_built_or_existing',sif_path=str(image),sif_sha256=sha_file(image),sif_bytes=image.stat().st_size,input_identity=identity,definition_adapter=mutation or {'existing_verified_receipt':str(receipt) if receipt.exists() else None},remaining='Actual official workspace startup/isolation/first request and base_commit admission still required.')
 if archive or python_cache:
  # Preserve existing matching provenance; never replace it for an existing cache hit.
  if not receipt.exists() or a.force:receipt.write_text(json.dumps(plan,indent=2)+'\n')
 print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
