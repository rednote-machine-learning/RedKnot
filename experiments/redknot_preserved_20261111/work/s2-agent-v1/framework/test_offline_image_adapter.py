import hashlib,io,json,pathlib,tarfile,tempfile,unittest
from offline_image_adapter import adapt_definition,archive_provenance,python_tree_inventory,sha_file,managed_python_files_line
class OfflineImageAdapterTests(unittest.TestCase):
 def test_actual_official_generator_only_authorized_deltas(self):
  import ast,os
  from pathlib import Path
  source=Path(__file__).parent/'benchmarks/benchmarks/swebench/apptainer_build.py'
  tree=ast.parse(source.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_definition_file_content','_package_install_script','_wrap_swebench_deps_script'}]
  ns={'Path':Path,'os':os,'_sdk_root':lambda:Path('/source/sdk')};exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
  original=ns['_definition_file_content']('docker.io/official:test','pinnedsha','pinnedref',True,Path('/usr/bin/uv'),None)
  cache={'host_path':'/offline/python','installation_name':'cpython-3.13.18-linux-x86_64-gnu','installation_host_path':'/offline/python/cpython-3.13.18-linux-x86_64-gnu','installation_container_path':'/agent-server/uv-managed-python/cpython-3.13.18-linux-x86_64-gnu'}
  adapted,proof=adapt_definition(original,'docker.io/official:test','/offline/image.tar',cache)
  self.assertTrue(adapted.startswith('Bootstrap: docker-archive\nFrom: /offline/image.tar\n'));self.assertEqual(adapted.split('%post',1)[1],original.split('%post',1)[1]);self.assertTrue(proof['post_environment_runscript_byte_identical']);self.assertIn('uv python install 3.13',adapted)
  with self.assertRaises(ValueError):adapt_definition(original,'wrong','/offline/image.tar')
  self.assertIn('    /offline/python/cpython-3.13.18-linux-x86_64-gnu /agent-server/uv-managed-python/cpython-3.13.18-linux-x86_64-gnu\n',adapted)
  self.assertNotIn('    /offline/python /agent-server/uv-managed-python\n',adapted)
 def make_archive(self,root,tamper=False):
  layer=b'official layer content';diff='sha256:'+hashlib.sha256(layer).hexdigest();config=json.dumps({'os':'linux','architecture':'amd64','rootfs':{'type':'layers','diff_ids':[diff]}}).encode();registry=json.dumps({'schemaVersion':2,'config':{'digest':'sha256:'+hashlib.sha256(config).hexdigest()},'layers':[{'digest':'sha256:compressedblob'}]}).encode();manifest=root/'registry.json';manifest.write_bytes(registry)
  archive=root/'image.tar';saved=json.dumps([{'Config':'config.json','RepoTags':['repo/task:latest'],'Layers':['layer.tar']}]).encode()
  with tarfile.open(archive,'w') as t:
   for name,data in [('config.json',config),('manifest.json',saved),('layer.tar',b'changed' if tamper else layer)]:
    info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
  return archive,sha_file(archive),manifest,'sha256:'+hashlib.sha256(registry).hexdigest(),'docker.io/repo/task:latest'
 def test_archive_layers_are_bound_through_official_config(self):
  with tempfile.TemporaryDirectory() as td:
   args=self.make_archive(pathlib.Path(td));r=archive_provenance(*args);self.assertEqual(len(r['layers']),1)
   with self.assertRaises(ValueError):archive_provenance(*args[:-1],'docker.io/wrong:latest')
   args=self.make_archive(pathlib.Path(td),True)
   with self.assertRaises(ValueError):archive_provenance(*args)
 def test_python_cache_tree_is_bound_and_escape_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);install=root/'cpython-3.13.18-linux-x86_64-gnu';(install/'bin').mkdir(parents=True);(install/'bin/python3.13').write_bytes(b'CPU fixture, not executable proof')
   r=python_tree_inventory(root);self.assertEqual(python_tree_inventory(root,r['tree_sha256'])['tree_sha256'],r['tree_sha256'])
   (install/'bin/escape').symlink_to('/tmp')
   with self.assertRaises(ValueError):python_tree_inventory(root)
 def test_version_tree_digest_ignores_excluded_minor_alias_and_cache_lock(self):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);name='cpython-3.13.18-linux-x86_64-gnu';install=root/name;(install/'bin').mkdir(parents=True);(install/'bin/python3.13').write_bytes(b'actual version fixture');(install/'bin/python').symlink_to('python3.13')
   first=python_tree_inventory(root)
   (root/'cpython-3.13-linux-x86_64-gnu').symlink_to(name,target_is_directory=True);(root/'.lock').write_text('cache lock')
   second=python_tree_inventory(root)
   self.assertEqual(first['installation_tree_sha256'],second['installation_tree_sha256'])
   self.assertNotEqual(first['tree_sha256'],second['tree_sha256'])
   self.assertEqual(second['excluded_cache_root_entries'],['.lock','cpython-3.13-linux-x86_64-gnu'])
   line=managed_python_files_line(second)
   self.assertEqual(line,f'    {install.resolve()} /agent-server/uv-managed-python/{name}\n')
   self.assertNotIn('/cpython-3.13-linux-x86_64-gnu',line);self.assertNotIn('.lock',line)
   (install/'bin/python3.13').write_bytes(b'changed installation')
   self.assertNotEqual(second['installation_tree_sha256'],python_tree_inventory(root)['installation_tree_sha256'])
if __name__=='__main__':unittest.main()
