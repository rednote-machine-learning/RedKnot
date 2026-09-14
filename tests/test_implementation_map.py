"""Code-navigation and no-native-SGLang boundary checks, without model imports."""

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

from vllm_redknot.code_map import implementation_map

ROOT = Path(__file__).resolve().parents[1]


class ImplementationMapTests(unittest.TestCase):
    def test_all_markers_and_symbols_are_real(self):
        index = implementation_map()
        self.assertEqual(index["schema_version"], 1)
        seen = set()
        for entry in index["entries"]:
            self.assertNotIn(entry["id"], seen)
            seen.add(entry["id"])
            path = Path(entry["path"])
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            source = (ROOT / path).read_text(encoding="utf-8")
            self.assertIn(f"# REDKNOT: {entry['id']} —", source)
            names = {
                node.name
                for node in ast.walk(ast.parse(source))
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            }
            self.assertIn(entry["symbol"], names)
            self.assertIn(entry["id"], (ROOT / "README.md").read_text())
        self.assertGreaterEqual(len(seen), 11)

    def test_core_is_not_claimed_runtime_integrated(self):
        self.assertEqual(
            implementation_map()["extracted_core"]["status"],
            "extracted_not_runtime_integrated",
        )

    def test_no_native_sglang_imports_or_namespace_tree(self):
        package = ROOT / "vllm_redknot"
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                elif isinstance(node, ast.Call) and node.args:
                    name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                    if name in {"__import__", "import_module"}:
                        if isinstance(node.args[0], ast.Constant):
                            modules.append(str(node.args[0].value))
                for module in modules:
                    self.assertNotIn(
                        module.split(".")[0], {"sglang", "sgl_kernel"}, str(path)
                    )
        self.assertFalse((ROOT / "sglang").exists())
        self.assertFalse((ROOT / "python" / "sglang").exists())

    def test_cli_does_not_import_engines_or_torch(self):
        code = (
            "import sys; from vllm_redknot.cli import main; "
            "assert main(['code-map']) == 0; "
            "assert not {'torch', 'vllm', 'sglang', 'triton'} & set(sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
