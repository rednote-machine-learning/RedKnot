"""Check the documentation's migrated-versus-wired model asset index."""

import ast
import unittest
from pathlib import Path

from vllm_redknot.code_map import implementation_map

ROOT = Path(__file__).resolve().parents[1]


class MultiModelIndexTests(unittest.TestCase):
    def test_asset_paths_and_symbols_exist_without_importing_models(self):
        model = implementation_map()["migrated_model_backends"]
        self.assertIs(model["runtime_integrated"], False)
        self.assertEqual(model["status"], "migrated_not_runtime_integrated")
        for group in ("mha_swa", "qwen35", "benchmarks"):
            entry = model[group]
            paths = entry.get("paths", [entry.get("path")])
            for name in paths + [entry["documentation"]]:
                path = Path(name)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertTrue((ROOT / path).is_file(), name)
        names = {
            node.name
            for node in ast.walk(
                ast.parse((ROOT / model["mha_swa"]["path"]).read_text())
            )
            if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(set(model["mha_swa"]["symbols"]) <= names)

    def test_readme_names_all_four_entries_and_separates_runtime_status(self):
        model = implementation_map()["migrated_model_backends"]
        readme = (ROOT / "README.md").read_text()
        self.assertEqual(len(model["benchmarks"]["paths"]), 4)
        for name in model["benchmarks"]["paths"]:
            self.assertIn(Path(name).name, readme)
        self.assertIn("runtime_integrated: false", readme)
        self.assertIn("CPU", readme)
        self.assertEqual(model["benchmarks"]["default_action"], "cpu_migration_plan")


if __name__ == "__main__":
    unittest.main()
