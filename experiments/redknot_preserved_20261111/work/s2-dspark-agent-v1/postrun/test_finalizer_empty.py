from pathlib import Path
import tempfile
import unittest

import finalize_first_task as f


class ClosedOutputTests(unittest.TestCase):
    def test_empty_and_blank_output_have_no_closed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.jsonl'
            for content in ('', '\n  \n'):
                path.write_text(content)
                self.assertEqual(f.official_rows(path, 'task0'), [])

    def test_nonempty_task_mismatch_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.jsonl'
            path.write_text('{"instance_id":"another-task"}\n')
            with self.assertRaisesRegex(ValueError, 'official output task differs'):
                f.official_rows(path, 'task0')
            path.write_text('{"instance_id":"task0","test_result":{"git_patch":""}}\n')
            self.assertEqual(len(f.official_rows(path, 'task0')), 1)


if __name__ == '__main__':
    unittest.main()
