import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demo import CatApp
from upload_gitee import repository_url


class DefaultModel(unittest.TestCase):
    def test_upload_rejects_other_hosts_credentials_and_url_parameters(self):
        self.assertEqual(repository_url('https://gitee.com/student/cats.git'),
                         'https://gitee.com/student/cats')
        for value in ['https://evil.com/a/b', 'https://gitee.com@evil.com/a/b',
                      'https://u:p@gitee.com/a/b', 'https://gitee.com/a/b?x=1',
                      'https://gitee.com/a/b;whoami', 'https://gitee.com/../cats']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                repository_url(value)

    def test_saved_selection_controls_default_even_when_newer_run_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ('finetune', 'optimized', 'expanded'):
                folder = root / 'artifacts' / run
                folder.mkdir(parents=True)
                (folder / 'best.pt').touch()
            (root / 'artifacts' / 'final_selection.json').write_text(
                json.dumps({'run': 'optimized'}), encoding='utf-8')
            with patch('demo.ROOT', root):
                models = CatApp.find_models(None)
            self.assertEqual(next(iter(models.values())).parent.name, 'optimized')
            self.assertEqual(len(models), 3)


if __name__ == '__main__':
    unittest.main()
