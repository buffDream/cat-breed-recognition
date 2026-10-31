import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demo import CatApp


class DefaultModel(unittest.TestCase):
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
