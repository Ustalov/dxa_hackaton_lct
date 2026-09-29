import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
from dxa.prototype_export import annotation_archive


class AnnotationExport(unittest.TestCase):
    def test_bundle_maps_images_and_preserves_failed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('overlays', 'masks', 'images'):
                (root/name).mkdir()
            data = {'rows': [{'path_to_study': 'case.dcm'}, {'processing_status': 'Failure'}],
                    'details': [{'index': 0, 'file': 'case.dcm', 'has_overlay': True},
                                {'index': 1, 'file': 'broken.dcm', 'has_overlay': False}]}
            (root/'results.json').write_text(json.dumps(data))
            (root/'results.xlsx').write_bytes(b'xlsx')
            (root/'overlays/0.png').write_bytes(b'overlay')
            (root/'masks/0_lt.png').write_bytes(b'mask')
            (root/'images/0.png').write_bytes(b'original')
            (root/'private.json').write_text('not exported')
            bundle = annotation_archive(root)
            with ZipFile(bundle) as archive:
                self.assertEqual(set(archive.namelist()), {'results.json', 'results.xlsx',
                    'overlays/0.png', 'masks/0_lt.png', 'README.txt'})
                self.assertEqual(json.loads(archive.read('results.json')), data)
                self.assertIsNone(archive.testzip())
            timestamp = bundle.stat().st_mtime_ns
            self.assertEqual(annotation_archive(root).stat().st_mtime_ns, timestamp)
            self.assertFalse(list(root.glob('.annotations-*.tmp')))

    def test_missing_overlay_does_not_leave_partial_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'results.json').write_text(json.dumps({'details':[{'index':0,'has_overlay':True}]}))
            (root/'results.xlsx').write_bytes(b'xlsx')
            with self.assertRaises(FileNotFoundError):
                annotation_archive(root)
            self.assertFalse((root/'annotations.zip').exists())
            self.assertFalse(list(root.glob('.annotations-*.tmp')))


if __name__ == '__main__':
    unittest.main()
