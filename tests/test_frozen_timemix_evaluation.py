import json
from pathlib import Path
import tempfile
import unittest
from scripts.evaluate_frozen_timemix import verify_artifact,sha


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'checkpoint.pth').write_bytes(b'dummy weights')
        config=dict(seq_len=96,pred_len=96,features='M',data='ETTm1',inverse=False,model='OscillatorLinearBlend')
        (self.root/'model_config.json').write_text(json.dumps(config))
        self.manifest=dict(kind='OscillatorLinearBlend',selection_status='validation_candidate_not_test_frozen',checkpoint_sha256=sha(self.root/'checkpoint.pth'),model_config_sha256=sha(self.root/'model_config.json'))
        self.save()
    def tearDown(self):self.temp.cleanup()
    def save(self):(self.root/'manifest.json').write_text(json.dumps(self.manifest))
    def freeze(self):
        self.manifest['selection_status']='frozen_for_test';self.save()
        (self.root/'freeze.json').write_text(json.dumps({**self.manifest,'reason':'fixed beforehand','validation_evidence':'report.json'}))
    def test_unfrozen_allows_validation_but_not_test(self):
        verify_artifact(self.root,'val',32)
        with self.assertRaisesRegex(ValueError,'selection'):verify_artifact(self.root,'test')
    def test_frozen_complete_test_only(self):
        self.freeze();verify_artifact(self.root,'test')
        with self.assertRaisesRegex(ValueError,'every official'):verify_artifact(self.root,'test',32)
    def test_changed_config_or_weights_rejected(self):
        self.freeze();(self.root/'checkpoint.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):verify_artifact(self.root,'test')
    def test_mismatched_freeze_rejected(self):
        self.freeze();f=self.root/'freeze.json';d=json.loads(f.read_text());d['checkpoint_sha256']='other';f.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'identity'):verify_artifact(self.root,'test')


if __name__=='__main__':unittest.main()


class RuntimeManifestTests(FrozenEvaluationTests):
    def test_runtime_manifest_requires_original_identity_and_sources(self):
        from scripts.evaluate_frozen_timemix import ROOT
        source='models/OscillatorLinearBlend.py'
        runtime=dict(original_manifest_sha256=sha(self.root/'manifest.json'),
                     checkpoint_sha256=self.manifest['checkpoint_sha256'],
                     model_config_sha256=self.manifest['model_config_sha256'],
                     source_sha256={source:sha(ROOT/source)})
        path=self.root/'runtime_manifest.json'
        path.write_text(json.dumps(runtime))
        verify_artifact(self.root,'val')
        runtime['source_sha256'][source]='0'*64
        path.write_text(json.dumps(runtime))
        with self.assertRaisesRegex(ValueError,'Runtime source'):
            verify_artifact(self.root,'val')
        runtime['original_manifest_sha256']='0'*64
        path.write_text(json.dumps(runtime))
        with self.assertRaisesRegex(ValueError,'Historical manifest'):
            verify_artifact(self.root,'val')
