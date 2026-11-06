"""Joint predictions use the model verdict and one shared explanation trajectory."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from forensics.runtime import Runtime


class JointInferenceTests(unittest.TestCase):
    def runtime(self, prediction, suffix):
        runtime = Runtime.__new__(Runtime)
        runtime.device = torch.device('cpu')
        runtime.spatial_ready = True
        runtime.control_ids = [1, 2, 3]
        runtime.base_generation = SimpleNamespace(_from_model_config=True)
        runtime.tokenizer = SimpleNamespace(eos_token_id=0,
            decode=Mock(return_value='Unnatural geometry. [SEG]</s>'))
        runtime.classify_many = Mock(return_value=[dict(prediction_centered=prediction,
            margin=-1., score_fake=.4, center=-9.190834884678093)])
        runtime.sample = Mock(return_value=dict(label=torch.zeros(2, 3)))
        def batch(samples, prefix, **kwargs):
            return dict(input_ids=torch.tensor([[9, 1, 2 if prefix == '[FAKE]' else 3]]),
                        global_enc_images=torch.ones(1, 3, 2, 2), bboxes=None)
        runtime.backend = SimpleNamespace(_batch_many=Mock(side_effect=batch))
        def generate(**kwargs):
            return SimpleNamespace(sequences=torch.cat((kwargs['input_ids'], torch.tensor([suffix])), dim=1))
        runtime.owner = SimpleNamespace(seg_token_idx=8, config=SimpleNamespace(pad_token_id=0),
            _rine_evidence_token=Mock(return_value=torch.ones(1, 1, 2)),
            generate=Mock(side_effect=generate),
            _extract_projected_seg_predictor_hidden=Mock(return_value=([torch.ones(1, 2)], [])))
        runtime.native = Mock(return_value=np.ones((2, 3), dtype=bool))
        return runtime

    @patch('model.llava.model.language_model.llava_llama.LlavaLlamaForCausalLM.forward')
    def test_mask_uses_the_single_explanation_sequence(self, replay):
        replay.return_value = SimpleNamespace(hidden_states=torch.ones(1, 8, 2))
        runtime = self.runtime(1, [10, 8, 0])
        result, mask = runtime.predict(dict(sample_id='image', label=0))
        self.assertEqual(result['label'], 'synthetic')
        self.assertEqual(result['explanation'], 'Unnatural geometry.')
        self.assertTrue(mask.all())
        runtime.owner.generate.assert_called_once()
        self.assertEqual(runtime.backend._batch_many.call_args.args[1], '[FAKE]')
        self.assertEqual(replay.call_args.kwargs['input_ids'].tolist(), [[9, 1, 2, 10, 8, 0]])
        self.assertIs(replay.call_args.kwargs['forensic_evidence_tokens'],
                      runtime.owner.generate.call_args.kwargs['forensic_evidence_tokens'])
        runtime.native.assert_called_once()

    @patch('model.llava.model.language_model.llava_llama.LlavaLlamaForCausalLM.forward')
    def test_missing_query_does_not_regenerate_the_explanation(self, replay):
        replay.return_value = SimpleNamespace(hidden_states=torch.ones(1, 8, 2))
        runtime = self.runtime(1, [10, 0])
        runtime.predict(dict(sample_id='image'))
        runtime.owner.generate.assert_called_once()
        self.assertEqual(replay.call_args.kwargs['input_ids'].tolist(), [[9, 1, 2, 10, 8]])

    @patch('model.llava.model.language_model.llava_llama.LlavaLlamaForCausalLM.forward')
    def test_real_prediction_has_an_empty_mask_without_reference_labels(self, replay):
        runtime = self.runtime(0, [10, 8, 0])
        result, mask = runtime.predict(dict(sample_id='image', label=1))
        self.assertEqual(result['label'], 'real')
        self.assertEqual(mask.shape, (2, 3))
        self.assertFalse(mask.any())
        self.assertEqual(runtime.backend._batch_many.call_args.args[1], '[REAL]')
        runtime.owner.generate.assert_called_once()
        runtime.native.assert_not_called()
        replay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
