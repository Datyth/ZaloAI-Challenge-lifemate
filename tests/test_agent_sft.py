import unittest

from scripts.prepare_agent_sft import history_audio_ids
from scripts.train_agent_sft import decisions
from scripts.compare_agent_sft import one_sample, summarize


class AgentSFTTest(unittest.TestCase):
    def test_decisions_keep_tool_observations_in_later_prompts(self):
        row = {"tools": [], "messages": [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {}}}]},
            {"role": "tool", "content": "observed"},
            {"role": "assistant", "content": "done"},
        ]}
        examples = decisions([row])
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[1]["prompt"][-1]["content"], "observed")
        self.assertEqual(examples[0]["completion"][0]["role"], "assistant")

    def test_history_audio_validation_id_detection(self):
        item = {"audio": "audio/train_0001/train_0001.wav", "history": [
            {"role": "user", "audio": "audio/train_0234/train_0234.wav"}]}
        self.assertEqual(history_audio_ids(item), {"train_0001", "train_0234"})

    def test_comparison_counts_selection_arguments_and_runtime(self):
        gold = [{"name": "calendar.list", "arguments": {}},
                {"name": "calendar.create", "arguments": {"title": "A"}}]
        pred = {"tool_calls": [{"name": "calendar.list", "arguments": {}},
                               {"name": "calendar.create", "arguments": {"title": "B"}}],
                "status": "ok", "train_state_match": False, "latency_seconds": 2.0}
        sample = one_sample(pred, gold, {"calendar.list": "read", "calendar.create": "write"})
        self.assertEqual(sample["selection_matched"], 2)
        self.assertEqual(sample["argument_correct"], 1)
        self.assertTrue(sample["read_to_write_complete"])
        self.assertFalse(sample["state_match"])
        summary = summarize("test", __import__("pathlib").Path("test"), [sample])
        self.assertEqual(summary["argument_accuracy_on_aligned_calls"], 0.5)
        self.assertEqual(summary["tool_selection_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
