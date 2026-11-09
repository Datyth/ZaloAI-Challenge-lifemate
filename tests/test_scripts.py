import tempfile
import unittest
from pathlib import Path

from lifemate.submission import shard_dirs
from scripts.data.prepare_agent_sft import history_audio_ids


class ScriptsTest(unittest.TestCase):
    def test_history_audio_validation_id_detection(self):
        item = {"audio": "audio/train_0001/train_0001.wav", "history": [
            {"role": "user", "audio": "audio/train_0234/train_0234.wav"}]}
        self.assertEqual(history_audio_ids(item), {"train_0001", "train_0234"})

    def test_shard_dirs_match_only_own_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            for name in ["pub_s0", "pub_s1", "pub_s10", "pub_synth_s0", "pub_sx"]:
                (runs / name).mkdir()
            self.assertEqual([d.name for d in shard_dirs(runs / "pub")], ["pub_s0", "pub_s1", "pub_s10"])
            self.assertEqual(shard_dirs(runs / "pub_synth_s0"), [runs / "pub_synth_s0"])
            self.assertEqual(shard_dirs(runs / "none"), [])


if __name__ == "__main__":
    unittest.main()
