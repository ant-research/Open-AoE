import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v4_two_full_pipelines.sh"


class WorkerSocketPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_default_worker_sockets_use_a_short_run_hash(self) -> None:
        self.assertIn('worker_socket_id="$(printf', self.script)
        self.assertIn("sha256sum", self.script)
        self.assertRegex(
            self.script,
            re.compile(r'sam3_socket=.*?/tmp/eis3_.*?\$\{worker_socket_id\}\.sock'),
        )
        self.assertNotIn("/tmp/egoinfinity_sam3_${USER:-user}_${run_name}.sock", self.script)

    def test_explicit_socket_overrides_are_length_checked(self) -> None:
        self.assertIn('for worker_socket in "$sam3_socket" "$sam3d_socket"', self.script)
        self.assertIn("${#worker_socket} > 99", self.script)
        self.assertIn("AF_UNIX limit", self.script)


if __name__ == "__main__":
    unittest.main()
