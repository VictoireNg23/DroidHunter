import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droidhunter.images import to_square  # noqa: E402
from droidhunter.metrics import aut  # noqa: E402
from droidhunter.models import DroidHunterCNN  # noqa: E402
from droidhunter.records import Record, RecordBuilder  # noqa: E402
from droidhunter.smali import iter_smali_instructions, method_path, parse_instruction  # noqa: E402
from droidhunter.vocab import Scaler, trigram_keys  # noqa: E402

SMALI = """.class public LC;
.method public a()V
    .locals 2
    invoke-direct {p0, v1}, Ljava/lang/Object;-><init>()V
    const-string v0, "a, b"
    :cond_0
    if-eqz v0, :cond_0
    .packed-switch 0x1
        :pswitch_0
    .end packed-switch
    iget-object v0, p0, LC;->f:Ljava/lang/String;
    return-void
.end method
"""


class Parsing(unittest.TestCase):
    def test_operands(self):
        self.assertEqual(parse_instruction("invoke-direct", "{p0, v1}, Ljava/lang/Object;-><init>()V"),
                         ("invoke-direct", "p0|v1", "Ljava/lang/Object;-><init>()V"))
        self.assertEqual(parse_instruction("const-string", 'v0, "a, b"'), ("const-string", "v0", "<STR>"))
        self.assertEqual(parse_instruction("iget-object", "v0, p0, LC;->f:I"), ("iget-object", "v0|p0", "LC;->f:I"))
        self.assertEqual(parse_instruction("invoke-static/range", "{v0 .. v5}, LA;->b()V")[1], "v0|v5")
        self.assertEqual(parse_instruction("return-void", ""), ("return-void", "<NONE>", "<NONE>"))

    def test_multidex_order_and_method_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("smali_classes2", "smali"):
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / "C.smali").write_text(SMALI)
            ops = {"invoke-direct", "const-string", "if-eqz", "iget-object", "return-void"}
            got = list(iter_smali_instructions(tmp, ops))
            self.assertEqual(len(got), 10)
            self.assertEqual([g[0] for g in got[:5]],
                             ["invoke-direct", "const-string", "if-eqz", "iget-object", "return-void"])

    def test_method_path(self):
        self.assertEqual(method_path("Ljava/lang/Object;-><init>()V", 6), ["java", "lang", "Object", "<init>", "()V"])
        self.assertEqual(method_path("<STR>", 6), ["<STR>"])


class Encoding(unittest.TestCase):
    def test_record_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RecordBuilder()
            for t in [("a", "v0", "<NONE>"), ("b", "v1", "X"), ("a", "v0", "<NONE>")]:
                b.add(t)
            b.save(Path(tmp) / "r.npz")
            r = Record(Path(tmp) / "r.npz")
            self.assertEqual(len(r), 3)
            self.assertEqual(r.tables[0][r.idx[0]].tolist(), ["a", "b", "a"])

    def test_scaler_unknown_is_black(self):
        s = Scaler(np.array([1.0, 5.0, 9.0]), "linear")
        self.assertEqual(s([0, 1, 9]).tolist(), [0, 0, 255])

    def test_square_padding(self):
        img = to_square(np.full((5, 3), 7, np.uint8))
        self.assertEqual(img.shape, (3, 3, 3))
        self.assertEqual(int(img.reshape(-1, 3)[5:].sum()), 0)

    def test_trigram_keys_deterministic(self):
        h = np.array([1, 2, 3, 4], dtype=np.uint64)
        self.assertEqual(trigram_keys(h).size, 2)
        self.assertTrue(np.array_equal(trigram_keys(h), trigram_keys(h.copy())))


class Models(unittest.TestCase):
    def test_cnn_feature_size(self):
        # 400x400 input -> 5x5x128 feature maps -> 3,200 SVM features
        m = DroidHunterCNN(3, 400)
        self.assertEqual(m.features(torch.zeros(1, 3, 400, 400)).shape, (1, 3200))

    def test_aut(self):
        self.assertAlmostEqual(aut([1.0, 1.0, 1.0]), 1.0)
        self.assertAlmostEqual(aut([1.0, 0.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
