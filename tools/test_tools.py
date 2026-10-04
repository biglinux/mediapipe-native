import importlib.util
import json
import os
import pathlib
import struct
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "corpus", ROOT / "tools/compare_corpus.py"
)
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class DevelopmentTools(unittest.TestCase):
    def test_identical_signed_zero(self):
        with tempfile.TemporaryDirectory() as d:
            a = pathlib.Path(d) / "a"
            b = pathlib.Path(d) / "b"
            a.mkdir()
            b.mkdir()
            for p in (a, b):
                (p / "case00-out0.bin").write_bytes(struct.pack("<ff", 0.0, -0.0))
            self.assertTrue(corpus.compare(a, b)["bit_exact"])
            (b / "case00-out0.bin").write_bytes(struct.pack("<ff", -0.0, 0.0))
            r = corpus.compare(a, b)
            self.assertFalse(r["bit_exact"])
            self.assertEqual(r["changed_bits_values"], 2)
            self.assertEqual(r["max_abs_error"], 0.0)

    def test_malformed_nonfinite_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            a = pathlib.Path(d) / "a"
            b = pathlib.Path(d) / "b"
            a.mkdir()
            b.mkdir()
            (a / "case00-out0.bin").write_bytes(struct.pack("<f", 1.0))
            for value in (
                b"x",
                struct.pack("<f", float("nan")),
                struct.pack("<f", float("inf")),
            ):
                (b / "case00-out0.bin").write_bytes(value)
                with self.assertRaises(ValueError):
                    corpus.compare(a, b)
            (b / "case00-out0.bin").unlink()
            with self.assertRaises(ValueError):
                corpus.compare(a, b)

    def test_abba_resume_and_plan_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            binary = d / "fake"
            counter = d / "counter"
            out = d / "result.json"
            binary.write_text(
                "#!/usr/bin/python3\nimport json,os,sys\nwith open("
                + repr(str(counter))
                + ',"a") as f:f.write("x")\nn=int(sys.argv[2]);print(json.dumps(dict(tier=int(os.environ["MEDIAPIPE_NATIVE_TIER"]),frames=n,allocations=0,first_frame_allocations=0,p50_ms=1.,p95_ms=1.,p99_ms=1.,cpu_ms_per_frame=1.,samples_ms=[1.]*n)))\n'
            )
            binary.chmod(0o755)
            plan = d / "plans/face_detector/face_detector.mpplan"
            plan.parent.mkdir(parents=True)
            plan.write_bytes(b"fake-plan")
            cmd = [
                "python3",
                str(ROOT / "tools/bench_abba.py"),
                "--baseline",
                str(binary),
                "--candidate",
                str(binary),
                "--root",
                str(d),
                "--output",
                str(out),
                "--cpu",
                str(min(os.sched_getaffinity(0))),
                "--frames",
                "1",
                "--blocks",
                "1",
                "--tiers",
                "0",
                "--models",
                "face_detector",
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=15)
            self.assertEqual(len(counter.read_text()), 4)
            self.assertNotEqual(
                subprocess.run(
                    cmd, capture_output=True, timeout=15, check=False
                ).returncode,
                0,
            )
            subprocess.run(
                cmd + ["--resume"], check=True, capture_output=True, timeout=15
            )
            self.assertEqual(len(counter.read_text()), 4)
            x = json.loads(out.read_text())
            x["runs"] = x["runs"][:2]
            x["status"] = "RUNNING"
            out.write_text(json.dumps(x))
            subprocess.run(
                cmd + ["--resume"], check=True, capture_output=True, timeout=15
            )
            x = json.loads(out.read_text())
            self.assertEqual(len(counter.read_text()), 8)
            self.assertEqual(len(x["interrupted_runs"]), 2)
            self.assertEqual(x["status"], "PASS")
            plan.write_bytes(b"changed")
            self.assertNotEqual(
                subprocess.run(
                    cmd + ["--resume"], capture_output=True, timeout=15, check=False
                ).returncode,
                0,
            )

    def test_invalid_benchmark_arguments(self):
        binary = os.environ.get("MPBENCH_BIN", str(ROOT / "target/release/mpbench"))
        plan = str(ROOT / "plans/face_detector/face_detector.mpplan")
        for args in ([], [plan, "0"], [plan, "-1"], [plan, "1", "2"], [plan, "--bad"]):
            self.assertEqual(
                subprocess.run(
                    [binary] + args, capture_output=True, timeout=10, check=False
                ).returncode,
                2,
            )


@unittest.skipUnless(importlib.util.find_spec("onnx"), "the exporters need onnx")
class ActivationFolding(unittest.TestCase):
    def exporter(self, ops):
        spec = importlib.util.spec_from_file_location(
            "export_graph", ROOT / "tools/export_graph.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        exporter = module.Exporter.__new__(module.Exporter)
        exporter.ops = ops
        exporter.fold_activations()
        return exporter.ops

    def test_prelu_folds_into_a_producer_listed_earlier(self):
        # TFLite lists the other branch's pooling between a conv and its PReLU.
        ops = self.exporter(
            [
                {"kind": "conv", "_x": "in", "_y": "c"},
                {"kind": "maxpool", "_x": "skip", "_y": "m"},
                {"kind": "prelu", "_x": "c", "_y": "p", "w": [0.25]},
                {"kind": "add", "_x": "p", "_x2": "m", "_y": "out"},
            ]
        )
        self.assertEqual([o["kind"] for o in ops], ["conv", "maxpool", "add"])
        self.assertEqual((ops[0]["act"], ops[0]["_y"]), ([0.25], "p"))

    def test_prelu_stays_when_its_input_has_another_reader(self):
        ops = self.exporter(
            [
                {"kind": "conv", "_x": "in", "_y": "c"},
                {"kind": "prelu", "_x": "c", "_y": "p", "w": [0.25]},
                {"kind": "add", "_x": "p", "_x2": "c", "_y": "out"},
            ]
        )
        self.assertEqual([o["kind"] for o in ops], ["conv", "prelu", "add"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
