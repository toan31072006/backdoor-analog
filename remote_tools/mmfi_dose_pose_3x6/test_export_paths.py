"""Standard-library path/preflight regression tests; no model or GPU work."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dose_export_paths", HERE / "Export-MMFiDosePose.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ExportPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dose export path test ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def args(self, *extra):
        return EXPORT.parse_args(["--root", str(self.root / "data home"), *extra])

    def fake_checkout(self, git=True):
        repo = self.root / "helper clone"
        script = repo / "remote_tools/mmfi_dose_pose_3x6/Export-MMFiDosePose.py"
        source = repo / "ATKBackd"
        if git:
            (repo / ".git").mkdir(parents=True)
        (source / "models").mkdir(parents=True)
        (source / "models/factory.py").write_text("# fixture only\n")
        return script, source

    def fake_files(self):
        paths = {k: self.root / k for k in ("source", "data", "checkpoint", "action", "output")}
        for name in EXPORT.SOURCE_FILES:
            path = paths["source"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# path fixture only; not scientific source\n")
        for csi, gt, _ in EXPORT.PAIRS:
            for rel in (csi, gt):
                path = paths["data"] / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"path fixture only; not CSI")
        for key in ("checkpoint", "action"):
            paths[key].write_bytes(b"fixture, never a trained checkpoint or action")
        return paths

    def test_missing_legacy_source_uses_helper_clone(self):
        script, source = self.fake_checkout()
        with mock.patch.object(EXPORT, "__file__", str(script)):
            paths, choice = EXPORT.resolve_paths(self.args())
        self.assertEqual(paths["source"], source)
        self.assertIn("helper Git checkout", choice)

    def test_existing_legacy_source_keeps_priority(self):
        script, _ = self.fake_checkout()
        legacy = self.root / "data home/ATKBackd"
        legacy.mkdir(parents=True)
        with mock.patch.object(EXPORT, "__file__", str(script)):
            paths, choice = EXPORT.resolve_paths(self.args())
        self.assertEqual(paths["source"], legacy)
        self.assertIn("existing data-home", choice)

    def test_explicit_source_keeps_priority_even_if_missing(self):
        script, _ = self.fake_checkout()
        explicit = self.root / "original frozen source"
        with mock.patch.object(EXPORT, "__file__", str(script)):
            paths, choice = EXPORT.resolve_paths(self.args("--source", str(explicit)))
        self.assertEqual(paths["source"], explicit)
        self.assertEqual(choice, "explicit --source")

    def test_no_implicit_fallback_to_unversioned_parent(self):
        script, _ = self.fake_checkout(git=False)
        with mock.patch.object(EXPORT, "__file__", str(script)):
            paths, choice = EXPORT.resolve_paths(self.args())
        self.assertEqual(paths["source"], self.root / "data home/ATKBackd")
        self.assertIn("missing", choice)

    def test_all_path_overrides_and_spaces_are_preserved(self):
        flags = [item for key in ("source", "checkpoint", "data", "action", "output")
                 for item in (f"--{key}", str(self.root / f"chosen {key}"))]
        paths, _ = EXPORT.resolve_paths(self.args(*flags))
        for key in ("source", "checkpoint", "data", "action", "output"):
            self.assertEqual(paths[key], self.root / f"chosen {key}")

    def test_reports_all_missing_roots_without_hashing(self):
        paths = {k: self.root / f"missing {k}" for k in
                 ("source", "checkpoint", "data", "action", "output")}
        log = io.StringIO()
        with contextlib.redirect_stdout(log), mock.patch.object(EXPORT, "sha") as hash_fn:
            self.assertFalse(EXPORT.check_paths(paths, "fixture"))
        hash_fn.assert_not_called()
        for key in ("source", "checkpoint", "data", "action"):
            self.assertIn(f"MISSING --{key}:", log.getvalue())
        self.assertFalse(paths["output"].exists())

    def test_missing_source_and_raw_sample_files_are_reported(self):
        paths = self.fake_files()
        (paths["source"] / "models/hpeli.py").unlink()
        (paths["data"] / EXPORT.PAIRS[0][0]).unlink()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertFalse(EXPORT.check_paths(paths, "fixture"))
        self.assertIn("models", log.getvalue())
        self.assertIn("hpeli.py", log.getvalue())
        self.assertIn("frame001_processed.npy", log.getvalue())

    def test_wrong_checkpoint_hash_still_refuses(self):
        paths = self.fake_files()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA256 differs"):
                EXPORT.check_paths(paths, "fixture")
        self.assertFalse(paths["output"].exists())

    def test_check_only_success_never_imports_numpy_or_torch_or_writes(self):
        paths = self.fake_files()
        argv = [item for key, path in paths.items() for item in (f"--{key}", str(path))]
        argv.append("--check-only")
        # A path fixture cannot have the real checkpoint SHA; mock only the hash.
        with mock.patch.object(EXPORT, "sha", return_value=EXPORT.SHA), \
                mock.patch("builtins.__import__", wraps=__import__) as imports, \
                contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(EXPORT.main(argv), 0)
        imported_names = [call.args[0] for call in imports.call_args_list]
        self.assertNotIn("torch", imported_names)
        self.assertNotIn("numpy", imported_names)
        self.assertIn("DOSE_POSE_PATH_CHECK PASS", log.getvalue())
        self.assertFalse(paths["output"].exists())

    def test_check_only_failure_runs_without_site_packages(self):
        out = self.root / "not created/subdir/poses.npz"
        result = subprocess.run([sys.executable, "-S", "-B", str(HERE / "Export-MMFiDosePose.py"),
            "--root", str(self.root / "missing data home"), "--output", str(out), "--check-only"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("DOSE_POSE_PATH_CHECK FAIL", result.stdout)
        self.assertIn("CHECKPOINT [MISSING]", result.stdout)
        self.assertIn("ACTION [MISSING]", result.stdout)
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertFalse(out.parent.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
