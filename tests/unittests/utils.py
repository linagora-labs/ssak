import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


class Test(unittest.TestCase):
    def setUp(self):
        # print("Running", self.__class__.__name__)
        os.environ["DATAPATH"] = self.get_data_path()
        self.maxDiff = None
        self.createdReferences = []

    def tearDown(self):
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        self.assertEqual(
            self.createdReferences,
            [],
            "Created references: " + ", ".join(self.createdReferences).replace(self.get_data_path() + "/", ""),
        )

    def get_data_path(self, fn=None, check=True):
        return self._get_path("tests/data", fn, check)

    def get_lib_path(self, fn=None):
        return self._get_path("ssak", fn)

    def get_tool_path(self, fn=None):
        return self._get_path("tools", fn)

    def get_temp_path(self, fn=None):
        tmpdir = tempfile.gettempdir()
        if fn:
            tmpdir = os.path.join(tmpdir, fn)
        return tmpdir

    def get_output_path(self, fn=None):
        if fn == None:
            return tempfile.tempdir
        return os.path.join(tempfile.tempdir, fn)

    def loosehash(self, obj):
        # Hash the approximation of an object into a deterministic string
        return self.hash(self.loose(obj))

    def hash(self, obj):
        # Hash an object into a deterministic string
        return hashlib.md5(pickle.dumps(obj)).hexdigest()

    def loose(self, obj, t=None):
        # Return an approximative value of an object
        if isinstance(obj, list):
            try:
                stats = {"mean": np.mean(obj), "std": np.std(obj)}
            except TypeError:
                return [self.loose(a) for a in obj]
            return self.loose({"type": t if t else "list", "len": len(obj)} | stats)
        if isinstance(obj, float):
            f = round(obj, 3)
            return 0.0 if f == -0.0 else f
        if isinstance(obj, dict):
            return {k: self.loose(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self.loose(list(obj)))
        if isinstance(obj, set):
            return self.loose(list(obj), "set")
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return self.loose(obj.tolist())
        return obj

    def assertClose(self, first, second, msg=None):
        return self.assertEqual(self.loose(first), self.loose(second), msg=f"{first} != {second}" if msg is None else msg)

    def assertRun(self, cmd):
        if isinstance(cmd, str):
            return self.assertRun(cmd.split())
        curdir = os.getcwd()
        os.chdir(tempfile.tempdir)
        if cmd[0].endswith(".py"):
            cmd = [sys.executable] + cmd
        print("Running:", " ".join(cmd))
        p = subprocess.Popen(
            cmd,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),  # Otherwise ".local" path might be missing
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.chdir(curdir)
        (stdout, stderr) = p.communicate()
        self.assertEqual(p.returncode, 0, msg=stderr.decode("utf-8"))
        return stdout.decode("utf-8")

    def assertNonRegression(self, content, reference_name, process=None, process_reference_lines=None):
        """
        Check that a file/folder is the same as a reference file/folder.
        """
        self.assertTrue(os.path.exists(content))

        reference = self._get_path("tests/expected", reference_name, check=False)
        is_file = os.path.isfile(reference) if os.path.exists(reference) else os.path.isfile(content)
        if not os.path.exists(reference):
            self.assertTrue(not process_reference_lines)
            dirname = os.path.dirname(reference)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            if is_file:
                shutil.copyfile(content, reference)
            else:
                shutil.copytree(content, reference)
            self.createdReferences.append(reference)

        if is_file:
            self.assertTrue(os.path.isfile(content))
            self._check_file_non_regression(content, reference, process=process, process_reference_lines=process_reference_lines)
        else:
            self.assertTrue(os.path.isdir(content))
            for root, dirs, files in os.walk(content):
                for f in files:
                    f_ref = os.path.join(reference, f)
                    self.assertTrue(os.path.isfile(f_ref), f"Additional file: {f} in {reference_name}")
                    self._check_file_non_regression(os.path.join(root, f), f_ref, process=process, process_reference_lines=process_reference_lines)
            for root, dirs, files in os.walk(reference):
                for f in files:
                    f = os.path.join(content, f)
                    self.assertTrue(os.path.isfile(f), f"Missing file: {f} in {reference_name}")

    def _check_file_non_regression(self, file, reference, process, process_reference_lines):
        # TODO: handle numeric difference (by correctly loading json files...)
        with open(file) as f:
            content = f.readlines()
        with open(reference) as f:
            reference_content = f.readlines()
        if process_reference_lines:
            reference_content = [process_reference_lines(l) for l in reference_content]
        if process:
            reference_content = [process(l) for l in reference_content]
            content = [process(l) for l in content]
        self.assertEqual(content, reference_content, msg=f"File {file} does not match reference {reference}")

    def _get_path(self, prefix, fn=None, check=True):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), prefix)
        if fn:
            path = os.path.join(path, fn)
        if check:
            self.assertTrue(os.path.exists(path), f"Cannot find {path}")
        return path
