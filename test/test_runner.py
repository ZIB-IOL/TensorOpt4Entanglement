"""Runner integration tests with local Julia/Slurm stubs; submits no real jobs.

Run with: python3 -m unittest discover -s test -p test_runner.py
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pdgr-runner-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("runjobs.sh", "run.slurm", "Project.toml", "Manifest.toml",
                     "scripts/lib.sh", "scripts/exp_main.sh", "scripts/check_env.sh",
                     "lib/PDGR/PDGR.jl"):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, dest)
        shutil.copytree(REPO / "benchmark", self.root / "benchmark")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.jsonl"
        self.julia = self.bin / "julia"
        self.julia.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['RUNNER_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if '--version' in sys.argv:
    print('julia version 1.11.4')
elif any('PDGR.solve' in arg for arg in sys.argv):
    print('PDGR_OK')
""")
        self.julia.chmod(0o755)
        srun = self.bin / "srun"
        srun.write_text('#!/usr/bin/env bash\nexec "$@"\n')
        srun.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        JULIA_BIN=str(self.julia), RUNNER_TEST_CALLS=str(self.calls),
                        MOSEKLM_LICENSE_FILE=str(self.root / "no-license"),
                        JOBLIST_STAMP="test")
        for key in ("RESULTS_DIR", "TRACE_DIR", "LOG_DIR", "PART", "M", "FORCE",
                    "DRY_RUN", "USE_SLURM", "SLURM_ARRAY_TASK_ID", "JOB_LIST"):
            self.env.pop(key, None)

    def run_script(self, script, *args, env=None):
        result = subprocess.run(["bash", script, *args], cwd=self.root,
                                env=self.env if env is None else env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def recorded_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_dry_run_lists_all_pdgr_states_without_installation(self):
        output = self.run_script("runjobs.sh", "--dry-run", "main", "--algo", "PDGR",
                                 "--force", "--seed", "7", "--pdgr-max-steps", "2")
        self.assertEqual(output.count("  MISSING "), 11)
        self.assertNotIn("instantiating", output)
        self.assertTrue(all(call == ["--version"] for call in self.recorded_calls()))

    def test_local_installation_uses_selected_julia_and_forwards_options(self):
        env = dict(self.env, JULIA_BIN="/nonexistent/julia")
        self.run_script("runjobs.sh", "--julia", str(self.julia), "--local", "main",
                        "--algo", "PDGR", "--state", "state_13.jl", "-t", "0",
                        "--seed", "17", "--pdgr-mode", "sep", "--pdgr-max-steps", "2",
                        env=env)
        calls = self.recorded_calls()
        self.assertIn("Pkg.instantiate()", calls[0][-1])
        job = next(call for call in calls if "scripts/run_experiment.jl" in call)
        self.assertEqual(job[-6:], ["--seed", "17", "--pdgr-mode", "sep",
                                    "--pdgr-max-steps", "2"])
        self.assertIn("state_13.jl", job)

    def test_default_main_batch_includes_pdgr(self):
        output = self.run_script("runjobs.sh", "--dry-run", "main", "--force",
                                 "--state", "state_0.jl")
        self.assertEqual(output.count("  MISSING "), 7)
        self.assertIn("  MISSING state_0.jl PDGR", output)

    def test_local_ir_forwards_manopt_and_ladmm_options(self):
        options = ["--heur-manopt-maxiter", "40",
                   "--heur-ladmm-maxiter", "40", "--heur-ladmm1-maxiter", "40",
                   "--heur-ladmm-penalty-update", "balance",
                   "--heur-ladmm-conjugates", "true",
                   "--cp-real-master", "true", "--cp-rounds-per-ir", "3",
                   "--cp-certify-every", "4", "--ir-refit-scalar", "true",
                   "--maxnnodes", "7", "--maxeffortnnodes", "15",
                   "--heur-sbb-restarts", "4", "--heur-sbb-maxiter", "200",
                   "--heur-sbb-node-restarts", "10"]
        self.run_script("runjobs.sh", "--local", "main", "--algo", "IR",
                        "--state", "state_13.jl", "-t", "5", *options)
        job = next(call for call in self.recorded_calls() if "scripts/run_experiment.jl" in call)
        self.assertEqual(job[-len(options):], options)
        self.run_script("scripts/exp_main.sh", "--slurm", "--algo", "IR",
                        "--state", "state_13.jl", "-t", "5", *options)
        job_list = self.root / "joblists/main-test.txt"
        self.assertEqual(job_list.read_text().split()[4:], options)
        self.run_script("run.slurm", "1", env=dict(self.env, JOB_LIST=str(job_list)))
        self.assertEqual(self.recorded_calls()[-1][-len(options):], options)

    def test_slurm_worker_preserves_options_and_accepts_old_job_lists(self):
        options = ["--seed", "9", "--pdgr-fw-epsilon", "1e-8",
                   "--pdgr-witness-max-length", "100000"]
        self.run_script("scripts/exp_main.sh", "--slurm", "--algo", "PDGR",
                        "--state", "state_13.jl", "-t", "5", *options)
        job_list = self.root / "joblists/main-test.txt"
        fields = job_list.read_text().split()
        self.assertEqual(fields[4:], options)
        self.run_script("run.slurm", "1", env=dict(self.env, JOB_LIST=str(job_list)))
        self.assertEqual(self.recorded_calls()[-1][-len(options):], options)
        for suffix in ("", " " + str(self.root / "old-results")):
            job_list.write_text("state_033.jl DDPS+ 5" + suffix + "\n")
            self.run_script("run.slurm", "1", env=dict(self.env, JOB_LIST=str(job_list)))
            self.assertEqual(self.recorded_calls()[-1][-6:],
                             ["-s", "state_033.jl", "-a", "DDPS+", "-t", "5"])

    def test_pdgr_environment_check_skips_license_validation(self):
        output = self.run_script("runjobs.sh", "--env", "--algo", "PDGR")
        self.assertIn("PDGR certified a product state without MOSEK", output)
        self.assertNotIn("== mosek ==", output)
        self.assertFalse(any("using JuMP, MosekTools" in arg
                             for call in self.recorded_calls() for arg in call))

    def test_mixed_algorithm_check_still_requires_mosek(self):
        result = subprocess.run(["bash", "runjobs.sh", "--env", "--algo", "PDGR",
                                 "--algo", "DDPS+"], cwd=self.root, env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertIn("licence file unreadable", result.stdout)
        self.assertIn("== mosek ==", result.stdout)


if __name__ == "__main__":
    unittest.main()
