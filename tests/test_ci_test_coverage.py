"""Guards against test files that CI does not really run.

CI enumerates test files explicitly — one ``python tests/<file>.py`` line per
test in ``.buildkite/pipeline.yml``, one ``(file, num_gpus, ...)`` tuple per
test in ``.buildkite/gpu_suites.py``. Nothing walks ``tests/`` and runs what it
finds, so a test only ever runs if its author also wired it into a pipeline
job. Two ways that goes wrong, both of which leave CI green:

1. The file is never added to a job, so it simply never runs again.
2. The file is added as a bare ``python tests/<file>.py`` line but has no
   ``if __name__ == "__main__"`` entry point. Importing a module full of
   ``test_*`` functions executes none of them, so the command exits 0 without
   asserting anything.

The tests below fail on either. :data:`NOT_RUN_IN_CI` waives (1) for files that
deliberately stay out of CI, and is itself checked for staleness so it cannot
rot: an entry must name a file that still exists and that CI still does not run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
BUILDKITE_DIR = REPO_ROOT / ".buildkite"

# CI files that name the test files they run.
CI_FILES = (
    BUILDKITE_DIR / "pipeline.yml",
    BUILDKITE_DIR / "pipeline-rocm.yaml",
    BUILDKITE_DIR / "gpu_suites.py",
)

# Directories a CI job runs wholesale (``python -m pytest tests/utils``), so
# their contents need no per-file reference.
DIRECTORIES_RUN_WHOLESALE = ("tests/utils/",)

# Test files intentionally not wired into any CI job, with the reason. Keep this
# list short: a test nobody runs is a test nobody maintains.
NOT_RUN_IN_CI: dict[str, str] = {
    "tests/test_glm52_6layer_deterministic_e2e.py": (
        "Disabled at the source: run_gate() raises "
        "'GLM-5.2 deterministic alignment is temporarily unsupported with vLLM 0.27.1'. "
        "Re-wire into gpu_suites once sparse MLA supports batch-invariant inference."
    ),
    "tests/test_glm52_layerwise_zero_e2e.py": (
        "Calls the same disabled run_gate() as test_glm52_6layer_deterministic_e2e.py."
    ),
    "tests/test_qwen2.5_0.5B_async_short.py": (
        "4-GPU end-to-end training run; not selected by any gpu_suites entry. "
        "The 'short' suite covers this shape via test_qwen2.5_0.5B_fully_async_short.py."
    ),
    "tests/test_qwen2.5_0.5B_short.py": ("4-GPU end-to-end training run; not selected by any gpu_suites entry."),
    "tests/test_qwen3_5_vl_train_rollout_e2e.py": (
        "8-GPU Qwen3.5-VL end-to-end run; not selected by any gpu_suites entry."
    ),
}


def _ci_referenced_basenames() -> set[str]:
    """Basenames of every ``test_*.py`` named anywhere in the CI definitions."""
    referenced: set[str] = set()
    for ci_file in CI_FILES:
        assert ci_file.exists(), f"CI definition {ci_file} is missing; update CI_FILES."
        text = ci_file.read_text(encoding="utf-8")
        for match in re.findall(r"[\w./-]*\btest_[\w.-]*\.py", text):
            referenced.add(match.rsplit("/", 1)[-1])
    return referenced


def _bare_python_invocations() -> list[str]:
    """Repo-relative paths CI runs as ``python <path>`` rather than via pytest."""
    text = (BUILDKITE_DIR / "pipeline.yml").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^\s*python (tests/[\w./-]+\.py)\s*$", text, re.MULTILINE)))


def _all_test_files() -> list[str]:
    """Repo-relative paths of every test file, POSIX-separated."""
    return sorted(path.relative_to(REPO_ROOT).as_posix() for path in TESTS_DIR.glob("**/test_*.py"))


def _is_covered(rel_path: str, referenced: set[str]) -> bool:
    if rel_path.startswith(DIRECTORIES_RUN_WHOLESALE):
        return True
    return rel_path.rsplit("/", 1)[-1] in referenced


@pytest.mark.unit
def test_every_test_file_runs_in_ci() -> None:
    referenced = _ci_referenced_basenames()
    missing = [
        rel_path
        for rel_path in _all_test_files()
        if not _is_covered(rel_path, referenced) and rel_path not in NOT_RUN_IN_CI
    ]
    assert not missing, (
        "These test files are not run by any CI job:\n  "
        + "\n  ".join(missing)
        + "\n\nAdd each to a job in .buildkite/pipeline.yml (CPU) or to SUITES in "
        ".buildkite/gpu_suites.py (GPU), or record it in NOT_RUN_IN_CI in this file "
        "with the reason."
    )


@pytest.mark.unit
def test_not_run_in_ci_entries_still_exist() -> None:
    deleted = sorted(rel_path for rel_path in NOT_RUN_IN_CI if not (REPO_ROOT / rel_path).exists())
    assert not deleted, (
        "NOT_RUN_IN_CI lists test files that no longer exist:\n  " + "\n  ".join(deleted) + "\n\nDrop these entries."
    )


@pytest.mark.unit
def test_not_run_in_ci_entries_are_actually_uncovered() -> None:
    referenced = _ci_referenced_basenames()
    now_covered = sorted(rel_path for rel_path in NOT_RUN_IN_CI if _is_covered(rel_path, referenced))
    assert not now_covered, (
        "NOT_RUN_IN_CI lists test files that CI now runs:\n  "
        + "\n  ".join(now_covered)
        + "\n\nDrop these entries so the waiver list stays meaningful."
    )


@pytest.mark.unit
def test_bare_python_invocations_have_main_entrypoint() -> None:
    """``python tests/foo.py`` runs nothing unless ``foo.py`` calls pytest itself."""
    no_entrypoint = []
    for rel_path in _bare_python_invocations():
        path = REPO_ROOT / rel_path
        assert path.exists(), f".buildkite/pipeline.yml runs {rel_path}, which does not exist."
        if '__name__ == "__main__"' not in path.read_text(encoding="utf-8"):
            no_entrypoint.append(rel_path)
    assert not no_entrypoint, (
        "CI runs these as `python <file>`, but they have no "
        '`if __name__ == "__main__"` block, so the command collects nothing and '
        "exits 0 without running a single test:\n  "
        + "\n  ".join(no_entrypoint)
        + '\n\nAdd `if __name__ == "__main__": raise SystemExit(pytest.main([__file__]))` '
        "to each, or invoke them as `python -m pytest <file>` in the pipeline."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
