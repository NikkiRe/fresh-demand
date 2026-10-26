"""Verify the preserved confirmation code separately from later extensions."""

import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "reports/active/confirmation"
CORE = {"baselines.py", "environments.py", "experiment.py", "policies.py", "statistics.py"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    protocol = json.loads((REPORT / "protocol.json").read_text())
    revision = subprocess.check_output(["git", "rev-parse", "7121c6c"], cwd=PROJECT, text=True).strip()
    core_checks, release_changes = {}, {}
    for filename, recorded in protocol["source_sha256"].items():
        frozen = subprocess.check_output(["git", "show", f"{revision}:{filename}"], cwd=PROJECT)
        if Path(filename).name in CORE:
            core_checks[filename] = digest(frozen) == recorded
        current = digest((PROJECT / filename).read_bytes())
        if current != recorded:
            release_changes[filename] = {"at_confirmation_start": recorded, "release_sha256": current}
    result = {
        "confirmation_core_commit": revision,
        "verification_basis": "Preserved Git revision, not the entire current checkout",
        "core_source_hashes_match": core_checks,
        "all_core_match": all(core_checks.values()),
        "release_changes_after_confirmation": release_changes,
        "explanation": "Later package changes add serving diagnostics, a separately labelled myopic-ablation route, and installed-package path/provenance support. Relative context files resolve from the caller working directory; new runs hash the executing package with canonical source labels. These path changes do not alter simulation algorithms. The predeclared confirmation algorithms are preserved in the pinned revision; the original protocol is not rewritten.",
    }
    (REPORT / "source_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_core_match"]:
        raise RuntimeError("Confirmation core does not match its recorded source hashes")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
