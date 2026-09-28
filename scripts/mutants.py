#!/usr/bin/env python
"""Manual mutation testing: every mutant must be killed by the suite.

Each mutant is a plausible bug injected into src/loadbalancer/proxy.py, applied
one at a time. The mutant is killed when the test suite fails; afterwards the
file is restored from git, so the restore is verifiable with `git diff`.

    uv run python scripts/mutants.py

Exit code 0 = all mutants killed. Any survivor is a hole in the gauntlet.
A suite that *hangs* counts as killed too (a timeout is an observable failure),
but it is reported distinctly so a hanging test can be fixed separately.
"""

import subprocess
import sys
from pathlib import Path

TARGET = Path("src/loadbalancer/proxy.py")

# The clean suite finishes in ~1s. A mutant can make it block forever (e.g. a
# retry into an already-committed response), which would otherwise hang this
# whole script instead of reporting a result.
SUITE_TIMEOUT = 120

SESSION_PROP = """        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session"""

MUTANTS: list[tuple[str, str, str]] = [
    (
        "K1: committed-response guard dropped (retry after a stream was flushed)",
        "if committed:",
        "if False:",
    ),
    (
        "K2: content-encoding/content-length relayed again",
        'ENTITY_HEADERS = frozenset({"content-encoding", "content-length"})',
        "ENTITY_HEADERS = frozenset()",
    ),
    (
        "K2: `connection` dropped from the hop-by-hop set",
        'HOP_BY_HOP = frozenset(\n    {\n        "connection",\n',
        "HOP_BY_HOP = frozenset(\n    {\n",
    ),
    (
        "K3: query string dropped from the upstream URL",
        "p._forward(request, request.path_qs)",
        "p._forward(request, request.path)",
    ),
    (
        "K4: a fresh ClientSession per access (no pooling, no cleanup)",
        SESSION_PROP,
        "        return aiohttp.ClientSession()",
    ),
    (
        "K4: /v1/models backends queried serially",
        "results = await asyncio.gather(*(_fetch_models(p.session, url) for url in config.vllm_urls))",
        "results = [await _fetch_models(p.session, url) for url in config.vllm_urls]",
    ),
    (
        "bug: 504 body reverts to the un-set last_error ('upstream error: None')",
        'text=f"upstream error: {type(exc).__name__} ({target})"',
        'text=f"upstream error: {last_error}"',
    ),
]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run the suite; None means it exceeded SUITE_TIMEOUT (a hang)."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=SUITE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return None


def main() -> int:
    original = TARGET.read_text()
    survivors: list[str] = []
    print(f"running {len(MUTANTS)} mutants against the suite\n")
    for name, old, new in MUTANTS:
        if original.count(old) != 1:
            print(f"  PATTERN-NOT-FOUND  {name}")
            print(f"    (expected exactly 1 occurrence, got {original.count(old)})")
            survivors.append(name)
            continue
        TARGET.write_text(original.replace(old, new))
        try:
            result = run(
                ["uv", "run", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"]
            )
            if result is None:
                killed = True
                first = f"suite hung (timeout {SUITE_TIMEOUT}s) -> counted as killed"
                label = "HUNG/KILLED"
            else:
                killed = result.returncode != 0
                first = next(
                    (ln for ln in result.stdout.splitlines() if ln.startswith("FAILED")),
                    "no FAILED line",
                )
                label = "KILLED  " if killed else "SURVIVED"
            print(f"  {label}  {name}")
            print(f"            {first}")
            if not killed:
                survivors.append(name)
        finally:
            TARGET.write_text(original)
            # the restore must be verifiable without git: the file on disk has
            # to be byte-identical to what we started from
            if TARGET.read_text() != original:
                print(f"    RESTORE FAILED for {name}")
                sys.exit(2)
    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    if survivors:
        print("survivors (holes in the gauntlet):")
        for s in survivors:
            print(f"  - {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
