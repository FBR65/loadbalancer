"""SC-6: the container image builds, runs unprivileged and stays healthy."""

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"
TEXT = DOCKERFILE.read_text()


def test_dockerfile_exists_and_pins_the_base_image() -> None:
    assert DOCKERFILE.is_file()
    match = re.search(r"^FROM python:([\d.]+)-slim$", TEXT, re.MULTILINE)
    assert match, "the base image must be pinned, not `latest`"
    assert match.group(1) == "3.12", "pyproject requires >=3.12"


def test_the_image_runs_as_a_non_root_user() -> None:
    user = re.search(r"^USER\s+(\S+)$", TEXT, re.MULTILINE)
    assert user, "no USER instruction: the image would run as root"
    assert user.group(1) != "root"
    create = re.search(r"useradd[^\n]*--uid\s+(\d+)", TEXT)
    assert create, "the unprivileged user should get an explicit non-0 uid"


def _instruction(name: str) -> str:
    """The full logical instruction, joining `\\`-continued lines."""
    lines: list[str] = []
    for line in TEXT.splitlines():
        if lines or line.startswith(f"{name} "):
            lines.append(line.rstrip())
            if not line.endswith("\\"):
                break
    assert lines, f"no {name} instruction"
    return " ".join(part.strip().rstrip("\\").strip() for part in lines)


def test_the_image_declares_a_healthcheck_against_liveness() -> None:
    check = _instruction("HEALTHCHECK")
    assert "/health" in check, "the probe must hit the liveness endpoint"
    assert "/ready" not in check, (
        "restarting cannot repair a dead backend; probing readiness would restart-loop"
    )


def test_the_image_installs_the_system_trust_store() -> None:
    """The CA bundle was removed; HTTPS to backends needs the system roots."""
    assert "ca-certificates" in TEXT
    assert "itzbund" not in TEXT.lower()


def test_the_entrypoint_is_the_console_script() -> None:
    entry = re.search(r'^ENTRYPOINT\s+(\["[^]]*\])$', TEXT, re.MULTILINE)
    assert entry, "no ENTRYPOINT"
    assert '"loadbalancer"' in entry.group(1), (
        "the entry point must be the console script declared in pyproject"
    )


def test_no_secrets_are_baked_into_the_image() -> None:
    for line in TEXT.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("ENV ", "ARG ")):
            continue
        for pair in stripped.split()[1:]:
            name, _, value = pair.partition("=")
            if name in ("AUTH_TOKEN", "LISTEN_PORT", "CONFIG_DIR"):
                assert not value, f"{name} must not be baked in: {stripped}"
