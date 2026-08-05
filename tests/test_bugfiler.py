import json
import pytest
from gpuqueue import bugfiler
from gpuqueue.config import AutofixConfig


@pytest.fixture
def cfg(tmp_path):
    return AutofixConfig(enabled=True, repo="you/gpuq",
                         state_file=tmp_path / "autofix.json")


@pytest.fixture
def gh(monkeypatch):
    """Record every gh invocation and answer from a scripted queue."""
    calls, replies = [], {}

    def fake(_cfg, args, stdin=None):
        calls.append((args, stdin))
        for prefix, out in replies.items():
            if args[:len(prefix)] == list(prefix):
                return out
        return "[]"

    monkeypatch.setattr(bugfiler, "_gh", fake)
    # ensure_labels caches "done" in a module global so a long-lived runner
    # does not re-create four labels per bug. Reset it, or every test after
    # the first sees a different call sequence.
    # raising=False because this global does not exist yet at Task 5.
    monkeypatch.setattr(bugfiler, "_labels_ensured", False, raising=False)
    fake.calls, fake.replies = calls, replies
    return fake


def test_an_open_issue_with_the_signature_is_found(gh, cfg):
    gh.replies[("issue", "list")] = json.dumps(
        [{"number": 7, "body": "sig: abc123abc123\nphase: reap"}])
    found = bugfiler.find_open_issue(cfg, "abc123abc123")
    assert found["number"] == 7


def test_a_fuzzy_search_hit_without_the_exact_line_is_rejected(gh, cfg):
    """gh's search is full-text and will return near misses. Two different
    bugs sharing one issue is worse than filing a second one."""
    gh.replies[("issue", "list")] = json.dumps(
        [{"number": 7, "body": "sig: 999999999999\nphase: reap"}])
    assert bugfiler.find_open_issue(cfg, "abc123abc123") is None


def test_no_open_issue_returns_none(gh, cfg):
    assert bugfiler.find_open_issue(cfg, "abc123abc123") is None


def test_an_open_pr_referencing_the_signature_is_found(gh, cfg):
    gh.replies[("pr", "list")] = json.dumps(
        [{"number": 12, "body": "fixes sig: abc123abc123"}])
    assert bugfiler.find_open_pr(cfg, "abc123abc123") == 12


def test_a_recently_closed_issue_is_found(gh, cfg):
    gh.replies[("issue", "list")] = json.dumps(
        [{"number": 3, "body": "sig: abc123abc123"}])
    assert bugfiler.find_recent_closed(cfg, "abc123abc123") == 3


def test_the_closed_lookup_bounds_the_window(gh, cfg):
    cfg.closed_lookback_days = 30
    bugfiler.find_recent_closed(cfg, "abc123abc123")
    args, _ = gh.calls[-1]
    search = args[args.index("--search") + 1]
    assert "closed:>=" in search


def test_the_lookups_scope_themselves_to_the_configured_repo(gh, cfg):
    bugfiler.find_open_issue(cfg, "abc123abc123")
    args, _ = gh.calls[-1]
    assert args[args.index("--repo") + 1] == "you/gpuq"


def test_the_token_is_passed_to_gh_as_gh_token(monkeypatch, cfg, tmp_path):
    """The PAT reaches gh through the environment, never through argv --
    argv is world-readable in /proc."""
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["env"] = argv, kw.get("env") or {}
        class P:
            returncode, stdout, stderr = 0, "[]", ""
        return P()

    monkeypatch.setenv("GPUQ_GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setattr(bugfiler.subprocess, "run", fake_run)
    bugfiler._gh(cfg, ["issue", "list"])
    assert seen["env"]["GH_TOKEN"] == "ghp_secret"
    assert "ghp_secret" not in " ".join(seen["argv"])


def test_a_failing_gh_raises_gh_error(monkeypatch, cfg):
    def fake_run(argv, **kw):
        class P:
            returncode, stdout, stderr = 1, "", "HTTP 401: Bad credentials"
        return P()

    monkeypatch.setattr(bugfiler.subprocess, "run", fake_run)
    with pytest.raises(bugfiler.GhError, match="401"):
        bugfiler._gh(cfg, ["issue", "list"])


def test_a_missing_gh_binary_raises_gh_error(monkeypatch, cfg):
    def fake_run(argv, **kw):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(bugfiler.subprocess, "run", fake_run)
    with pytest.raises(bugfiler.GhError, match="gh"):
        bugfiler._gh(cfg, ["issue", "list"])


def test_gpuq_commit_is_a_short_sha_or_unknown():
    got = bugfiler.gpuq_commit()
    assert got == "unknown" or (7 <= len(got) <= 12 and got.isalnum())


from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def test_a_fresh_box_may_dispatch(cfg):
    assert bugfiler.may_dispatch(cfg, NOW) is True


def test_three_dispatches_in_a_day_close_the_gate(cfg):
    for _ in range(3):
        bugfiler.record_dispatch(cfg, NOW)
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 3
    assert bugfiler.may_dispatch(cfg, NOW) is False


def test_the_window_rolls(cfg):
    bugfiler.record_dispatch(cfg, NOW - timedelta(hours=25))
    bugfiler.record_dispatch(cfg, NOW - timedelta(hours=23))
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 1
    assert bugfiler.may_dispatch(cfg, NOW) is True


def test_old_entries_are_pruned_from_the_file(cfg):
    bugfiler.record_dispatch(cfg, NOW - timedelta(days=9))
    bugfiler.record_dispatch(cfg, NOW)
    assert len(json.loads(cfg.state_file.read_text())["dispatches"]) == 1


def test_a_corrupt_state_file_does_not_stop_filing(cfg):
    """The state file is a budget guard, not queue state. If it is
    unreadable, fail towards filing -- evidence is never lost."""
    cfg.state_file.write_text("{ not json")
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 0
    assert bugfiler.may_dispatch(cfg, NOW) is True


def test_the_state_file_is_created_with_its_parents(tmp_path, cfg):
    cfg.state_file = tmp_path / "deep" / "nested" / "autofix.json"
    bugfiler.record_dispatch(cfg, NOW)
    assert cfg.state_file.exists()


def test_a_cap_of_zero_never_dispatches(cfg):
    cfg.max_dispatches_per_day = 0
    assert bugfiler.may_dispatch(cfg, NOW) is False
