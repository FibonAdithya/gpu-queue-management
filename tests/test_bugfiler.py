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
    # ensure_labels caches confirmed labels in a module global so a
    # long-lived runner does not re-create four labels per bug. Reset it,
    # or every test after the first sees a different call sequence.
    # raising=False because this global does not exist yet at Task 5.
    monkeypatch.setattr(bugfiler, "_labels_ensured", set(), raising=False)
    fake.calls, fake.replies = calls, replies
    return fake


def test_an_open_issue_with_the_signature_is_found(gh, cfg):
    gh.replies[("issue", "list")] = json.dumps(
        [{"number": 7, "body": "sig: abc123abc123\nphase: reap"}])
    found = bugfiler.find_open_issue(cfg, "abc123abc123")
    assert found["number"] == 7


def test_the_open_issue_lookup_does_not_depend_on_the_search_index(gh, cfg):
    """This is the lookup that decides whether a bug is already filed, and
    GitHub's search index is eventually consistent -- two occurrences
    inside the indexing lag would both miss it and file twice. Every issue
    this module files carries AUTO_LABEL, so listing on the label is a
    direct query rather than an index read, and `_exact` still does the
    precise matching locally. The other two lookups have no such label to
    stand on, and there a stale index costs a duplicate rather than a
    missed fix-in-flight.
    """
    bugfiler.find_open_issue(cfg, "abc123abc123")
    args, _ = gh.calls[-1]
    assert args[args.index("--label") + 1] == bugfiler.AUTO_LABEL
    assert "--search" not in args


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


def test_the_pr_lookup_searches_for_the_sig_line_like_the_other_lookups(
        gh, cfg):
    """The workflow prompt tells the fixer to copy `sig: <hex>` verbatim
    into the PR body -- it never mentions a bare hex anywhere else -- so
    the search query must look for that literal line, exactly like
    find_open_issue and find_recent_closed do, or production PRs (which
    only ever carry the prefixed line) can never match."""
    bugfiler.find_open_pr(cfg, "abc123abc123")
    args, _ = gh.calls[-1]
    search = args[args.index("--search") + 1]
    assert search == "sig: abc123abc123 in:body"


def test_a_pr_with_the_bare_hex_but_no_sig_prefix_is_rejected(gh, cfg):
    """A PR that happens to mention the hex some other way -- not the
    literal `sig: <hex>` line the fixer is told to copy -- must not be
    treated as already addressing this bug."""
    gh.replies[("pr", "list")] = json.dumps(
        [{"number": 12, "body": "relates to abc123abc123 somehow"}])
    assert bugfiler.find_open_pr(cfg, "abc123abc123") is None


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


def test_an_unset_token_is_not_replaced_by_the_boxs_own_credentials(
        monkeypatch, cfg):
    """The whole security argument for autofix is that this token is scoped
    to `issues: write` and cannot push. An unset GPUQ_GITHUB_TOKEN must
    therefore mean *no credentials*, not "whatever else this box happens to
    be authenticated as" -- a box where someone ran `gh auth login`, or
    exported a full-scope GH_TOKEN for another tool, would otherwise file
    issues as that identity with that identity's permissions, and the
    documented `cannot push` check would be describing a token that is not
    the one in use. Fail closed into the no-token path instead.
    """
    seen = {}

    def fake_run(argv, **kw):
        seen["env"] = kw.get("env") or {}
        class P:
            returncode, stdout, stderr = 0, "[]", ""
        return P()

    monkeypatch.delenv("GPUQ_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_someone_elses_full_scope_token")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_and_this_one_too")
    monkeypatch.setattr(bugfiler.subprocess, "run", fake_run)
    bugfiler._gh(cfg, ["issue", "list"])
    assert "GH_TOKEN" not in seen["env"]
    assert "GITHUB_TOKEN" not in seen["env"]


def test_the_configured_token_still_wins_over_an_ambient_one(monkeypatch, cfg):
    seen = {}

    def fake_run(argv, **kw):
        seen["env"] = kw.get("env") or {}
        class P:
            returncode, stdout, stderr = 0, "[]", ""
        return P()

    monkeypatch.setenv("GPUQ_GITHUB_TOKEN", "ghp_the_configured_one")
    monkeypatch.setenv("GH_TOKEN", "ghp_ambient")
    monkeypatch.setattr(bugfiler.subprocess, "run", fake_run)
    bugfiler._gh(cfg, ["issue", "list"])
    assert seen["env"]["GH_TOKEN"] == "ghp_the_configured_one"


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


def test_a_gh_binary_that_exists_but_is_not_executable_raises_gh_error(
        monkeypatch, cfg):
    """PermissionError is an OSError sibling of FileNotFoundError -- a gh
    that exists but lacks +x must not escape as a raw builtin exception."""
    def fake_run(argv, **kw):
        raise PermissionError("gh")

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


from gpuqueue.bugreport import CallerError
from gpuqueue.git_ops import GitError
from gpuqueue.spec import JobSpec


def _boom(exc=None):
    try:
        raise exc or GitError("git worktree add failed (128): fatal")
    except Exception as e:
        return e


def _spec():
    return JobSpec(id="j1", lane="gpu", project="p", commit="deadbeef",
                   branch="main", cmd=["python", "train.py"])


def _created(gh):
    """The body handed to `gh issue create`, or None."""
    for args, stdin in gh.calls:
        if args[:2] == ["issue", "create"]:
            return args, stdin
    return None, None


def _labels(args):
    """Every --label value passed to `gh issue create`, in order."""
    return [args[i + 1] for i, a in enumerate(args) if a == "--label"]


def test_a_disabled_config_does_nothing(gh, cfg):
    cfg.enabled = False
    assert bugfiler.file_bug(cfg, _boom(), "checkout") == "disabled"
    assert gh.calls == []


def test_a_caller_fault_never_files(gh, cfg):
    exc = _boom(CallerError("declared artifact not produced: runs/s.json"))
    assert bugfiler.file_bug(cfg, exc, "artifacts") == "caller-fault"
    assert gh.calls == []


def test_a_gpuq_fault_files_an_issue(gh, cfg):
    assert bugfiler.file_bug(cfg, _boom(), "checkout", spec=_spec()) == "filed"
    args, body = _created(gh)
    assert args is not None
    assert "sig: " in body
    assert "GitError" in body


def test_a_filed_issue_carries_the_auto_label(gh, cfg):
    bugfiler.file_bug(cfg, _boom(), "checkout")
    args, _ = _created(gh)
    assert _labels(args) == [bugfiler.AUTO_LABEL]


def test_a_filed_issue_records_a_dispatch(gh, cfg):
    bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW)
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 1


def test_a_dispatch_is_recorded_before_the_issue_is_created(gh, cfg,
                                                             monkeypatch):
    """The survivable failure must go second. If record_dispatch runs after
    _create_issue and then fails, a run has already been authorised (the
    issue exists with gpuq-auto) and the budget was never debited -- an
    unwritable state file would silently remove the cap forever. Recording
    first means a failure there can still fall back to filing throttled
    with nothing authorised."""
    order = []
    real_record = bugfiler.record_dispatch
    real_create = bugfiler._create_issue

    def spy_record(cfg, now):
        order.append("record_dispatch")
        return real_record(cfg, now)

    def spy_create(cfg, title, body, labels):
        order.append("create_issue")
        return real_create(cfg, title, body, labels)

    monkeypatch.setattr(bugfiler, "record_dispatch", spy_record)
    monkeypatch.setattr(bugfiler, "_create_issue", spy_create)
    bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW)
    assert order == ["record_dispatch", "create_issue"]


def test_a_failed_issue_creation_gives_the_dispatch_back(gh, cfg,
                                                          monkeypatch):
    """Recording first is what stops an unwritable counter authorising
    infinite runs, but it leaves the opposite hole: if `gh issue create`
    then fails, the budget was debited for a run that cannot happen,
    because no issue exists to trigger one. Three creation failures during
    a GitHub outage would otherwise spend the whole rolling-24h budget with
    nothing filed and no way to notice -- the next real bug files
    `throttled` for a cap that was never actually used. Refunding is safe
    precisely because it only ever narrows: it happens on the path where we
    know nothing was authorised.
    """
    def boom(cfg, title, body, labels):
        raise bugfiler.GhError("HTTP 502: Bad gateway")

    monkeypatch.setattr(bugfiler, "_create_issue", boom)
    with pytest.raises(bugfiler.GhError):
        bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW)
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 0


def test_a_refund_leaves_earlier_dispatches_alone(gh, cfg, monkeypatch):
    """Only the stamp this call wrote comes back, not the budget."""
    bugfiler.record_dispatch(cfg, NOW - timedelta(hours=1))

    def boom(cfg, title, body, labels):
        raise bugfiler.GhError("HTTP 502: Bad gateway")

    monkeypatch.setattr(bugfiler, "_create_issue", boom)
    with pytest.raises(bugfiler.GhError):
        bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW)
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 1


def test_an_unwritable_state_file_files_throttled_instead_of_dispatching(
        gh, cfg, monkeypatch):
    """A broken counter must degrade to no-dispatch, never to
    infinite-dispatch. Evidence (the issue) is still filed either way --
    what must not happen is the run being authorised while the budget
    silently stops moving."""
    def boom(cfg, now):
        raise OSError("state dir is not writable")

    monkeypatch.setattr(bugfiler, "record_dispatch", boom)
    outcome = bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW)
    assert outcome == "filed-throttled"
    args, _ = _created(gh)
    assert args is not None  # evidence still files
    # Both labels: AUTO_LABEL too, or a `label:gpuq-auto` triage query
    # misses this bug entirely even though it is the same structural
    # evidence as a dispatched one -- only the budget held it back.
    assert set(_labels(args)) == {bugfiler.AUTO_LABEL, bugfiler.THROTTLED_LABEL}


def test_a_missing_state_file_path_files_throttled_rather_than_crashing(
        gh, cfg):
    """Path(None) raises TypeError. A config with autofix enabled but no
    state_file (exactly what tests/test_runner.py::_enable builds) must
    still file the issue, just without a working dispatch cap."""
    cfg.state_file = None
    outcome = bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW)
    assert outcome == "filed-throttled"
    args, _ = _created(gh)
    assert args is not None
    assert set(_labels(args)) == {bugfiler.AUTO_LABEL, bugfiler.THROTTLED_LABEL}


def test_an_existing_open_issue_is_commented_not_refiled(gh, cfg):
    sig_body = "sig: {sig}\noccurrences: 1\nfirst seen: x\nlast seen: x"

    exc = _boom()
    from gpuqueue.bugreport import signature
    sig = signature(exc, "checkout")
    gh.replies[("issue", "list")] = json.dumps(
        [{"number": 7, "body": sig_body.format(sig=sig)}])

    assert bugfiler.file_bug(cfg, exc, "checkout") == "commented-issue"
    assert _created(gh)[0] is None
    assert any(a[:2] == ["issue", "comment"] for a, _ in gh.calls)


def test_commenting_bumps_the_occurrence_count_in_the_body(gh, cfg):
    from gpuqueue.bugreport import signature
    exc = _boom()
    sig = signature(exc, "checkout")
    gh.replies[("issue", "list")] = json.dumps(
        [{"number": 7, "body": f"sig: {sig}\noccurrences: 1\n"
                               "first seen: a\nlast seen: a"}])
    bugfiler.file_bug(cfg, exc, "checkout")
    edits = [stdin for a, stdin in gh.calls if a[:2] == ["issue", "edit"]]
    assert edits and "occurrences: 2" in edits[0]


def test_a_recurrence_does_not_burn_a_dispatch(gh, cfg):
    from gpuqueue.bugreport import signature
    exc = _boom()
    gh.replies[("issue", "list")] = json.dumps(
        [{"number": 7, "body": f"sig: {signature(exc, 'checkout')}\n"
                               "occurrences: 1\nfirst seen: a\nlast seen: a"}])
    bugfiler.file_bug(cfg, exc, "checkout", now=NOW)
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 0


def test_an_open_pr_takes_the_comment_instead(gh, cfg):
    """One fix run per bug, not one per job the bug kills."""
    from gpuqueue.bugreport import signature
    exc = _boom()
    sig = signature(exc, "checkout")
    gh.replies[("pr", "list")] = json.dumps(
        [{"number": 12, "body": f"addresses sig: {sig}"}])
    assert bugfiler.file_bug(cfg, exc, "checkout") == "commented-pr"
    assert any(a[:2] == ["pr", "comment"] for a, _ in gh.calls)


def test_past_the_cap_the_issue_still_files_but_is_labelled_throttled(gh, cfg):
    for _ in range(3):
        bugfiler.record_dispatch(cfg, NOW)
    assert bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW) \
        == "filed-throttled"
    args, _ = _created(gh)
    assert set(_labels(args)) == {bugfiler.AUTO_LABEL, bugfiler.THROTTLED_LABEL}


def test_a_throttled_file_does_not_record_a_dispatch(gh, cfg):
    for _ in range(3):
        bugfiler.record_dispatch(cfg, NOW)
    bugfiler.file_bug(cfg, _boom(), "checkout", now=NOW)
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 3


def test_a_previously_fixed_bug_says_the_fix_did_not_hold(gh, cfg,
                                                          monkeypatch):
    """The closed lookup answers with the same `issue list` verb as the open
    one, so this test needs a fake that reads the query, not just the verb."""
    from gpuqueue.bugreport import signature
    exc = _boom()
    sig = signature(exc, "checkout")

    def fake(_cfg, args, stdin=None):
        gh.calls.append((args, stdin))
        if args[:2] == ["issue", "list"] and "--state" in args \
                and args[args.index("--state") + 1] == "closed":
            return json.dumps([{"number": 3, "body": f"sig: {sig}"}])
        return "[]"

    monkeypatch.setattr(bugfiler, "_gh", fake)
    bugfiler.file_bug(cfg, exc, "checkout")
    _, body = _created(gh)
    assert "#3" in body and "did not hold" in body


def test_an_agent_report_is_labelled_reported_and_carries_the_prose(gh, cfg):
    gh.replies[("issue", "create")] = \
        "https://github.com/you/gpuq/issues/42\n"
    number = bugfiler.file_agent_report(cfg, "gpuq wait hangs", "it hangs")
    args, body = _created(gh)
    assert args[args.index("--label") + 1] == bugfiler.REPORTED_LABEL
    assert "it hangs" in body
    assert number == 42


def test_an_agent_report_carries_no_signature(gh, cfg):
    """No traceback, so no signature -- and therefore no automatic dedup.
    The owner closes duplicates when they add `fix-me`."""
    gh.replies[("issue", "create")] = \
        "https://github.com/you/gpuq/issues/42\n"
    bugfiler.file_agent_report(cfg, "t", "b")
    _, body = _created(gh)
    assert "sig: " not in body


def test_a_failed_label_creation_is_retried_not_cached_as_done(gh, cfg,
                                                                monkeypatch):
    """A transient failure creating one label (auth hiccup, rate limit)
    must not be remembered as success for the whole set. If it were, every
    later file_bug/file_agent_report in this process's life would raise
    GhError on `gh issue create --label` refusing a label gh never actually
    made -- and since the runner swallows GhError, filing would silently
    stop for the rest of the process."""
    attempts = {"n": 0}

    def fake(_cfg, args, stdin=None):
        gh.calls.append((args, stdin))
        if args[:2] == ["label", "create"] and args[2] == bugfiler.AUTO_LABEL:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise bugfiler.GhError("rate limited")
        return "[]"

    monkeypatch.setattr(bugfiler, "_gh", fake)
    bugfiler.ensure_labels(cfg)
    bugfiler.ensure_labels(cfg)

    creates = [a for a, _ in gh.calls if a[:2] == ["label", "create"]]
    auto_creates = [a for a in creates if a[2] == bugfiler.AUTO_LABEL]
    other_creates = [a for a in creates if a[2] != bugfiler.AUTO_LABEL]
    # The failed label is retried on the second call...
    assert len(auto_creates) == 2
    # ...but labels that succeeded the first time are not recreated.
    assert len(other_creates) == 3


def test_ensuring_labels_does_not_overwrite_what_an_owner_edited(gh, cfg):
    """`gh label create --force` *edits* a label that already exists. The
    labels here are triage furniture the owner lives with, so recolouring
    `throttled` or rewriting its description by hand must survive -- under
    --force it was silently reset on every runner restart, which reads as
    the queue fighting you. Creation stays idempotent by treating "already
    exists" as success (below), not by overwriting.
    """
    bugfiler.ensure_labels(cfg)
    creates = [a for a, _ in gh.calls if a[:2] == ["label", "create"]]
    assert creates, "labels are still created on a fresh repo"
    assert not any("--force" in a for a in creates)


def test_a_label_that_already_exists_counts_as_ensured(gh, cfg, monkeypatch):
    """Without --force, gh fails on an existing label. That is the normal
    steady state of every box after the first bug, and it must not mean
    four failed `gh` calls per bug forever."""
    def fake(_cfg, args, stdin=None):
        gh.calls.append((args, stdin))
        if args[:2] == ["label", "create"]:
            raise bugfiler.GhError(
                "HTTP 422: Validation Failed (already exists)")
        return "[]"

    monkeypatch.setattr(bugfiler, "_gh", fake)
    bugfiler.ensure_labels(cfg)
    bugfiler.ensure_labels(cfg)
    creates = [a for a, _ in gh.calls if a[:2] == ["label", "create"]]
    assert len(creates) == len(bugfiler._LABEL_SPECS)


def test_a_non_iterable_dispatches_value_does_not_stop_filing(cfg):
    """{"dispatches": 5} is malformed, not merely unreadable -- the loop in
    _load_dispatches must not escape as a bare TypeError trying to iterate
    an int. The state file is a budget guard, not queue state: malformed
    means zero recorded dispatches, not a raise."""
    cfg.state_file.write_text(json.dumps({"dispatches": 5}))
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 0
    assert bugfiler.may_dispatch(cfg, NOW) is True


def test_a_naive_timestamp_does_not_crash_the_comparison(cfg):
    """A hand-edited file, or a future writer passing a naive `now`, can
    leave one timestamp without tzinfo. Comparing it against an aware `now`
    must not raise -- naive timestamps are treated as UTC, the only zone
    _save_dispatches ever writes."""
    from datetime import datetime as dt
    naive = dt(2026, 8, 5, 11, 0).isoformat()
    aware = (NOW - timedelta(hours=1)).isoformat()
    cfg.state_file.write_text(json.dumps({"dispatches": [naive, aware]}))
    assert bugfiler.dispatches_in_last_day(cfg, NOW) == 2
    assert bugfiler.may_dispatch(cfg, NOW) is True
