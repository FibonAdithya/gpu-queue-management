"""The cgroup membership test behind `gpu-claim --scope-pid`.

`proc_root` is a parameter on every function here so these run against a
real fixture directory rather than a monkeypatched `cgroup_of`. A suite
that stubbed the parser at every call site would still pass with the
parser deleted, which is the one thing these tests exist to prevent.
"""
import pytest

from gpuqueue import cgroups


def _proc(tmp_path, mapping):
    """A stand-in /proc: {pid: contents of its cgroup file}."""
    root = tmp_path / "proc"
    for pid, text in mapping.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "cgroup").write_text(text)
    return str(root)


DOCKER = "/system.slice/docker-43faa0ee4d16.scope"


def test_cgroup_of_reads_the_unified_line(tmp_path):
    root = _proc(tmp_path, {42: f"0::{DOCKER}\n"})
    assert cgroups.cgroup_of(42, root) == DOCKER


def test_cgroup_of_is_none_on_a_cgroup_v1_box(tmp_path):
    # v1 has one line per controller and no `0::`. Returning line 1
    # regardless of prefix would hand back "/docker/abc" here, which is a
    # v1 path that means nothing to `in_scope`'s v2 comparison.
    root = _proc(tmp_path, {42: "12:pids:/docker/abc\n"
                                "11:memory:/docker/abc\n"
                                "0:name=systemd:/user.slice\n"})
    assert cgroups.cgroup_of(42, root) is None


def test_cgroup_of_is_none_for_a_pid_that_is_gone(tmp_path):
    root = _proc(tmp_path, {42: f"0::{DOCKER}\n"})
    assert cgroups.cgroup_of(99999, root) is None


def test_in_scope_covers_a_nested_cgroup(tmp_path):
    # A container that makes its own sub-cgroups is still inside it.
    root = _proc(tmp_path, {42: f"0::{DOCKER}/worker\n"})
    assert cgroups.in_scope(42, DOCKER, root) is True


def test_in_scope_rejects_a_sibling_sharing_a_prefix(tmp_path):
    # `/a/bc` is not inside `/a/b`. Bare `startswith` says it is.
    root = _proc(tmp_path, {42: "0::/system.slice/bc\n"})
    assert cgroups.in_scope(42, "/system.slice/b", root) is False


def test_in_scope_is_false_for_a_scope_that_would_be_refused(tmp_path):
    # `docs/design.md` makes hand-repair of a record supported, so a
    # scope of "/" can reach this function even though `refuse_reason`
    # would never have let it be claimed. It must not then match the box.
    root = _proc(tmp_path, {42: "0::/system.slice/anything.scope\n"})
    assert cgroups.in_scope(42, "/", root) is False


def test_refuse_reason_admits_a_container_scope():
    assert cgroups.refuse_reason(DOCKER) is None


def test_refuse_reason_rejects_a_login_session():
    # Three components deep, so a depth-only check passes it -- and this
    # is the shape EVERY host shell pid resolves to, so it is the
    # likeliest accident rather than the least.
    reason = cgroups.refuse_reason(
        "/user.slice/user-0.slice/session-1848.scope")
    assert reason is not None
    assert "session" in reason


def test_refuse_reason_rejects_a_user_slice():
    assert cgroups.refuse_reason("/user.slice/user-0.slice") is not None


@pytest.mark.parametrize("scope", ["/", "/init.scope", "/system.slice",
                                   "/user.slice"])
def test_refuse_reason_rejects_top_level_scopes_without_the_message_table(
        monkeypatch, scope):
    # The refusal must come from the depth rule, not from NAMED_SCOPES.
    # That table exists only so `--scope-pid 1` says "the whole box"
    # instead of "fewer than 2 components"; emptying it must cost a good
    # message and never an exemption.
    monkeypatch.setattr(cgroups, "NAMED_SCOPES", {})
    assert cgroups.refuse_reason(scope) is not None


def test_refuse_reason_rejects_a_relative_path():
    assert cgroups.refuse_reason("system.slice/x.scope") is not None


def test_scope_process_count_counts_only_the_scope(tmp_path):
    root = _proc(tmp_path, {
        11: f"0::{DOCKER}\n",
        12: f"0::{DOCKER}/worker\n",
        13: "0::/user.slice/user-0.slice/session-1.scope\n",
    })
    assert cgroups.scope_process_count(DOCKER, root) == 2


def test_scope_process_count_is_none_when_proc_is_unreadable(tmp_path):
    # None is not zero. A count we could not take must not read as "you
    # named an empty cgroup".
    assert cgroups.scope_process_count(DOCKER, str(tmp_path / "absent")) is None
