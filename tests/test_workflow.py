"""Structural guards on `.github/workflows/autofix.yml`.

Nothing here runs the workflow. These are the properties whose failure
nobody is present to see. A gate that never opens still reports success
(the job is `skipped`, and a run full of skipped jobs is a green run), and
a trigger that hands the OAuth token to fork-authored code looks exactly
like one that does not.

The missing-permission check is the odd one out: it fails loudly rather
than silently. It is here because of *when* it is heard -- after checkout,
after install, after the suite has gone green, on an unattended box that
has just hit a bug. A workflow this rarely exercised gets one chance to
work, and none of these three failures gets a second look until then.

Both checks are scoped to the block they are about rather than the whole
file, so the surrounding comments stay free to name the thing they are
warning about. A guard that forces its own documentation to talk around
its subject is a guard that will be deleted by whoever needs that
documentation most.
"""
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/autofix.yml"


def _block(text: str, opener: str) -> str:
    """The lines under `opener`, sliced by indentation."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.rstrip() == opener)
    indent = len(lines[start]) - len(lines[start].lstrip())
    out = []
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


def gate_expression(text: str) -> str:
    """The job's `if:` expression, comments excluded."""
    return "\n".join(line for line in _block(text, "    if: >-").splitlines()
                     if not line.lstrip().startswith("#"))


def test_the_off_switch_cannot_swallow_an_unset_variable():
    """An unset GPUQ_AUTOFIX must mean "on", and once did not.

    GitHub's expressions use loose equality and coerce anything numeric-
    looking to a number. An unset variable is the empty string, which
    coerces to zero, so a bare zero in the off-list matched it -- verified
    live, with the variable unset, against a real runner. Dispatch was
    disabled on every repo that had never touched the variable, which is
    every repo by default, while filing kept working. It read as "the
    fixer is broken", not "the gate is shut".

    The prefixes stop the coercion. If you are here because this failed,
    the gate is only correct while both sides are non-numeric.
    """
    gate = gate_expression(WORKFLOW.read_text())
    assert "format('x{0}', vars.GPUQ_AUTOFIX)" in gate, (
        "the off switch must compare a prefixed value; a bare "
        "vars.GPUQ_AUTOFIX lets an unset variable coerce to zero and match")
    assert '"x0"' in gate, "the off-list entries must carry the same prefix"
    assert '"0"' not in gate.replace('"x0"', ""), (
        "a bare zero entry in the off-list matches an unset variable and "
        "disables dispatch on every repo that never set it")


def test_the_workflow_never_triggers_on_pull_request_target():
    """The OAuth token must stay unreachable from fork-authored code."""
    assert "pull_request_target" not in _block(WORKFLOW.read_text(), "on:"), (
        "pull_request_target runs fork-authored code with access to "
        "secrets, and this workflow holds CLAUDE_CODE_OAUTH_TOKEN")


def test_the_fixer_can_mint_an_oidc_token():
    """claude-code-action authenticates by exchanging a GitHub OIDC token.

    Without `id-token: write` it cannot mint one, and the run dies at the
    action with "Could not fetch an OIDC token" -- having already spent a
    checkout, an install and a full test run getting there. Observed: the
    first real dispatch this system ever performed failed exactly here.

    The permission grants no access to repository contents. It lets the job
    prove to a third party which workflow and repository it is.
    """
    permissions = _block(WORKFLOW.read_text(), "permissions:")
    assert "id-token: write" in permissions, (
        "claude-code-action cannot authenticate without id-token: write")
