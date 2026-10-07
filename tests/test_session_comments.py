"""End-to-end tests for session review comments.

Review comments are registered on JSON Pointer locations of an
un-archived session; the value at the location is snapshotted at
registration time.  A batch or JSONPath-replacement commit that changes
the value a pending comment points to durably marks the comment stale
(the flag never clears automatically, even if the value later returns).
Comments are processed with an expected revision; stale comments require
explicit confirmation.  A session cannot be archived until every comment
is resolved.  Registering/processing comments adds no revision and no
change history, and comments of an archived session can no longer change.
"""
from __future__ import annotations

import json
import multiprocessing

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Operation
from app.sessions import (
    OpenReviewCommentsError,
    ReviewCommentValidationError,
    SessionArchivedError,
    RevisionConflictError,
    SessionStore,
)


@pytest.fixture()
def client(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    return TestClient(app)


def op(actor, seq, version, path, action, value=None, index=None):
    body = {
        "actor": actor,
        "seq": seq,
        "version": version,
        "path": path,
        "action": action,
        "value": value,
    }
    if index is not None:
        body["index"] = index
    return body


def create_session(client, document=None, version=None):
    resp = client.post(
        "/sessions", json={"document": document, "version": version or {}}
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


def submit(client, session_id, expected_revision, operations):
    return client.post(
        f"/sessions/{session_id}/batches",
        json={"expected_revision": expected_revision, "operations": operations},
    )


def replace(client, session_id, expected_revision, actor, path, value):
    return client.post(
        f"/sessions/{session_id}/replace",
        json={
            "expected_revision": expected_revision,
            "actor": actor,
            "path": path,
            "value": value,
        },
    )


def archive(client, session_id, expected_revision):
    return client.post(
        f"/sessions/{session_id}/archive",
        json={"expected_revision": expected_revision},
    )


def register(client, session_id, comments):
    return client.post(
        f"/sessions/{session_id}/comments", json={"comments": comments}
    )


def resolve(client, session_id, comment_id, expected_revision, confirm_stale=False):
    return client.post(
        f"/sessions/{session_id}/comments/{comment_id}/resolve",
        json={
            "expected_revision": expected_revision,
            "confirm_stale": confirm_stale,
        },
    )


def get(client, session_id):
    return client.get(f"/sessions/{session_id}")


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_sessions_without_comments_read_back_empty_list(client):
    sid = create_session(client, {"a": 1})
    state = get(client, sid).json()
    assert state["comments"] == []


def test_register_comments_snapshots_path_values_without_revision_bump(client):
    sid = create_session(client, {"title": "draft", "items": [10, 20], "n": None})
    resp = register(
        client,
        sid,
        [
            {"path": "/title", "comment": "capitalize?"},
            {"path": "/items/1", "comment": "why 20?"},
            {"path": "/n", "comment": "explicit null stays a value"},
            {"path": "", "comment": "whole document"},
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["revision"] == 0
    assert [c["status"] for c in body["comments"]] == [
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert [c["value"] for c in body["comments"]] == [
        "draft",
        20,
        None,
        {"title": "draft", "items": [10, 20], "n": None},
    ]
    assert [c["revision"] for c in body["comments"]] == [0, 0, 0, 0]
    assert all(len(c["comment_id"]) == 32 for c in body["comments"])
    assert all(c["stale_revision"] is None for c in body["comments"])
    assert all(c["resolved_revision"] is None for c in body["comments"])
    # no revision, no log, no history from registration
    state = get(client, sid).json()
    assert state["revision"] == 0
    assert state["document"] == {"title": "draft", "items": [10, 20], "n": None}
    assert [c["path"] for c in state["comments"]] == [
        "/title",
        "/items/1",
        "/n",
        "",
    ]


def test_register_missing_path_is_422_and_persists_nothing(client):
    sid = create_session(client, {"a": 1})
    resp = register(client, sid, [{"path": "/missing", "comment": "x"}])
    assert resp.status_code == 422
    assert "does not exist" in resp.json()["detail"]
    # a bad item in a multi-item batch rolls back the whole request
    resp = register(
        client,
        sid,
        [
            {"path": "/a", "comment": "ok"},
            {"path": "/missing", "comment": "bad"},
        ],
    )
    assert resp.status_code == 422
    assert get(client, sid).json()["comments"] == []


def test_register_null_value_path_is_allowed(client):
    sid = create_session(client, {"n": None})
    resp = register(client, sid, [{"path": "/n", "comment": "null ok"}])
    assert resp.status_code == 200
    assert resp.json()["comments"][0]["value"] is None


def test_register_invalid_pointer_is_422(client):
    sid = create_session(client, {})
    for bad in ("/bad~2", "noslash", "/a/~/b"):
        resp = register(client, sid, [{"path": bad, "comment": "x"}])
        assert resp.status_code == 422, bad


def test_register_validation_limit(client):
    sid = create_session(client, {"k": 0})
    empty = client.post(
        f"/sessions/{sid}/comments", json={"comments": []}
    )
    assert empty.status_code == 422
    too_many = client.post(
        f"/sessions/{sid}/comments",
        json={"comments": [{"path": "/k", "comment": "c"} for _ in range(51)]},
    )
    assert too_many.status_code == 422


def test_register_on_unknown_session_is_404(client):
    missing = "0" * 32
    resp = register(client, missing, [{"path": "/x", "comment": "c"}])
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# stale marking on commits
# ---------------------------------------------------------------------------


def test_changing_reviewed_value_marks_comment_stale_and_commit_succeeds(client):
    sid = create_session(client, {"title": "draft", "other": 0})
    cid = register(client, sid, [{"path": "/title", "comment": "?"}]).json()[
        "comments"
    ][0]["comment_id"]

    resp = submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/title", "set", "new")]
    )
    assert resp.status_code == 200
    assert resp.json()["revision"] == 1

    state = get(client, sid).json()
    note = state["comments"][0]
    assert note["comment_id"] == cid
    assert note["status"] == "stale"
    assert note["stale_revision"] == 1
    assert note["resolved_revision"] is None
    # the snapshotted value is still available
    assert note["value"] == "draft"


def test_unrelated_commit_leaves_comment_pending(client):
    sid = create_session(client, {"title": "draft", "other": 0})
    register(client, sid, [{"path": "/title", "comment": "?"}])

    resp = submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/other", "set", 1)]
    )
    assert resp.status_code == 200
    note = get(client, sid).json()["comments"][0]
    assert note["status"] == "pending"
    assert note["stale_revision"] is None


def test_setting_same_value_leaves_comment_pending(client):
    sid = create_session(client, {"title": "draft"})
    register(client, sid, [{"path": "/title", "comment": "?"}])
    # a commit that stores an operation but leaves the visible document
    # unchanged must not mark the comment stale
    resp = submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/title", "set", "draft")],
    )
    assert resp.status_code == 200
    assert get(client, sid).json()["comments"][0]["status"] == "pending"


def test_stale_flag_is_sticky_when_value_returns(client):
    sid = create_session(client, {"title": "draft"})
    register(client, sid, [{"path": "/title", "comment": "?"}])
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/title", "set", "new")])
    assert get(client, sid).json()["comments"][0]["status"] == "stale"
    # restore the original value: the comment must not auto-recover
    submit(
        client,
        sid,
        1,
        [op("bob", 1, {"alice": 1, "bob": 1}, "/title", "set", "draft")],
    )
    note = get(client, sid).json()["comments"][0]
    assert note["status"] == "stale"
    assert note["stale_revision"] == 1
    assert get(client, sid).json()["document"]["title"] == "draft"


def test_deleting_reviewed_location_marks_stale_missing_vs_null(client):
    sid = create_session(client, {"n": None})
    register(client, sid, [{"path": "/n", "comment": "?"}])
    resp = submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "delete")]
    )
    assert resp.status_code == 200
    note = get(client, sid).json()["comments"][0]
    assert note["status"] == "stale"  # null present -> missing is a change


def test_setting_null_when_missing_marks_stale(client):
    sid = create_session(client, {})
    # cannot register a missing path
    bad = register(client, sid, [{"path": "/n", "comment": "?"}])
    assert bad.status_code == 422
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", None)])
    # now the explicit-null location exists and can be reviewed
    ok = register(client, sid, [{"path": "/n", "comment": "?"}])
    assert ok.status_code == 200
    cid = ok.json()["comments"][0]["comment_id"]
    submit(
        client,
        sid,
        1,
        [op("bob", 1, {"alice": 1, "bob": 1}, "/n", "delete")],
    )
    note = get(client, sid).json()["comments"][0]
    assert note["comment_id"] == cid
    assert note["status"] == "stale"


def test_boolean_vs_number_marks_stale(client):
    sid = create_session(client, {"flag": True, "num": 1})
    register(
        client,
        sid,
        [
            {"path": "/flag", "comment": "bool"},
            {"path": "/num", "comment": "num"},
        ],
    )
    submit(
        client,
        sid,
        0,
        [
            op("alice", 1, {"alice": 1}, "/flag", "set", 1),
            op("alice", 2, {"alice": 2}, "/num", "set", 1.0),
        ],
    )
    notes = get(client, sid).json()["comments"]
    statuses = {n["path"]: n["status"] for n in notes}
    assert statuses["/flag"] == "stale"  # true != 1
    assert statuses["/num"] == "pending"  # 1 == 1.0


def test_ancestor_replacement_marks_descendant_comment_stale(client):
    sid = create_session(client, {"locked": {"v": 1}, "other": 1})
    register(client, sid, [{"path": "/locked/v", "comment": "deep"}])
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/locked", "set", {"v": 2})],
    )
    assert get(client, sid).json()["comments"][0]["status"] == "stale"


def test_ancestor_replacement_preserving_subtree_keeps_comment_pending(client):
    sid = create_session(client, {"locked": {"v": 1}, "keep": 1})
    register(client, sid, [{"path": "/locked/v", "comment": "deep"}])
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/locked", "set", {"v": 1})],
    )
    assert get(client, sid).json()["comments"][0]["status"] == "pending"


def test_array_index_shift_marks_comment_stale(client):
    sid = create_session(client, {"items": [10, 20, 30]})
    register(client, sid, [{"path": "/items/0", "comment": "first"}])
    # insert at the head: /items/0 now holds the new element
    resp = submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/items", "insert", 99, index=0)],
    )
    assert resp.status_code == 200
    note = get(client, sid).json()["comments"][0]
    assert note["status"] == "stale"
    assert note["stale_revision"] == 1


def test_appending_past_reviewed_index_keeps_comment_pending(client):
    sid = create_session(client, {"items": [10]})
    register(client, sid, [{"path": "/items/0", "comment": "first"}])
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/items", "insert", 20)],
    )
    assert get(client, sid).json()["comments"][0]["status"] == "pending"


def test_root_comment_is_marked_stale_by_any_document_change(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, [{"path": "", "comment": "root"}])
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/b", "set", 2)]
    )
    assert get(client, sid).json()["comments"][0]["status"] == "stale"


def test_replace_commit_marks_stale_comments(client):
    sid = create_session(
        client, {"items": [{"n": 1}, {"n": 2}, {"n": 3}]}
    )
    register(
        client,
        sid,
        [
            {"path": "/items/1/n", "comment": "two"},
            {"path": "/items/0/n", "comment": "one"},
        ],
    )
    resp = replace(
        client, sid, 0, "zoe", "$.items[?(@.n >= 2)].n", 0
    )
    assert resp.status_code == 200
    assert resp.json()["revision"] == 1
    statuses = {c["path"]: c["status"] for c in get(client, sid).json()["comments"]}
    assert statuses["/items/1/n"] == "stale"
    assert statuses["/items/0/n"] == "pending"


def test_retransmitted_batch_does_not_mark_stale(client):
    sid = create_session(client, {"title": "draft"})
    register(client, sid, [{"path": "/title", "comment": "?"}])
    batch = [op("alice", 1, {"alice": 1}, "/title", "set", "x")]
    submit(client, sid, 0, batch)
    # after the first commit the comment is stale; an exact retransmission
    # adds no revision and must not rewrite the stale revision marker
    again = submit(client, sid, 1, batch)
    assert again.status_code == 200
    assert again.json()["revision"] == 1
    note = get(client, sid).json()["comments"][0]
    assert note["status"] == "stale"
    assert note["stale_revision"] == 1


def test_zero_hit_replace_does_not_mark_stale(client):
    sid = create_session(client, {"title": "draft"})
    register(client, sid, [{"path": "/title", "comment": "?"}])
    resp = replace(client, sid, 0, "zoe", "$.nowhere", None)
    assert resp.status_code == 200
    assert resp.json()["revision"] == 0
    assert get(client, sid).json()["comments"][0]["status"] == "pending"


def test_only_pending_comments_are_evaluated_on_commit(client):
    sid = create_session(client, {"a": 1, "b": 1})
    register(
        client,
        sid,
        [
            {"path": "/a", "comment": "ca"},
            {"path": "/b", "comment": "cb"},
        ],
    )
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)])
    # ca is stale at revision 1; now commit changes /b and restores /a.
    submit(
        client,
        sid,
        1,
        [
            op("bob", 1, {"alice": 1, "bob": 1}, "/a", "set", 1),
            op("bob", 2, {"alice": 1, "bob": 2}, "/b", "set", 9),
        ],
    )
    notes = {c["path"]: c for c in get(client, sid).json()["comments"]}
    # sticky stale keeps its original stale revision
    assert notes["/a"]["status"] == "stale"
    assert notes["/a"]["stale_revision"] == 1
    assert notes["/b"]["status"] == "stale"
    assert notes["/b"]["stale_revision"] == 2


# ---------------------------------------------------------------------------
# resolving
# ---------------------------------------------------------------------------


def test_resolve_pending_comment(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, [{"path": "/a", "comment": "ok?"}]).json()[
        "comments"
    ][0]["comment_id"]
    resp = resolve(client, sid, cid, 0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["comment_id"] == cid
    assert body["status"] == "resolved"
    assert body["resolved_revision"] == 0
    assert body["stale_revision"] is None
    # no revision was added
    assert get(client, sid).json()["revision"] == 0
    assert get(client, sid).json()["comments"][0]["status"] == "resolved"


def test_resolve_stale_comment_without_confirmation_is_409(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, [{"path": "/a", "comment": "?"}]).json()[
        "comments"
    ][0]["comment_id"]
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)])
    resp = resolve(client, sid, cid, 1)
    assert resp.status_code == 409
    assert "stale" in resp.json()["detail"]
    # nothing changed
    assert get(client, sid).json()["comments"][0]["status"] == "stale"


def test_resolve_stale_comment_with_confirmation_succeeds(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, [{"path": "/a", "comment": "?"}]).json()[
        "comments"
    ][0]["comment_id"]
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)])
    resp = resolve(client, sid, cid, 1, confirm_stale=True)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resolved"
    assert body["stale_revision"] == 1
    assert body["resolved_revision"] == 1


def test_resolve_with_stale_expected_revision_is_409(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, [{"path": "/a", "comment": "?"}]).json()[
        "comments"
    ][0]["comment_id"]
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)])
    submit(client, sid, 1, [op("bob", 1, {"alice": 1, "bob": 1}, "/x", "set", 1)])
    # request carries the revision before bob's commit; even with
    # confirm_stale the optimistic check must win
    resp = resolve(client, sid, cid, 1, confirm_stale=True)
    assert resp.status_code == 409
    assert "expected revision 1" in resp.json()["detail"]
    assert get(client, sid).json()["comments"][0]["status"] == "stale"


def test_resolve_unknown_comment_is_404(client):
    sid = create_session(client, {})
    resp = resolve(client, sid, "0" * 32, 0)
    assert resp.status_code == 404


def test_resolve_unknown_session_is_404(client):
    resp = resolve(client, "0" * 32, "0" * 32, 0)
    assert resp.status_code == 404


def test_resolving_resolved_comment_is_idempotent(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, [{"path": "/a", "comment": "?"}]).json()[
        "comments"
    ][0]["comment_id"]
    first = resolve(client, sid, cid, 0).json()
    second = resolve(client, sid, cid, 0).json()
    assert first == second
    # a later commit does not reopen a resolved comment
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)])
    assert get(client, sid).json()["comments"][0]["status"] == "resolved"
    # and resolving it again at the new revision is idempotent too
    again = resolve(client, sid, cid, 1)
    assert again.status_code == 200
    assert again.json()["status"] == "resolved"
    assert again.json()["resolved_revision"] == 0


# ---------------------------------------------------------------------------
# archiving
# ---------------------------------------------------------------------------


def test_archive_blocked_while_pending_or_stale_comments(client):
    sid = create_session(client, {"a": 1, "b": 1})
    register(
        client,
        sid,
        [{"path": "/a", "comment": "ca"}, {"path": "/b", "comment": "cb"}],
    )
    # pending comments block
    resp = archive(client, sid, 0)
    assert resp.status_code == 409
    cids = {c["path"]: c["comment_id"] for c in get(client, sid).json()["comments"]}
    resolve(client, sid, cids["/a"], 0)
    still_blocked = archive(client, sid, 0)
    assert still_blocked.status_code == 409

    # make /b stale; a stale comment blocks as well
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/b", "set", 2)]
    )
    blocked = archive(client, sid, 1)
    assert blocked.status_code == 409
    assert "stale" in blocked.json()["detail"]

    # process the stale one explicitly -> archival proceeds
    confirmed = resolve(client, sid, cids["/b"], 1, confirm_stale=True)
    assert confirmed.status_code == 200
    sealed = archive(client, sid, 1)
    assert sealed.status_code == 200
    assert sealed.json()["archived"] is True
    # comments remain readable in their final state
    state = get(client, sid).json()
    assert {c["status"] for c in state["comments"]} == {"resolved"}


def test_blocked_archive_changes_nothing(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, [{"path": "/a", "comment": "?"}])
    before = get(client, sid).json()
    archive(client, sid, 0)
    after = get(client, sid).json()
    assert after["archived"] is False
    assert after["revision"] == before["revision"] == 0
    assert after["document"] == before["document"]
    # writes still accepted
    ok = submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    assert ok.status_code == 200


def test_archive_with_no_comments_still_works(client):
    sid = create_session(client, {"a": 1})
    assert archive(client, sid, 0).status_code == 200


def test_comment_changes_after_archive_are_409(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, [{"path": "/a", "comment": "?"}]).json()[
        "comments"
    ][0]["comment_id"]
    resolve(client, sid, cid, 0)
    archive(client, sid, 0)

    # new registrations -> 409 (seal checked before pointer validation)
    reg = register(client, sid, [{"path": "/a", "comment": "late"}])
    assert reg.status_code == 409
    # resolving an existing resolved comment again -> 409 as well
    again = resolve(client, sid, cid, 0)
    assert again.status_code == 409


# ---------------------------------------------------------------------------
# comments add neither revisions nor change history
# ---------------------------------------------------------------------------


def test_comments_do_not_add_revisions_or_change_events(client):
    sid = create_session(client, {"n": 0})
    register(client, sid, [{"path": "/n", "comment": "?"}])
    resolve(
        client,
        sid,
        get(client, sid).json()["comments"][0]["comment_id"],
        0,
    )
    page = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 0,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    ).json()
    assert page["changes"] == []
    assert page["next_cursor"] == 0
    assert get(client, sid).json()["revision"] == 0


# ---------------------------------------------------------------------------
# durability and legacy records
# ---------------------------------------------------------------------------


def test_comments_survive_store_restart(tmp_path):
    store_dir = str(tmp_path / "sessions")
    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = create_session(client, {"a": 1, "items": [1, 2]})
    reg = register(
        client,
        sid,
        [{"path": "/a", "comment": "ca"}, {"path": "/items/0", "comment": "ci"}],
    ).json()
    cids = [c["comment_id"] for c in reg["comments"]]
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)])

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    notes = get(client, sid).json()["comments"]
    by_id = {c["comment_id"]: c for c in notes}
    assert by_id[cids[0]]["status"] == "stale"
    assert by_id[cids[0]]["stale_revision"] == 1
    assert by_id[cids[0]]["value"] == 1
    assert by_id[cids[1]]["status"] == "pending"

    # processing still works after restart
    resp = resolve(client, sid, cids[0], 1, confirm_stale=True)
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"


def test_legacy_record_without_comments_has_empty_comments(tmp_path):
    store_dir = tmp_path / "sessions"
    store_dir.mkdir()
    sid = "b" * 32
    (store_dir / f"{sid}.json").write_text(
        json.dumps(
            {
                "session_id": sid,
                "revision": 0,
                "document": {"old": True},
                "version": {},
                "log": [],
            }
        )
    )
    app.state.session_store = SessionStore(str(store_dir))
    client = TestClient(app)
    state = get(client, sid).json()
    assert state["comments"] == []
    # old sessions behave exactly as before: no review gating on archive
    assert archive(client, sid, 0).status_code == 200


# ---------------------------------------------------------------------------
# store-level API
# ---------------------------------------------------------------------------


def _mp_resolver(store_dir, sid, cid, expected, confirm, queue):
    store = SessionStore(store_dir)
    try:
        store.resolve_comment(sid, cid, expected, confirm_stale=confirm)
        queue.put("resolved")
    except RevisionConflictError:
        queue.put("conflict")
    except Exception as exc:  # pragma: no cover - surfaces as test failure
        queue.put(f"other:{type(exc).__name__}")


def _mp_committer(store_dir, sid, queue):
    store = SessionStore(store_dir)
    try:
        store.submit_batch(
            sid,
            1,
            [
                Operation(
                    actor="y",
                    seq=1,
                    version={"x": 1, "y": 1},
                    path="/b",
                    action="set",
                    value=3,
                )
            ],
        )
        queue.put("committed")
    except RevisionConflictError:
        queue.put("conflict")


def test_resolve_and_commit_race_across_processes(tmp_path):
    # A stale comment at revision 1: a process resolves it (expected=1)
    # while another commits at expected=1.  The lock orders the two:
    #   * resolve first -> both succeed (resolving consumes no revision);
    #   * commit first  -> the resolve sees revision 2 and gets 409.
    # In every valid ordering the loser reads a self-consistent record.
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({"a": 1}, {})
    cid = store.register_comments(rec.session_id, [("/a", "c")])["comments"][0][
        "comment_id"
    ]
    store.submit_batch(
        rec.session_id,
        0,
        [
            Operation(
                actor="x",
                seq=1,
                version={"x": 1},
                path="/a",
                action="set",
                value=2,
            )
        ],
    )

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()

    def resolve_then_barrier():
        barrier.wait(timeout=15)
        _mp_resolver(store_dir, rec.session_id, cid, 1, True, queue)

    def commit_then_barrier():
        barrier.wait(timeout=15)
        _mp_committer(store_dir, rec.session_id, queue)

    procs = [
        ctx.Process(target=resolve_then_barrier),
        ctx.Process(target=commit_then_barrier),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    results = sorted(queue.get(timeout=15) for _ in procs)
    assert results in (
        ["committed", "resolved"],  # resolve landed first
        ["committed", "conflict"],  # commit landed first: stale expected rev
    ), results

    state = SessionStore(store_dir).get_state(rec.session_id)
    note = state["comments"][0]
    if results == ["committed", "conflict"]:
        assert state["revision"] == 2
        assert note["status"] == "stale"  # the rejected resolve changed nothing
        # retrying at the new revision with confirmation succeeds
        again = ctx.Queue()
        proc = ctx.Process(
            target=_mp_resolver,
            args=(store_dir, rec.session_id, cid, 2, True, again),
        )
        proc.start()
        proc.join(timeout=60)
        assert proc.exitcode == 0
        assert again.get(timeout=15) == "resolved"
        note = SessionStore(store_dir).get_state(rec.session_id)["comments"][0]
        assert note["status"] == "resolved"
        assert note["resolved_revision"] == 2
    else:
        assert state["revision"] == 2
        assert note["status"] == "resolved"
        assert note["resolved_revision"] == 1


def test_store_level_errors(tmp_path):
    store = SessionStore(str(tmp_path / "sessions"))
    rec = store.create({"a": 1}, {})
    with pytest.raises(ReviewCommentValidationError):
        store.register_comments(rec.session_id, [("/missing", "x")])
    created = store.register_comments(rec.session_id, [("/a", "x")])["comments"]
    cid = created[0]["comment_id"]
    store.submit_batch(
        rec.session_id,
        0,
        [
            Operation(
                actor="alice",
                seq=1,
                version={"alice": 1},
                path="/a",
                action="set",
                value=2,
            )
        ],
    )
    # stale without confirmation -> 409-class conflict
    with pytest.raises(RevisionConflictError):
        store.resolve_comment(rec.session_id, cid, 1)
    # stale expected revision also conflicts
    with pytest.raises(RevisionConflictError):
        store.resolve_comment(rec.session_id, cid, 0, confirm_stale=True)
    # archival is blocked by the stale comment
    with pytest.raises(OpenReviewCommentsError):
        store.archive_session(rec.session_id, 1)
    store.resolve_comment(rec.session_id, cid, 1, confirm_stale=True)
    sealed = store.archive_session(rec.session_id, 1)
    assert sealed["archived"] is True
    # comments frozen: no more changes
    with pytest.raises(SessionArchivedError):
        store.register_comments(rec.session_id, [("/a", "late")])
    with pytest.raises(SessionArchivedError):
        store.resolve_comment(rec.session_id, cid, 1)
