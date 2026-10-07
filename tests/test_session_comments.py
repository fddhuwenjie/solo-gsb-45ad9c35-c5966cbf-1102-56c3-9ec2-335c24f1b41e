"""End-to-end tests for session review comments.

Review comments are metadata pinned to RFC 6901 JSON Pointers of an open
session: registration snapshots the pointer's value at that moment without
bumping the revision, committing batches/replacements persistently marks
comments outdated when the registered value changes, comments are processed
one by one (outdated ones require an explicit confirmation), and archiving
is refused while any comment is still pending or outdated.
"""
from __future__ import annotations

import json
import multiprocessing
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Operation
from app.sessions import SessionStore


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


def register(client, session_id, path, comment, expected_revision=None):
    body = {"path": path, "comment": comment}
    if expected_revision is not None:
        body["expected_revision"] = expected_revision
    return client.post(f"/sessions/{session_id}/comments", json=body)


def resolve(client, session_id, comment_id, expected_revision, confirm=False):
    return client.post(
        f"/sessions/{session_id}/comments/{comment_id}/resolve",
        json={
            "expected_revision": expected_revision,
            "confirm_outdated": confirm,
        },
    )


def archive(client, session_id, expected_revision):
    return client.post(
        f"/sessions/{session_id}/archive",
        json={"expected_revision": expected_revision},
    )


def get_comments(client, session_id):
    return client.get(f"/sessions/{session_id}").json()["comments"]


# ---------------------------------------------------------------------------
# registration: snapshot and no revision impact
# ---------------------------------------------------------------------------


def test_register_comment_snapshots_current_value_and_stays_pending(client):
    sid = create_session(client, {"a": {"b": 1}})
    resp = register(client, sid, "/a/b", "please double check")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["revision"] == 0
    note = body["comment"]
    assert note["path"] == "/a/b"
    assert note["comment"] == "please double check"
    assert note["status"] == "pending"
    assert note["registered_revision"] == 0
    assert note["value_present"] is True
    assert note["value"] == 1
    assert len(note["comment_id"]) == 32


def test_register_distinguishes_explicit_null_and_missing(client):
    sid = create_session(client, {"n": None})
    null_note = register(client, sid, "/n", "explicit null").json()["comment"]
    assert null_note["value_present"] is True
    assert null_note["value"] is None

    missing_note = register(client, sid, "/absent", "not there").json()["comment"]
    assert missing_note["value_present"] is False
    assert missing_note["value"] is None


def test_register_on_root_pointer_snapshots_whole_document(client):
    doc = {"a": 1, "items": [1, 2]}
    sid = create_session(client, doc)
    note = register(client, sid, "", "whole doc").json()["comment"]
    assert note["path"] == ""
    assert note["value"] == doc
    assert note["value_present"] is True


def test_register_adds_no_revision_history_or_change_event(client):
    sid = create_session(client, {"n": 0})
    register(client, sid, "/n", "one")
    register(client, sid, "/n", "two")
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 0
    assert len(state["comments"]) == 2
    changes = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 0,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    ).json()
    assert changes["current_revision"] == 0
    assert changes["changes"] == []


def test_new_session_read_returns_empty_comments(client):
    sid = create_session(client, {"a": 1})
    state = client.get(f"/sessions/{sid}").json()
    assert state["comments"] == []


def test_comments_listed_in_registration_order_on_read(client):
    sid = create_session(client, {"a": 1, "b": 2})
    c1 = register(client, sid, "/a", "first").json()["comment"]["comment_id"]
    c2 = register(client, sid, "/b", "second").json()["comment"]["comment_id"]
    assert [c["comment_id"] for c in get_comments(client, sid)] == [c1, c2]
    assert all(c["status"] == "pending" for c in get_comments(client, sid))


def test_register_invalid_pointer_returns_422(client):
    sid = create_session(client, {})
    resp = register(client, sid, "no-leading-slash", "x")
    assert resp.status_code == 422
    assert "JSON Pointer" in resp.json()["detail"]
    # illegal ~ escape, same strictness as protected paths / resolve
    assert register(client, sid, "/a~2", "x").status_code == 422


def test_register_empty_comment_returns_422(client):
    sid = create_session(client, {"a": 1})
    assert register(client, sid, "/a", "").status_code == 422


def test_register_unknown_session_returns_404(client):
    resp = register(client, "0" * 32, "/a", "x")
    assert resp.status_code == 404


def test_register_malformed_session_id_returns_404(client):
    assert register(client, "nope", "/a", "x").status_code == 404


def test_register_with_optional_expected_revision_guard(client):
    sid = create_session(client, {"a": 1})
    # stale optional guard -> 409
    resp = register(client, sid, "/a", "late", expected_revision=5)
    assert resp.status_code == 409
    assert client.get(f"/sessions/{sid}").json()["comments"] == []
    # matching guard registers
    assert register(client, sid, "/a", "ok", expected_revision=0).status_code == 200
    # guard omitted still registers
    assert register(client, sid, "/a", "ok2").status_code == 200


# ---------------------------------------------------------------------------
# outdated detection on committing batches and replacements
# ---------------------------------------------------------------------------


def test_batch_changing_registered_value_marks_comment_outdated(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, "/a", "watch a")
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/a", "set", 2)],
    )
    note = get_comments(client, sid)[0]
    assert note["status"] == "outdated"
    # the registration snapshot itself is preserved
    assert note["value"] == 1
    assert note["registered_revision"] == 0


def test_batch_writing_same_value_keeps_comment_pending(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, "/a", "watch a")
    # a visible no-op still commits a revision (existing semantics) but the
    # registered value is unchanged, so the comment stays pending
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/a", "set", 1)],
    )
    assert get_comments(client, sid)[0]["status"] == "pending"


def test_outdated_is_sticky_after_value_is_restored(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, "/a", "watch a")
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)]
    )
    assert get_comments(client, sid)[0]["status"] == "outdated"
    # change it back to the exact registered value: no automatic recovery
    submit(
        client, sid, 1, [op("alice", 2, {"alice": 2}, "/a", "set", 1)]
    )
    assert get_comments(client, sid)[0]["status"] == "outdated"


def test_boolean_and_number_never_compare_equal(client):
    sid = create_session(client, {"b": True})
    register(client, sid, "/b", "bool field")
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/b", "set", 1)]
    )
    assert get_comments(client, sid)[0]["status"] == "outdated"

    sid2 = create_session(client, {"b": 1})
    register(client, sid2, "/b", "number field")
    submit(
        client, sid2, 0, [op("bob", 1, {"bob": 1}, "/b", "set", True)]
    )
    assert get_comments(client, sid2)[0]["status"] == "outdated"

    # 1 and 1.0 are equal (numeric semantics, same as protected paths)
    sid3 = create_session(client, {"n": 1})
    register(client, sid3, "/n", "number field")
    submit(
        client, sid3, 0, [op("zoe", 1, {"zoe": 1}, "/n", "set", 1.0)]
    )
    assert get_comments(client, sid3)[0]["status"] == "pending"


def test_ancestor_replacement_marks_descendant_comment_outdated(client):
    sid = create_session(client, {"a": {"deep": {"x": 1}}})
    register(client, sid, "/a/deep/x", "deep leaf")
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/a", "set", {"deep": {"x": 9}})],
    )
    assert get_comments(client, sid)[0]["status"] == "outdated"

    # an ancestor rewrite that preserves the descendant value does not
    sid2 = create_session(client, {"a": {"deep": {"x": 1}, "keep": 1}})
    register(client, sid2, "/a/deep/x", "deep leaf")
    submit(
        client,
        sid2,
        0,
        [
            op(
                "alice",
                1,
                {"alice": 1},
                "/a",
                "set",
                {"deep": {"x": 1}, "added": 2},
            )
        ],
    )
    assert get_comments(client, sid2)[0]["status"] == "pending"


def test_array_insert_shifting_registered_index_marks_comment_outdated(client):
    sid = create_session(client, {"items": [10, 20, 30]})
    register(client, sid, "/items/1", "middle element")
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/items", "insert", 5, 0)],
    )
    # /items/1 now holds 10 rather than the registered 20
    assert get_comments(client, sid)[0]["status"] == "outdated"

    # an append past the registered index leaves it in place
    sid2 = create_session(client, {"items": [10, 20]})
    register(client, sid2, "/items/0", "first element")
    submit(
        client,
        sid2,
        0,
        [op("bob", 1, {"bob": 1}, "/items", "insert", 30)],
    )
    assert get_comments(client, sid2)[0]["status"] == "pending"


def test_delete_and_create_between_missing_and_null_marks_outdated(client):
    sid = create_session(client, {"m": None})
    register(client, sid, "/m", "explicit null")
    register(client, sid, "/abs", "was absent")
    # rewriting null with null changes nothing
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/m", "set", None)]
    )
    statuses = {c["path"]: c["status"] for c in get_comments(client, sid)}
    assert statuses == {"/m": "pending", "/abs": "pending"}
    # creating the previously-missing pointer (with an explicit null) is a change
    submit(
        client, sid, 1, [op("alice", 2, {"alice": 2}, "/abs", "set", None)]
    )
    statuses = {c["path"]: c["status"] for c in get_comments(client, sid)}
    assert statuses == {"/m": "pending", "/abs": "outdated"}
    # deleting the explicit-null pointer is a change for /m
    submit(
        client, sid, 2, [op("alice", 3, {"alice": 3}, "/m", "delete")]
    )
    statuses = {c["path"]: c["status"] for c in get_comments(client, sid)}
    assert statuses == {"/m": "outdated", "/abs": "outdated"}


def test_comment_on_unrelated_pointer_unaffected_by_commit(client):
    sid = create_session(client, {"a": 1, "b": 2})
    register(client, sid, "/a", "watch a")
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/b", "set", 22)]
    )
    statuses = {c["path"]: c["status"] for c in get_comments(client, sid)}
    assert statuses == {"/a": "pending"}


def test_resolved_comments_are_not_marked_by_later_commits(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    assert resolve(client, sid, cid, 0).status_code == 200
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)]
    )
    assert get_comments(client, sid)[0]["status"] == "resolved"


def test_jsonpath_replace_changing_value_marks_comment_outdated(client):
    sid = create_session(client, {"items": [{"n": 1}, {"n": 2}], "k": 0})
    register(client, sid, "/items/1/n", "second n")
    register(client, sid, "/k", "k field")
    resp = replace(client, sid, 0, "zoe", "$.items[?(@.n >= 2)].n", 0)
    assert resp.status_code == 200
    statuses = {c["path"]: c["status"] for c in get_comments(client, sid)}
    assert statuses == {"/items/1/n": "outdated", "/k": "pending"}


def test_jsonpath_replace_same_value_keeps_comment_pending(client):
    sid = create_session(client, {"k": 1})
    register(client, sid, "/k", "watch k")
    replace(client, sid, 0, "zoe", "$.k", 1)  # commits rev 1, same value
    assert get_comments(client, sid)[0]["status"] == "pending"
    # zero-hit replacement commits nothing and keeps the comment pending
    zero = replace(client, sid, 1, "zoe", "$.nowhere", 1)
    assert zero.status_code == 200
    assert zero.json()["revision"] == 1
    assert get_comments(client, sid)[0]["status"] == "pending"


def test_stale_and_rejected_commits_do_not_touch_comments(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, "/a", "watch a")
    # stale revision: the edit never commits, the comment stays pending
    resp = submit(
        client, sid, 7, [op("alice", 1, {"alice": 1}, "/a", "set", 2)]
    )
    assert resp.status_code == 409
    assert get_comments(client, sid)[0]["status"] == "pending"


def test_marking_happens_in_same_commit_as_revision(client):
    # no revision bump => no outdated flip; retransmitted batch is a no-op
    sid = create_session(client, {"a": 1})
    register(client, sid, "/a", "watch a")
    batch = [op("alice", 1, {"alice": 1}, "/a", "set", 1)]
    assert submit(client, sid, 0, batch).json()["revision"] == 1
    assert get_comments(client, sid)[0]["status"] == "pending"
    retransmit = submit(client, sid, 1, batch)
    assert retransmit.json() == {
        "session_id": sid,
        "revision": 1,
        "accepted": 0,
        "duplicates": 1,
    }
    assert get_comments(client, sid)[0]["status"] == "pending"


# ---------------------------------------------------------------------------
# resolving comments
# ---------------------------------------------------------------------------


def test_resolve_pending_comment_succeeds_without_confirmation(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    resp = resolve(client, sid, cid, 0)
    assert resp.status_code == 200
    assert resp.json()["comment"]["status"] == "resolved"
    assert resp.json()["revision"] == 0
    assert get_comments(client, sid)[0]["status"] == "resolved"


def test_resolve_outdated_comment_requires_explicit_confirmation(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)]
    )
    resp = resolve(client, sid, cid, 1)
    assert resp.status_code == 409
    assert "outdated" in resp.json()["detail"]
    # the comment was not processed by the failed call
    assert get_comments(client, sid)[0]["status"] == "outdated"
    # explicit confirmation processes it
    confirmed = resolve(client, sid, cid, 1, confirm=True)
    assert confirmed.status_code == 200
    assert confirmed.json()["comment"]["status"] == "resolved"


def test_resolve_stale_expected_revision_returns_409(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)]
    )
    resp = resolve(client, sid, cid, 0)
    assert resp.status_code == 409
    assert "expected revision 0" in resp.json()["detail"]
    assert get_comments(client, sid)[0]["status"] == "pending"


def test_resolve_is_idempotent(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    first = resolve(client, sid, cid, 0).json()
    second = resolve(client, sid, cid, 0).json()
    assert first == second
    # resolving an already-resolved comment ignores the outdated flag --
    # nothing can move a resolved comment anyway
    third = resolve(client, sid, cid, 0, confirm=False).json()
    assert third["comment"]["status"] == "resolved"


def test_resolve_unknown_comment_returns_404(client):
    sid = create_session(client, {})
    assert resolve(client, sid, "0" * 32, 0).status_code == 404
    assert resolve(client, sid, "not-hex", 0).status_code == 404


def test_resolve_unknown_session_returns_404(client):
    assert resolve(client, "0" * 32, "0" * 32, 0).status_code == 404


def test_resolve_adds_no_revision_or_history(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    resolve(client, sid, cid, 0)
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 0
    changes = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 0,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    ).json()
    assert changes["changes"] == []


# ---------------------------------------------------------------------------
# archiving gating
# ---------------------------------------------------------------------------


def test_archive_refused_with_pending_comment(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, "/a", "watch a")
    resp = archive(client, sid, 0)
    assert resp.status_code == 409
    assert "pending" in resp.json()["detail"]
    state = client.get(f"/sessions/{sid}").json()
    assert state["archived"] is False
    # the refused seal changed nothing: edits still land
    follow = submit(
        client, sid, 0, [op("bob", 1, {"bob": 1}, "/x", "set", 1)]
    )
    assert follow.status_code == 200


def test_archive_refused_with_outdated_comment(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)]
    )
    resp = archive(client, sid, 1)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "0 pending" in detail and "1 outdated" in detail
    assert client.get(f"/sessions/{sid}").json()["archived"] is False
    # confirmation on the resolve endpoint clears the blocker
    assert resolve(client, sid, cid, 1, confirm=True).status_code == 200
    assert archive(client, sid, 1).status_code == 200


def test_archive_succeeds_when_all_comments_resolved(client):
    sid = create_session(client, {"a": 1, "b": 2})
    c1 = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    c2 = register(client, sid, "/b", "watch b").json()["comment"]["comment_id"]
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)]
    )
    assert resolve(client, sid, c1, 1, confirm=True).status_code == 200
    assert resolve(client, sid, c2, 1).status_code == 200
    resp = archive(client, sid, 1)
    assert resp.status_code == 200
    assert resp.json()["archived"] is True


def test_archive_with_no_comments_works_as_before(client):
    sid = create_session(client, {"a": 1})
    assert archive(client, sid, 0).status_code == 200


def test_archive_stale_revision_with_unresolved_comments_still_409(client):
    sid = create_session(client, {"a": 1})
    register(client, sid, "/a", "watch a")
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)]
    )
    # stale revision conflicts; the unresolved-comment check never runs
    resp = archive(client, sid, 0)
    assert resp.status_code == 409
    assert "expected revision 0" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# archived sessions: comments frozen
# ---------------------------------------------------------------------------


def test_comments_immutable_after_archive(client):
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    resolve(client, sid, cid, 0)
    archive(client, sid, 0)

    late = register(client, sid, "/a", "too late")
    assert late.status_code == 409
    assert "archived" in late.json()["detail"]

    again = resolve(client, sid, cid, 0)
    assert again.status_code == 409
    assert "archived" in again.json()["detail"]

    # the archived read still shows the processed comment
    state = client.get(f"/sessions/{sid}").json()
    assert state["archived"] is True
    assert len(state["comments"]) == 1
    assert state["comments"][0]["status"] == "resolved"


# ---------------------------------------------------------------------------
# durability, legacy records and multi-process races
# ---------------------------------------------------------------------------


def test_comments_survive_store_restart(tmp_path):
    store_dir = str(tmp_path / "sessions")
    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = create_session(client, {"a": 1})
    cid = register(client, sid, "/a", "watch a").json()["comment"]["comment_id"]
    submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)]
    )

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    note = get_comments(client, sid)[0]
    assert note["comment_id"] == cid
    assert note["status"] == "outdated"
    assert note["value"] == 1

    # confirming the outdated comment after restart clears the seal blocker
    assert resolve(client, sid, cid, 1, confirm=True).status_code == 200
    assert archive(client, sid, 1).status_code == 200


def test_legacy_session_without_comments_field_has_empty_comments(tmp_path):
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
    assert get_comments(client, sid) == []
    # no comments means nothing blocks the seal -- old behavior preserved
    assert archive(client, sid, 0).status_code == 200


def test_comment_registration_does_not_seed_legacy_history_origin(
    client, tmp_path
):
    # A legacy record (no history field) gains its history origin only via
    # the existing migration paths; comment registration must not rewrite
    # history even though it persists the new comment.
    store_dir = tmp_path / "sessions"
    store_dir.mkdir()
    sid = "c" * 32
    (store_dir / f"{sid}.json").write_text(
        json.dumps(
            {
                "session_id": sid,
                "revision": 0,
                "document": {"a": 1},
                "version": {},
                "log": [],
            }
        )
    )
    app.state.session_store = SessionStore(str(store_dir))
    client = TestClient(app)
    register(client, sid, "/a", "watch a")
    raw = json.loads((store_dir / f"{sid}.json").read_text())
    # registration persisted the comment but seeded no history origin
    assert "history" not in raw
    assert len(raw["comments"]) == 1


def _resolve_worker(store_dir, session_id, comment_id, expected, queue):
    store = SessionStore(store_dir)
    try:
        store.resolve_comment(session_id, comment_id, expected, True)
        status = "resolved"
    except Exception as exc:  # noqa: BLE001 - report the class name
        status = type(exc).__name__
    queue.put(status)


def test_resolve_race_with_commit_one_ordering_must_lose(tmp_path):
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({"a": 1}, {})
    outcome = store.register_comment(rec.session_id, "/a", "watch a")
    comment_id = outcome["comment"]["comment_id"]

    operation = Operation(
        actor="alice",
        seq=1,
        version={"alice": 1},
        path="/x",
        action="set",
        value=1,
    )

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()

    def barrier_commit():
        SessionStore(store_dir).get(rec.session_id)
        barrier.wait(timeout=15)
        try:
            store.submit_batch(rec.session_id, 0, [operation])
            queue.put("committed")
        except Exception as exc:  # noqa: BLE001
            queue.put(type(exc).__name__)

    def barrier_resolve():
        SessionStore(store_dir).get(rec.session_id)
        barrier.wait(timeout=15)
        _resolve_worker(store_dir, rec.session_id, comment_id, 0, queue)

    procs = [ctx.Process(target=barrier_commit), ctx.Process(target=barrier_resolve)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    statuses = sorted(queue.get(timeout=10) for _ in procs)
    # Both orderings are legal, and the lock serializes them: either the
    # commit lands first (revision moves to 1, the stale resolve gets 409)
    # or the resolve lands first (processed, then the commit lands).
    assert statuses in (
        ["RevisionConflictError", "committed"],
        ["committed", "resolved"],
    )

    final = store.get_state(rec.session_id)
    assert final["revision"] == 1
    note = final["comments"][0]
    if statuses == ["RevisionConflictError", "committed"]:
        # the failed resolve changed nothing; /x was touched, not /a
        assert note["status"] == "pending"
    else:
        assert note["status"] == "resolved"


# ---------------------------------------------------------------------------
# store-level API
# ---------------------------------------------------------------------------


def _flock_queue_child(store_dir, session_id, kind, comment_id, operation,
                       signal_dir, queue):
    """Child that joins the session's flock queue and reports before blocking.

    A marker file is written immediately before the child's own ``flock``
    call, so the parent can wait until every child is actually queued on
    the held lock (rather than merely scheduled to run) before releasing.
    """
    import fcntl
    import time

    import app.sessions as sessions_mod

    real_flock = fcntl.flock

    def instrumented_flock(fd, operation):  # noqa: ANN001
        if operation == fcntl.LOCK_EX:
            (signal_dir / kind).touch()
            # yield until the parent has observed the queued marker
            deadline = time.monotonic() + 15
            while not (signal_dir / "go").exists() and time.monotonic() < deadline:
                time.sleep(0.005)
        return real_flock(fd, operation)

    sessions_mod.fcntl.flock = instrumented_flock
    store = SessionStore(store_dir)
    try:
        if kind == "commit":
            store.submit_batch(session_id, 0, [operation])
            queue.put("committed")
        else:
            store.resolve_comment(session_id, comment_id, 0, True)
            queue.put("resolved")
    except Exception as exc:  # noqa: BLE001
        queue.put(type(exc).__name__)


def test_resolve_and_commit_race_serialized_under_session_lock(tmp_path):
    # Deterministic race: the parent process holds this session's
    # cross-process flock while two child processes queue on it; the parent
    # releases only after both children have actually entered their own
    # flock calls, so exactly one can proceed and the other must observe
    # the moved revision (409).
    store_dir = str(tmp_path / "sessions")
    signals = tmp_path / "signals"
    signals.mkdir()
    store = SessionStore(store_dir)
    rec = store.create({"a": 1}, {})
    outcome = store.register_comment(rec.session_id, "/a", "watch a")
    comment_id = outcome["comment"]["comment_id"]
    operation = Operation(
        actor="alice",
        seq=1,
        version={"alice": 1},
        path="/x",
        action="set",
        value=1,
    )

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_flock_queue_child,
            args=(
                store_dir, rec.session_id, "commit", comment_id, operation,
                signals, queue,
            ),
        ),
        ctx.Process(
            target=_flock_queue_child,
            args=(
                store_dir, rec.session_id, "resolve", comment_id, operation,
                signals, queue,
            ),
        ),
    ]
    with store._locked(rec.session_id):
        for proc in procs:
            proc.start()
        # wait for both children to be blocked inside their flock calls
        deadline = time.monotonic() + 15
        while not ((signals / "commit").exists() and
                   (signals / "resolve").exists()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # both are queued on the parent's exclusive lock; releasing lets the
        # kernel hand it to exactly one of them, the other loses the race
        (signals / "go").touch()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    statuses = sorted(queue.get(timeout=10) for _ in procs)
    assert statuses == ["RevisionConflictError", "committed"]

    final = store.get_state(rec.session_id)
    assert final["revision"] == 1
    assert final["document"] == {"a": 1, "x": 1}
    # the loser's stale resolve did not process the comment; the commit
    # touched /x, not /a, so it is still pending
    assert final["comments"][0]["status"] == "pending"


def test_store_register_on_archived_raises(tmp_path):
    store = SessionStore(str(tmp_path / "sessions"))
    rec = store.create({}, {})
    store.archive_session(rec.session_id, 0)
    with pytest.raises(Exception):
        store.register_comment(rec.session_id, "/a", "late")


def test_store_comment_statuses_view_payload(tmp_path):
    store = SessionStore(str(tmp_path / "sessions"))
    rec = store.create({"a": 1}, {})
    store.register_comment(rec.session_id, "/a", "note")
    state = store.get_state(rec.session_id)
    assert state["revision"] == 0
    note = state["comments"][0]
    assert note["status"] == "pending"
    assert note["value"] == 1
    # state payload exposes the same view shape as the HTTP API
    assert set(note) == {
        "comment_id",
        "path",
        "comment",
        "status",
        "registered_revision",
        "value_present",
        "value",
    }
