# Operations

## Revisions

A collection has one search-visible **active revision**. Changing extraction,
chunking, or embedding settings builds a replacement in the background. Search
continues against the active revision until you promote the ready replacement:

```bash
cementic status
cementic collection promote research
```

Promotion refuses a revision with failed documents or chunks unless you pass
`--force`. A new collection has no active revision yet; search can return its
partial results while it builds.

## Routine commands

| Command | Use it to |
| --- | --- |
| `cementic status` | Check indexing progress, worker state, and recent errors. |
| `cementic doctor` | Read-only checks for configuration, Postgres, embeddings, and index health. |
| `cementic collection promote NAME` | Make a ready revision searchable. |
| `cementic collection reindex NAME` | Rebuild vector and full-text indexes after index-setting changes. It does not re-embed documents. |
| `cementic stop` | Stop the watcher and pipeline worker; Postgres remains running. |

A failed document does not stop the rest of a build. The next `cementic start`
retries it. One exception is a document the worker died on three times in a row,
from a crash or an out-of-memory kill. It stays failed across restarts, so it
cannot stop the rest of the collection, until the file changes. If the embedding
service becomes unreachable, affected chunks remain pending instead of becoming
permanent failures.

## Long-running indexing

For a multi-day import, run the watcher and worker under your platform's service
manager so they restart after logout or reboot. The repository includes systemd
user-unit templates in [`packaging/systemd/`](../packaging/systemd/) as a Linux
example; other init systems should supervise the equivalent embedding server,
watcher, and pipeline-worker processes.

Use either those service units or `cementic start`, not both for the same
collection.

## Machine-readable output

A command that returns a list prints JSONL: one compact JSON object per line,
nothing at all when the list is empty. A command that returns one object prints
one indented JSON document. So `search --json`, `chunk`, `collection list --json`
and `collection revisions --json` print JSONL, while `status --json`,
`doctor --json` and `embedding status --json` print one document. A failure
prints nothing on stdout; the message goes to stderr. A test compares the shapes
below against what the commands build, so a key added or renamed without
updating this page fails the suite.

`status --json`, with no `-c`:

<!-- json-shape: status -->
```json
{
  "supervisor": {"state": "str", "collection": "str|null", "directories": ["str"]},
  "source_watcher": {
    "process": "str", "state": "str", "pid": "str", "processed": "int",
    "failed": "int", "skipped_files": "list", "current_file": "str|null",
    "last_error": "str|null", "last_error_at": "str|null"
  },
  "pipeline_worker": {
    "process": "str", "state": "str", "pid": "str", "current_file": "str|null",
    "last_error": "str|null", "last_error_at": "str|null",
    "current_activity": "str|null"
  },
  "health": {
    "db_reachable": "bool", "embedding_provider": "str",
    "embedding_healthy": "bool", "llama_daemon": "str"
  },
  "collections": {
    "NAME": {
      "documents": "int", "extracted_done": "int", "extracted_failed": "int",
      "chunked_done": "int", "chunked_failed": "int", "total_chunks": "int",
      "pending_embeddings": "int", "processing_embeddings": "int",
      "done_embeddings": "int", "failed_embeddings": "int",
      "extraction_pct": "float", "chunking_pct": "float", "embedding_pct": "float",
      "active_revision_label": "str|null", "ready_revision_label": "str|null",
      "building_revision_label": "str|null"
    }
  }
}
```

Keys that change with the situation:

- With `-c NAME`, `pipeline` replaces `collections`. It holds one collection's
  counts plus a `collection` key. With `--verbose` as well, `files` lists
  per-file progress and `files_truncated` says whether `--limit` cut it.
- When the health probe fails, `health` keeps its four keys set to `null` and
  adds `error`.
- When the database cannot be read, the top level gains `error`, and
  `collections` or `pipeline` is absent. The command then exits 1.

`doctor --json`:

<!-- json-shape: doctor -->
```json
{
  "ok": "bool",
  "checks": {
    "config": {}, "database": {}, "extensions": {}, "model": {}, "daemon": {},
    "stranded_chunkings": {}, "lexical_index": {}, "chunk_budget": {},
    "tokenizer": {}, "embedding_server": {}, "ocr": {},
    "extraction_commands": {}, "extractor_profile": {}, "embedding_canary": {}
  }
}
```

Each check has `status` (`ok`, `warning` or `fail`) and usually `message`. The
other keys are specific to the check. `extensions` maps each Postgres extension
to its own `status` and `message`, and is empty when the database is down. Then
`database` carries the full driver text in `error`. `ok` is false, and the
command exits 1, when any check fails.

`search --json`, one line per result, best first:

<!-- json-shape: search -->
```json
{
  "document_id": "int", "collection": "str", "source_path": "str", "content": "str",
  "score": "float", "distance": "float|null", "score_kind": "str", "rank": "int"
}
```

`document_id` is the stable id of the document the result stands for. `distance`
is null for a result only the full-text arm found. `rank` is the 1-based position.

`collection list --json`, one line per collection:

<!-- json-shape: collection_list -->
```json
{
  "name": "str", "documents": "int", "active_revision_label": "str|null",
  "ready_revision_label": "str|null", "building_revision_label": "str|null"
}
```

`collection revisions NAME --json`, one line per revision, newest first. The
fingerprints are in full, not truncated to eight characters as in the table:

<!-- json-shape: collection_revisions -->
```json
{
  "id": "int", "status": "str", "label": "str|null",
  "extractor_profile": {"name": "str", "fingerprint": "str"},
  "chunk_profile": {"fingerprint": "str"},
  "embedding_profile": {"provider": "str", "fingerprint": "str"},
  "created_at": "str|null", "promoted_at": "str|null"
}
```

`embedding status --json`, from the pid file or a `/proc` scan, with no health
probe (`status` and `doctor` own that):

<!-- json-shape: embedding_status -->
```json
{"state": "running|stopped|ambiguous", "pid": "int|null", "recovered": "bool", "error": "str|null"}
```

`recovered` is true when the pid file was missing or stale and the daemon was
found by scanning. `error` is set only when `state` is `ambiguous`.

## Limits

- One CLI-managed indexing session runs at a time. Pass multiple directories to
  one `cementic start` command to index them into one collection.
- Moving a watched root directory is reconciled on the next `start`; moves and
  deletions inside it are handled live.
- Searches across collections built with different embedding models are refused,
  because their distances cannot be ranked together.
- A collection that is a tiny filtered slice of a shared vector index can return
  fewer results than exist. A dedicated embedding profile avoids that trade-off.
