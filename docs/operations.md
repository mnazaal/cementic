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
retries it. If the embedding service becomes unreachable, affected chunks remain
pending instead of becoming permanent failures.

## Long-running indexing

For a multi-day import, run the watcher and worker under your platform's service
manager so they restart after logout or reboot. The repository includes systemd
user-unit templates in [`packaging/systemd/`](../packaging/systemd/) as a Linux
example; other init systems should supervise the equivalent embedding server,
watcher, and pipeline-worker processes.

Use either those service units or `cementic start`, not both for the same
collection.

## Limits

- One CLI-managed indexing session runs at a time. Pass multiple directories to
  one `cementic start` command to index them into one collection.
- Moving a watched root directory is reconciled on the next `start`; moves and
  deletions inside it are handled live.
- Searches across collections built with different embedding models are refused,
  because their distances cannot be ranked together.
- A collection that is a tiny filtered slice of a shared vector index can return
  fewer results than exist. A dedicated embedding profile avoids that trade-off.
