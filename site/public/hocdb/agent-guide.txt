# HOCDB for agents

Use the native hocdb CLI for local storage and analysis without writing a
binding adapter. Start with hocdb --version and hocdb describe. The latter
returns the command contract, required parameters, defaults and limits.
This is a shell tool, not an MCP or HTTP server.

## First task

~~~bash
hocdb init --db ./market --preset ticks --timestamp-unit s
echo '{"timestamp":1700000000,"price":100,"volume":2}' | hocdb append --db ./market
hocdb info --db ./market
hocdb snapshot --db ./market
~~~

Use a new, explicit directory for init. The parent directory must exist.
Subsequent commands recover the schema and timestamp unit from hocdb.json.
The CLI opens databases it created; existing embedded databases keep using
their language binding and explicit schema.

## Tool calling contract

Pass an argument array to your process runner. Each parameter from describe
becomes a --kebab-case flag: timestamp_unit becomes --timestamp-unit. Send JSON
records over stdin or pass --input FILE; avoid interpolating records into a
shell command. Commands never prompt and never use terminal colors.

| Result | Stream | Exit | Shape |
| --- | --- | --- | --- |
| Success | stdout | 0 | {"ok":true,"command":"query","data":{...}} |
| Failure | stderr | 1 | {"ok":false,"error":{"code":"...","message":"..."}} |
| Help | stdout | 0 | Plain text, only for no arguments/help/--help |

Parse the appropriate stream as JSON, check ok, then inspect data or error.code.
Unknown commands, unknown flags, repeated flags and missing values fail.
Do not infer success from an empty output or ignore the exit code.

## Choose the smallest read

| Need | Command |
| --- | --- |
| Discover tools, arguments and limits | describe |
| Discover a database's schema and record count | info --db DIR |
| Inspect records in a known interval | query --db DIR --start N --end N |
| Min, max, sum, mean and count | stats --db DIR --field NAME |
| Latest price and indicator values | snapshot --db DIR |
| Gaps, outliers and volume quality | health --db DIR --gap-threshold N |

Query returns at most --limit rows (100 by default, 10000 maximum). A larger
result fails with BufferTooSmall; narrow the time range instead of repeatedly
issuing the same request. Analytics may scan more history than query.
Use bindings for custom indicator selections, large reads, calendars, pair
analytics, backtests, universe scans, ring buffers and retention policies.

## Data semantics

- Timestamps increase strictly. Equal timestamps are rejected. Valid CLI
  timestamps are Unix times from 0 through 9223372036854775806.
- Ranges are [start,end): inclusive start, exclusive end, in the database's unit.
  Bucket sizes and gap thresholds use that same unit.
- i64/u64 inputs accept integers or decimal strings. Integers outside
  JavaScript's safe range are returned as decimal strings to preserve precision.
  Keep them as strings or BigInt; do not coerce them to Number.
- Analytics return null for missing, non-finite or warm-up values.
  A null RSI is unavailable data, not RSI zero.
- Use a numeric close or price column for snapshot/health. volume is optional
  in custom schemas. Snapshot's --bars default is 2500 and maximum is 100000.
- Set --periods-per-year explicitly for comparable annualized risk statistics.
  The default 0 asks the engine to infer it.
- append accepts one object or an array up to 16 MiB. Every object must have
  exactly the schema fields. There is no NDJSON import in this release.
- No append is idempotent. A successful append is flushed and fsynced. Validation
  failures add no input records; storage/I/O failures can partially append.
  Inspect the stored tail before retrying.
- init and append are the only mutating commands. All other database commands
  use read-only handles. No CLI command removes data, changes a schema, or
  silently enables circular overwrite.

## Handle errors

| Code | Next action |
| --- | --- |
| PathAlreadyExists | Use the existing database or choose another new directory. |
| FileNotFound | Check the paths. Create a new database only when intended. |
| InvalidRecord / InvalidSchema | Inspect info or describe; fix field types and names. |
| TimestampOutOfOrder | Check the stored tail; sort/deduplicate input before retrying. |
| DatabaseLocked | Wait for the writer to close; reads may proceed. |
| BufferTooSmall | Narrow the query range or raise --limit within its maximum. |
| StreamTooLong | Split ingestion into smaller JSON arrays. |
| FileFull | Create another database; existing records are preserved. |

The CLI contract is version 1, reported by --version and describe. HOC2 remains
the storage format; the native library's C ABI is unchanged by the CLI.
See [quickstart](QUICKSTART.md), [analytics](../INDICATORS.md), and the
language-specific READMEs under bindings/.
