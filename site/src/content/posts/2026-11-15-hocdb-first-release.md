---
title: "HOCDB 0.1.1: local time-series data for people and agents"
seoTitle: "HOCDB 0.1.1: our first release"
date: "2026-11-15"
tags: ["hocdb", "releases", "zig", "databases", "agents"]
keywords: ["HOCDB", "embedded time-series database", "JSON CLI", "agent tools", "Zig database"]
excerpt: "HOCDB's first tagged release adds a native JSON CLI, safe database creation, validated imports and a command contract agents can discover."
---

**[HOCDB 0.1.1 is available](https://github.com/o1dstaRs/hocdb/releases/tag/v0.1.1).**
Our first tagged release brings the embedded time-series engine to a native
command line: create a database, import JSON records, query an interval and
compute statistics or market indicators. The downloads run on Linux x86_64
and Apple Silicon Macs without a compiler or language runtime.

A database library can have a useful storage engine and still be awkward to
try. HOCDB had that problem. Its executable was a demonstration program: it
cleared a data directory, wrote five records and read them back. That explained
the library, but it was the wrong starting point for someone who wanted to keep
real data or give an agent a storage tool.

That first interaction is what this release changes. Running `hocdb` now prints
help. Every database operation starts with an explicit directory, and ordinary
reads return structured JSON that a person, script or agent can inspect.
The version is 0.1.1 because that was already the library's version before we
created its first public release tag.

HOCDB is my other Zig project alongside [ZINC](/zinc/). ZINC runs models on local
GPUs. HOCDB stores timestamped records and computes analytics close to the
application. Both projects are about giving local software a useful, direct
interface to the machine it runs on.

## What ships

| Download | Platform |
| --- | --- |
| [Linux archive](https://github.com/o1dstaRs/hocdb/releases/download/v0.1.1/hocdb-0.1.1-linux-x86_64.tar.gz) | x86_64, statically linked |
| [macOS archive](https://github.com/o1dstaRs/hocdb/releases/download/v0.1.1/hocdb-0.1.1-macos-aarch64.tar.gz) | Apple Silicon, macOS 13 or later |

Each archive includes the CLI, quickstart, agent guide, JSON command contract
and MIT license. Verify the download against
[SHA256SUMS](https://github.com/o1dstaRs/hocdb/releases/download/v0.1.1/SHA256SUMS)
and extract it. The executable is `bin/hocdb`. These are CLI packages; the
language bindings remain available from source. The macOS binary is unsigned.

## From a record to a result

Put the binary on your PATH, or replace `hocdb` below with `./bin/hocdb` from
the extracted directory. Choose a new `./market` directory for this session:

~~~bash
hocdb init --db ./market --preset ticks --timestamp-unit s
echo '[{"timestamp":1700000000,"price":100,"volume":2},{"timestamp":1700000060,"price":110,"volume":3}]' \
  | hocdb append --db ./market
hocdb stats --db ./market --field price
~~~

The final command returns this JSON, shown with indentation for readability:

~~~json
{
  "ok": true,
  "command": "stats",
  "data": {
    "count": 2,
    "min": 100,
    "max": 110,
    "sum": 210,
    "mean": 105
  }
}
~~~

I checked this example against the published macOS archive after verifying its
checksum. The schema and timestamp unit are saved with the database, so later
commands do not need you to reconstruct them.

The ticks preset has timestamp, price and volume. There is also an OHLCV preset
for candle data and a custom-schema option for other time series, such as
sensor readings. A schema stays fixed once the database exists.

## An interface an agent can inspect

The command line treats its output as an API. Data commands return a JSON
envelope with `ok`, `command` and `data` on stdout. Errors return a JSON object
on stderr, including a code and a message that tells the caller what to check.
Success exits with code 0; failure exits with code 1. Commands never prompt.

| An agent needs to… | Command |
| --- | --- |
| Discover commands and argument limits | `hocdb describe` |
| Inspect schema, units and record count | `hocdb info --db ./market` |
| Read both example records | `hocdb query --db ./market --start 1700000000 --end 1700000061` |
| Compute a small statistical summary | `hocdb stats --db ./market --field price` |
| Read the latest indicator values | `hocdb snapshot --db ./market` |
| Check gaps and data quality | `hocdb health --db ./market` |

Ranges include the start and exclude the end. Here the unit is seconds, so
the query ends one second after the final example record.

The discovery command returns argument schemas as well as descriptions. An
agent can invoke the executable through a normal process runner, pass an
argument array and send records over stdin. It does not need an additional
server or an adapter for a particular assistant.

There are a few details that matter more than they first appear. A query
defaults to a maximum of 100 rows. If more match, it fails and asks the caller
to narrow the range or increase the limit. It does not silently return an
incomplete result or fill the agent's context with the entire database.

Integers outside JavaScript's safe range are represented as decimal strings.
That keeps a nanosecond timestamp exact through a JSON round trip. Indicators
that need more history return `null` during warm-up. Two records do not produce
a meaningful RSI, and the interface should make that absence clear.

## Keep writes explicit

Database creation requires a new directory, and every database command requires
an explicit path. Repeating `init` against an existing directory fails with
`PathAlreadyExists`; it preserves the original files. No command deletes a
database. Query, statistics, snapshot
and health use read-only handles, allowing them to inspect committed data while
another process owns the writer.

Append checks the whole input batch before writing records. Fields must match
the schema, numeric values must be finite and timestamps must increase
strictly, including relative to the existing data. A successful append flushes
and fsyncs before returning.

This is not a transaction system. A storage error can leave part of a batch
written, so an agent should inspect the stored tail before retrying. An append
is not idempotent. The [agent guide](/hocdb/agent-guide.txt) makes these recovery
rules part of the tool's documented behavior.

## The library behind the command line

The engine still uses fixed-size records, timestamp indexing, CRC32C integrity
checks and SIMD aggregation. The CLI does not change the HOC2 storage format or
the C ABI.

Applications can embed HOCDB through Zig or its C, C++, Python, Go, Node.js and
Bun bindings. Those APIs expose a broader analytics surface: custom indicators,
OHLCV resampling, calendars, pair analytics, backtesting and universe scans.
Ring buffers, retention and rollover are library options too.

The release is intentionally specific about its boundaries. Schemas are fixed,
each database has one writer, and CLI databases hold up to 2 GiB without
automatically overwriting earlier records. Language bindings require building
the native libraries with the pinned Zig 0.17.1 compiler. CLI-created databases
have a schema manifest; existing databases created through a binding continue
to use that binding and its explicit schema.

## What we checked before release

The [release build](https://github.com/o1dstaRs/hocdb/actions/runs/43773054280)
passed on Linux and macOS. Each platform built with ReleaseSafe, ran the core
and CLI tests, packaged the binary, then ran the CLI tests against the extracted
archive. The downloadable executable is part of the check.

The [test workflow](https://github.com/o1dstaRs/hocdb/actions/workflows/test.yml)
covers Debug and ReleaseSafe on both platforms and exercises every language
binding. The CLI checks cover preservation of existing files, read-only
behavior, rejected batches, exact integers, schema presets and the JSON
contract.

The [validation and performance report](https://github.com/o1dstaRs/hocdb/blob/v0.1.1/validation/zig17/README.md)
records the earlier engine measurements with their hardware, workloads and
raw samples. Those measurements describe the engine workloads in the report;
this release's main change is making that engine easier to try and automate.

The practical change is that trying HOCDB now starts with a few ordinary
commands. [Download it and follow the quickstart](/hocdb/), or give an agent
the [command contract](/hocdb/commands.json) and an explicit directory to work in.
