---
title: "HOCDB 0.1.1: local time-series data for people and agents"
date: "2026-11-15"
tags: ["hocdb", "zig", "databases", "agents"]
keywords: ["HOCDB", "embedded time-series database", "JSON CLI", "agent tools", "Zig database"]
excerpt: "HOCDB's first tagged release adds a native JSON CLI, safe database creation, validated imports and a command contract agents can discover."
---

A database library can have a useful storage engine and still be awkward to
try. HOCDB had that problem. Its executable was a demonstration program: it
cleared a data directory, wrote five records and read them back. That explained
the library, but it was the wrong starting point for someone who wanted to keep
real data or give an agent a storage tool.

[HOCDB 0.1.1](https://github.com/o1dstaRs/hocdb/releases/tag/v0.1.1) is the first
tagged release. It replaces that executable with a native command line for
creating a database, importing JSON records, querying a range and computing
statistics or market indicators. Running it without arguments prints help.

HOCDB is my other Zig project alongside [ZINC](/zinc/). ZINC runs models on local
GPUs. HOCDB stores timestamped records and computes analytics close to the
application. Both projects are about giving local software a useful, direct
interface to the machine it runs on.

## From a record to a result

Download the archive for Linux x86_64 or macOS Apple Silicon from the release
page, verify the checksum and extract it. The binary lives at bin/hocdb. After
putting it on your PATH, this is a complete first session:

~~~bash
hocdb init --db ./market --preset ticks
echo '[{"timestamp":1700000000,"price":100,"volume":2},{"timestamp":1700000060,"price":110,"volume":3}]' \
  | hocdb append --db ./market
hocdb stats --db ./market --field price
~~~

The result contains a count of two and a mean price of 105. The schema and
timestamp unit are saved with the database, so later commands do not need you
to reconstruct them.

The ticks preset has timestamp, price and volume. There is also an OHLCV preset
for candle data and a custom-schema option for other time series, such as
sensor readings. A schema stays fixed once the database exists.

## An interface an agent can inspect

The command line treats its output as an API. Data commands return a JSON
envelope with ok, command and data. Errors return a JSON object on stderr,
including a code and a message that tells the caller what to check.

| An agent needs to… | Command |
| --- | --- |
| Discover commands and argument limits | hocdb describe |
| Inspect schema, units and record count | hocdb info --db ./market |
| Read records from an interval | hocdb query --db ./market --start 1700000000 --end 1700000060 |
| Compute a small statistical summary | hocdb stats --db ./market |
| Read the latest indicator values | hocdb snapshot --db ./market |
| Check gaps and data quality | hocdb health --db ./market |

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
that need more history return null during warm-up. Two records do not produce
a meaningful RSI, and the interface should make that absence clear.

## Keep writes explicit

Database creation requires a new directory, and every database command requires
an explicit path. No command deletes a database. Query, statistics, snapshot
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
automatically overwriting earlier records. The downloaded archives contain
the command line; language bindings still require building the native libraries
with the pinned Zig 0.17.1 compiler. macOS binaries are unsigned.

The [validation and performance report](https://github.com/o1dstaRs/hocdb/blob/v0.1.1/validation/zig17/README.md)
records the earlier engine measurements with their hardware, workloads and
raw samples. This release adds black-box CLI checks for preservation of existing
files, read-only behavior, rejected batches, exact integers, schema presets and
the JSON contract. Release packaging runs the CLI checks again on the extracted
archives.

The practical change is that trying HOCDB now starts with a few ordinary
commands. [Download it and follow the quickstart](/hocdb/), or give an agent
the [command contract](/hocdb/commands.json) and an explicit directory to work in.
