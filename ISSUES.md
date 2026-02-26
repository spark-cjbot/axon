# Axon C# Support — Known Issues

Found during initial testing (2026-02-26) against a synthetic C# project with classes, interfaces, enums, generics, and attributes.

---

## 🐛 1. Interface method declarations flagged as dead code

**File:** `src/axon/core/ingestion/dead_code.py`

Interface method stubs (e.g. `User GetUser(int id)` inside `IUserService`) have no callers by definition — they're contracts, not implementations. The dead code phase incorrectly marks them as unreachable.

**Repro:** Any C# interface with method declarations → all methods show up in `axon dead-code`.

**Fix:** Skip nodes where the parent is an interface, or check if the symbol's content looks like an interface stub (no body block). Could also add an `is_abstract` flag in the parser for interface members.

---

## 🐛 2. Constructor stored under the class name (ambiguity)

**File:** `src/axon/core/parsers/csharp.py`

C# constructors are extracted with `name = class_name` (e.g. `UserService`), which is correct C# syntax but causes collisions in the graph — `context("UserService")` returns both the class node and the constructor node.

**Repro:** Any class with an explicit constructor → two nodes with the same name.

**Fix:** Store constructors with a `.ctor` suffix or a `kind=constructor` label so they're distinguishable. The signature already distinguishes them, but the name doesn't.

---

## 🐛 3. Method name collision in call tracing (`GetAll → GetAll`)

**File:** `src/axon/core/ingestion/calls.py`

When a method calls a dependency with the same name (e.g. `UserService.GetAll()` calling `_repo.GetAll()`), the call resolver may create a self-loop because it matches on name alone rather than receiver type.

**Repro:** `UserService.GetAll()` → calls `_repo.GetAll()` → shows as `GetAll → GetAll`.

**Fix:** Improve call resolution to use receiver type information when available. The C# parser already extracts `receiver` in `CallInfo` — use it during the call-tracing phase to narrow matches.

---

## 💡 4. C# attributes not stored as graph properties

**File:** `src/axon/core/storage/kuzu_backend.py`

The C# parser extracts attributes (`[HttpGet]`, `[Route("/api")]`) correctly into `SymbolInfo.decorators`, but the KuzuDB schema has no `decorators` column. They're only available via FTS through the `content` snippet.

**Impact:** Can't do structured queries like "find all methods with `[HttpGet]`" — only text search works.

**Fix:** Add a `decorators STRING` column to `_NODE_PROPERTIES` and serialize the list as JSON. Already handled by Python/TS parsers if they store decorators — worth checking consistency.

---

## 🌐 Multi-language (React + C# monorepo)

Works out of the box — file extension dispatch handles `.tsx/.jsx` → TypeScript parser and `.cs` → C# parser in the same pipeline run, unified into a single KuzuDB graph. No changes needed.
