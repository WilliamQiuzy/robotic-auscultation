# source26 partial public capsule

Read the [reproduction guide](../../docs/SOURCE26_REPRODUCTION.md) before running
anything. `SOURCE_SELECTION.json` is a **PARTIAL_PUBLIC_CAPSULE**, not the complete
original runtime manifest or an owner/physics qualification.

```sh
python3 -I -B repro/source26/verify_capsule.py
python3 -I -B repro/source26/verify_capsule.py --fixtures
```

Run those commands from the repository root with your CPU test environment's
Python. The fixture adapter starts an isolated subprocess rooted here and checks
actual project import paths. It selects only nine original tests, retaining their
skip reasons. The original source files are unchanged; the adapter is new public
tooling. Source19 at the repository root and historical inventories stay intact.
