# Append the later matched eight-scenario results

This is a preparation template, **not a new result release**. The live result JSON, homepage, README and two historical clips remain unchanged. The separate TRAIN expert recording in the November 5 draft is not one of these frozen cases.

## Fixed scope

`matched-eight-results-v1.template.json` binds the actual pre-outcome `CASES.json` receipt (SHA256 `5f9c24bc2c10e6a1cb27d859c81a3bc3c80327c63e5331fe11aaa773b72352f9`): two frozen identities, two existing poses, R3/L3, eight scenarios, three methods, **24 method slots**. The source8 plan requests expert/ACT/Astra parallel workers **3/3/1**; this is planned context until the frozen protocol and actual resource receipts are bound. Source config hashes are already populated; result fields, runtime receipts, metrics and video links are null/pending. Preparing a case does not qualify its native execution.

The two geometrically unavailable slots in the running **20-slot TRAIN/selection collection** belong to a different denominator. Keep that collection ledger intact. Do not inject those slots, the earlier one-episode TRAIN recording, or historical expert/ACT cohorts into this frozen eight-case artifact.

The template authorizes no repeat, alternate pose/target, new seed or post-outcome cohort replacement. Preserve failures and missing receipts. A changed comparison scope needs a separate declared protocol and a separate artifact; it must not overwrite this manifest.

## Evidence required for each method slot

- Exact source scenario and derived run config; immutable source inventory; reconstruction/reference/scene binding; actual run ID.
- Terminal original owner receipt and independent original native audit before any boolean physical verdict. A partial worker result alone leaves `physical_success` null in this matched release artifact.
- Actual physical normal-force and plane hold, forbidden/total forces, tilt, stroke, original Hall proxy, return-home and violation metrics. Decoded spring-load estimates are not substituted for surface-normal force.
- ACT: actual selected checkpoint hash and native learned-L2 provenance. Astra: exact `gpt-6-astra`, actual model call count and native admission/dispatch evidence, wall-time cost and whether physics paused; motors remain analytic `GivenContactExpert`. Unknown token/cost values stay null.
- Video: actual recorded Isaac RGB frames, original source/recording receipt, matching SHA256/byte count, dimensions, frame rate/count and duration. Record simulation-time playback and inference-pause semantics. No generated or replacement frames. If unavailable, set the explicit reason and keep the asset reference null.

The JSON Schema is a publication guard. It checks shape, the fixed cases/source hashes, method attribution, null states, basic gate summaries and required receipts. **It does not authenticate receipt bytes, reconstruct raw physics, replace the original native audit, or prove a claimed video came from a run.** Those checks stay with the existing experiment owner and review workflow.

## Prepare and validate without publishing

From the public repository root, copy the pending template to a new review file:

```sh
cp benchmarks/matched-eight-results-v1.template.json /tmp/rebuilt-matched-eight-review.json
```

Populate only values read from the actual terminal run/audit/video receipts. Retain `publication_status: draft` during review. A local/native artifact can be represented by its logical name, byte count and digest with `public_path: null`; do not copy its private absolute path into this release. Only reviewed public summary/media assets receive a relative public path. Raw arrays, human meshes/data, weights, server addresses, `.env` files and credentials are excluded.

Use a CPU Python environment with `jsonschema`; the original demo tests below also require NumPy:

```sh
python3 - <<'PY'
from pathlib import Path
import json
from collections import Counter
from jsonschema import Draft202012Validator, FormatChecker

schema = json.loads(Path('benchmarks/matched-eight-results-v1.schema.json').read_text())
record = json.loads(Path('/tmp/rebuilt-matched-eight-review.json').read_text())
Draft202012Validator.check_schema(schema)
Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
for method in record['methods']:
    slots = [case['arms'][method] for case in record['scenarios']]
    states = Counter(slot['execution_status'] for slot in slots)
    audited = [slot for slot in slots if slot['physical_success'] is not None]
    successes = sum(slot['physical_success'] is True for slot in audited)
    # Do not turn missing/running cases into observed model failures or a final rate.
    final = all(slot['execution_status'] in
                {'completed', 'failed', 'aborted', 'unavailable', 'not_launched', 'missing_receipt'}
                for slot in slots)
    print(method, dict(planned_slots=len(slots), states=dict(states),
          audited_verdicts=len(audited), successes=successes,
          final_planned_slot_rate=successes / len(slots) if final and len(audited)==len(slots) else None))
PY
```

Report the fixed planned denominator and the actual execution/audit coverage separately. Report preclassified geometric absences, infrastructure aborts, missing results, policy failures and physical rejections distinctly. A rate on only audited/runnable cases must name that conditional denominator and must accompany the full eight-slot accounting. A partial artifact cannot establish a completed matched leaderboard.

Check the historical release and portable interfaces separately:

```sh
python3 -m unittest discover -s demo_tests -v
python3 -m demo.benchmark --verify-assets --output /tmp/historical-assets-rechecked.json
```

## Later append, after actual evidence review

Only after independently checking the source/runtime/data/model/native receipts, actual video content and privacy boundary, mark the new artifact `reviewed_for_append`. Use a fresh destination; do not replace a previous evidence file:

```sh
mkdir -p docs/data/matched
# The destination must be absent; exclusive creation prevents accidental replacement.
python3 - <<'PY'
from pathlib import Path
import json
from jsonschema import Draft202012Validator, FormatChecker
src = Path('/tmp/rebuilt-matched-eight-review.json')
raw = src.read_bytes()
record = json.loads(raw)
assert record['publication_status'] == 'reviewed_for_append'
schema = json.loads(Path('benchmarks/matched-eight-results-v1.schema.json').read_text())
Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
dst = Path('docs/data/matched/rebuilt-frozen-eight-20261105-v1.json')
with dst.open('xb') as stream:
    stream.write(raw)
PY
```

This command copies a reviewed file; it does not validate, publish, update the homepage or approve a result. Add only the reviewed small summaries and actual MP4 assets to the later change. Recheck every relative link, media checksum and desktop/mobile playback before connecting the new artifact and article to the live Elaya page. Keep the old release as a historical cohort. Publishing remains a separate explicitly authorized step.
