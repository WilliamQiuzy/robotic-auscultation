# Research demo design

This revision applies [Elaya Design's landing page skill](https://github.com/elayadesign/ai-design-skills) and its [existing project companion](https://github.com/elayadesign/redesign-skill) to the static research preview. It keeps the existing vanilla HTML, CSS and JavaScript stack.

The intended audience is researchers inspecting the released Isaac Sim evidence. The primary action is **Watch the recordings**, repeated at the close. The existing long form research narrative remains: recordings, evidence summary, learning loop and array contract, task acceptance evidence, Astra prototype, research questions.

## Diagnosis and changes

- The previous page mixed a serif display face with a system sans face and used metadata as small as 8 px. This revision declares one design typeface, self hosted Manrope, and reusable named text styles. Metadata begins at 12 px. Browser glyph fallback covers scripts absent from Manrope.
- Spacing resolves through the skill's named scale. Flat paper and neighboring sage surfaces retain the project palette. The hero uses the prescribed text gradient; backgrounds have no gradients.
- The fixed edge header becomes a compact floating bar. At small widths it opens a full height native dialog with an accessible name, first and last focus wrapping, Escape handling, and focus restoration.
- Both actual video files remain unchanged. The viewer adds loading and buffering feedback, a specific inline error, retry, and a direct file link on error. Video switches pause the previous clip, update the caption and evidence link, and retain the language choice.
- A dedicated two line research principle separates model fitting from task acceptance. Words reveal once in reading order using IntersectionObserver. Sections reveal once. Reduced motion exposes the final text without movement.
- Six plain question and answer entries clarify simulation scope, diagnostic cohorts, ACT inputs, offline error, Astra's request only status, and reproduction limits.
- Social metadata, a canonical URL, a branded 404, and factual preview notes complete the static site. The 404 base path supports nested GitHub Pages URLs.

## Research boundaries

The original evidence JSON, benchmark protocol, recordings, plots and Python modules are unchanged. The page retains expert qualification **8/8**, ACT **0/2** on **1 TRAIN + 1 selection VAL** scene, two completed **5,000** update runs, every displayed action error and physical threshold, and the absence of an independent held out result. Astra remains a skill request and local validation prototype with no reported simulator execution score. The unavailable R3 recording has no link.

The skill's commercial sales elements are inapplicable here. No trial, testimonial, legal guarantee, invented privacy policy, account form or cookie banner is added. The preview notes describe the observable static source and point to the actual font license.

## Verification

- `node --check docs/app.js`
- `/Users/williamqiu/opt/anaconda3/bin/python -m unittest discover -s demo_tests -v` (15 existing CPU tests)
- Fresh Chromium sessions at desktop and 390 px: both clips advance during playback; expected durations are 104.05 s and 17.7 s; language switching preserves the selected recording and evidence.
- Mobile menu: full viewport panel, keyboard traversal, Escape, anchor navigation and current location indicator.
- Reduced motion, suspended media loading, aborted media request and retry recovery, no JavaScript fallback, local files and fragments, external links, and both support pages.

Manrope is distributed under the bundled [SIL Open Font License](assets/fonts/OFL.txt). The original font is from the [Google Fonts repository](https://github.com/google/fonts/tree/main/ofl/manrope).
