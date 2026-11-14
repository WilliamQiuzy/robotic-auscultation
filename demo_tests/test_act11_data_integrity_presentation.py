"""Collection reports cannot bypass data admission or rewrite model history."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import unittest

DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
ENTRY = 'iteration11-data-integrity'
BASE_ARTICLE_SHA256 = '033d44fa41dec869b543bae18f983507154390e6c092426d4115a9566251a289'


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.items = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


class ACT11DataIntegrityPresentation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()
        match = re.search(r'<section\b[^>]*\bid="' + ENTRY + r'".*?</section>\n', self.html, re.S)
        self.assertIsNotNone(match, 'The new collection case must be one separate historical section')
        self.section = match.group(0)
        self.e = json.loads((DRAFT / 'evidence/act11-source44-data-integrity-20261108.public.json').read_text())

    def test_entire_approved_article_is_byte_identical_outside_single_append(self):
        v10_id = 'iteration12-v10-postfit'
        v10 = re.findall(r'<section\b[^>]*\bid="' + v10_id + r'".*?</section>\n', self.html, re.S)
        self.assertEqual(len(v10), 1, 'V10 must remain one explicitly approved append')
        ids = [attrs.get('id') for _, attrs in Elements(self.html).items]
        self.assertEqual(ids.count(v10_id), 1)
        self.assertEqual(ids.count('iteration11-source48-results'), 1)
        self.assertLess(self.html.index('id="iteration11-source48-results"'), self.html.index('id="' + v10_id + '"'))
        self.assertLess(self.html.index('id="' + v10_id + '"'), self.html.index('id="evaluation"'))
        v11 = re.findall(r'<section\b[^>]*\bid="iteration13-source53-balanced".*?</section>\n', self.html, re.S)
        self.assertEqual(len(v11), 1, 'Source53 must remain one separate append')
        source56 = re.findall(r'<section\b[^>]*\bid="iteration13-source56-expert-data".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source56), 1, 'Source56 must remain one separate append')
        source59 = re.findall(r'<section\b[^>]*\bid="iteration14-source59-near-fit".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source59), 1)
        source62 = re.findall(r'<section\b[^>]*\bid="iteration16-source62-field".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source62), 1)
        source58 = re.findall(r'<section\b[^>]*\bid="iteration13-source58-l3-repair".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source58), 1, 'Source58 must remain one separate append')
        historical_html = self.html.replace(v10[0], '', 1).replace(v11[0], '', 1).replace(source56[0], '', 1).replace(source58[0], '', 1).replace(source59[0], '', 1).replace(source62[0], '', 1)
        historical_html = re.sub(r'<section\b[^>]*\bid="iteration17-source67-known4".*?</section>\n', '', historical_html, flags=re.S)
        historical_html = re.sub(r'<section\b[^>]*\bid="iteration17-source99-approach-feedback".*?</section>\n', '', historical_html, flags=re.S)
        followup = re.search(r'<section\b[^>]*\bid="iteration11-source48-results".*?</section>\n', historical_html, re.S)
        self.assertIsNotNone(followup, 'Source48 must remain one later append')
        historical_html = historical_html.replace(followup.group(0), '', 1)
        later = re.search(r'<section\b[^>]*\bid="iteration11-source45-data".*?</section>\n', historical_html, re.S)
        self.assertIsNotNone(later, 'Source45 must be one later append, preserving the Source44 history')
        self.assertLess(self.html.index('id="iteration11-source45-data"'), self.html.index('id="iteration11-source48-results"'))
        approved = historical_html.replace(later.group(0), '', 1)
        self.assertEqual(hashlib.sha256(approved.encode()).hexdigest(), 'ed09a416a33f93e65de0aef7dd9b7639856382f98a1f6393b657b17e2bf07a84')
        original = approved.replace(self.section, '', 1)
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), BASE_ARTICLE_SHA256)
        self.assertEqual(self.html.count('id="' + ENTRY + '"'), 1)
        self.assertLess(self.html.index('id="iteration10-closed"'), self.html.index('id="' + ENTRY + '"'))
        self.assertLess(self.html.index('id="' + ENTRY + '"'), self.html.index('id="evaluation"'))

    def test_worker_reports_do_not_admit_eight_missing_witnesses_or_one_rejected_capture(self):
        previous = self.e['source43_collection']
        smoke = self.e['source44_collection']
        self.assertEqual((previous['executed'], previous['worker_reported_positive'], previous['data_qualified']), (8, 8, 0))
        self.assertTrue(previous['normal_owner_closed'] and previous['raw_preserved'])
        self.assertFalse(previous['native_STOP_witness_present'])
        self.assertEqual((smoke['original_declared_slots'], smoke['executed'], smoke['intentionally_unlaunched'], smoke['data_qualified']), (8, 1, 7, 0))
        self.assertEqual((smoke['body'], smoke['pose'], smoke['target'], smoke['split']), ('p000758', '476', 'R3', 'TRAIN'))
        self.assertTrue(smoke['normal_owner_closed'] and smoke['worker_reported_positive'] and smoke['independent_STOP_witness_passed'])
        self.assertTrue(all(smoke['strict_same_source_checks'].values()))
        self.assertFalse(smoke['full_reader_passed'])
        self.assertEqual(self.e['model_status'], {'ACT10_known_qualified': 2, 'ACT10_known_denominator': 4, 'ACT10_accepted': False, 'ACT11_new_training_started': False, 'fresh_cases_tested': 0})

    def test_mixed_precision_diagnostic_changes_no_physical_standard(self):
        d = self.e['pose_diagnostic']
        self.assertEqual(d['captured_samples'], 482)
        self.assertEqual(d['max_position_norm_m'], 4.5215639763690366e-7)
        self.assertEqual(d['max_rotation_rad'], 1.3091678227151119e-6)
        self.assertEqual(d['rejected_mixed_matrix_absolute_tolerance'], 1e-10)
        self.assertLess(d['boundary_signed_slider_m'], 0)
        repair = self.e['reader_repair']
        self.assertTrue(repair['separate_length_and_angle'] and repair['preserve_actual_signed_compression'])
        self.assertFalse(repair['same_source_checks_relaxed'] or repair['physical_standards_changed'] or repair['PhysX_accuracy_proven'] or repair['new_capture_qualification_claimed'])
        self.assertTrue(all(v is False for v in self.e['unsupported_claims'].values()))

    def test_actual_media_hashes_dimensions_and_caption_keep_rejected_data_visible(self):
        media = self.e['media']; v = media['video']; p = media['poster']
        self.assertEqual((v['frames'], v['fps'], v['recorded_size']), (2093, 20, [256, 144]))
        self.assertAlmostEqual(v['encoded_s'], v['frames'] / v['fps'])
        self.assertAlmostEqual(v['native_last_time_s'], 104.65833333333333)
        self.assertTrue(v['full_decode_passed'] and v['all_captured_frames_encoded'] and v['lossy_h264'])
        self.assertFalse(v['re_rendered'] or v['preencode_RGB_identity_claimed'])
        for asset in [v, p]:
            raw = (DRAFT / asset['file']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (asset['bytes'], asset['sha256']))
        self.assertEqual(struct.unpack('>II', (DRAFT / p['file']).read_bytes()[16:24]), (256, 144))
        self.assertEqual((p['frame_index'], p['native_time_s'], p['phase_inferred']), (800, 40.05, False))
        items = Elements(self.section).items
        video = next(a for tag, a in items if tag == 'video')
        self.assertIn('controls', video); self.assertNotIn('autoplay', video)
        self.assertEqual((video['preload'], video['poster']), ('metadata', p['file']))
        self.assertIn('warm student + expert intervention; data rejected; not trained-model acceptance', self.section)
        for tag, attrs in items:
            for key in ['href', 'src', 'poster']:
                ref = attrs.get(key, '')
                if ref and not ref.startswith(('#', 'https://')):
                    self.assertTrue((DRAFT / ref).is_file(), ref)

    def test_public_evidence_is_small_sanitized_and_bound_to_closed_source(self):
        text = json.dumps(self.e)
        self.assertLess(len(text), 9000)
        for private in ['/Users/', '/opt/', '/workspace/', '/mnt/', '.npz', '.npy', 'ssh', 'credential']:
            self.assertNotIn(private, text)
        self.assertIsNone(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))
        self.assertEqual(self.e['source44_collection']['source_code_commit'], 'a3e7b0145fc5c8cb1b531749868c59e086659777')
        self.assertEqual(self.e['source_artifact_digests']['source44_media_COMPLETE']['sha256'], '6b3ca19bf04ba52e1e616526a3632a5b9d435dba1593e4b1dd0d03e8c058ecbf')
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', v['sha256']) for v in self.e['source_artifact_digests'].values()))


if __name__ == '__main__':
    unittest.main()
