"""Closed V8 accounting, privacy, exact media bytes and historical preservation."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import unittest

DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
HISTORICAL_SECTIONS = {'coverage': 'c69d68eb0433aec7c8db301e0b0683f657ad0f109486055f1462ad589d3ff4ce', 'prefix': '31ce2f635004f78f7dad4bb75a397b61b4bead311393b9d8a6a028270c852eb5', 'recovery': '0f383406cb07f07be5a349ef5d89039a8dbd75866ad066e473555529c763a87a', 'training': 'a59d74660ff0db6f13bc9099fdab466e5c67e067016e8d7ed09d9c14a383e3a4', 'native-negative': '40f19ba63d7ecc9178639fa377694aa31a9e3c2bb8c68b2e49b1d8a3e5d7783a', 'student-states': 'fe77e00469c4d1e065f79910df54418807d8c33d579bcf994739dc2ff99e91ad', 'iteration3': '102d642580b9f6e80e3747798aaa5dde71328007314a5093718fdcff06dde3b3', 'iteration4': 'f2f5c80ef53fedc3ca36b09187ac940b473aa25df043a9ee436c1d903600ea2a', 'v3-preparation': '0f5f68dfe5a9f4efc0f75e464ab2c2f2a886d0947f4b411e3c9bd17171c57f8a', 'iteration5': 'afabed93ca1f61daa2fd76df17faa884702fdea07f207d2c1a3fe02249b64995', 'iteration6': 'c5d726aea43245e3cf2a591e32ebe4ea5c1b19b6d173e079132d437ef6f3b695', 'iteration7': 'e32c74d14b82da541bcfa6a7600b8e1eea69ca33d569013022b38422fc8f1fa0', 'measured-recovery': 'dc44d0de50aeeb1b5faa7be8402f4fc6b346747e5944c6d3e16bff260dc6d36e', 'iteration9': 'aeae66b7a1b0c7dcebc3c89f049c9d1d2afc59f6e50ded1af39effc55e58a01f', 'iteration10': '76e4201f7f5f73cd6f121b6455436118444d7bdc7655df23996feebc14ede09f', 'evaluation': '2b53c112a331fd3ab1ba6eb6301036d91e1a194f75cb0d7e1451fc787c8a2cb6', 'recordings': '678d540b8019482d7f18948f8d1bde37e668c74ce8bea91592cbc8300c3c4728', 'evidence': '00a2cc46adede057e01747de5f07c5a95b7df93756b900a3b8c29c8f8087dad8'}

class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.items = []
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))

class ACT10ClosedPresentation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()
        self.e = json.loads((DRAFT / 'evidence/act10-v8-closed-20261108.public.json').read_text())

    def test_all_40_full_population_evaluations_select_earliest_minimum(self):
        t = self.e['training']; rows = t['all_full_native_evaluations']
        self.assertEqual([r['step'] for r in rows], list(range(500, 20001, 500)))
        self.assertTrue(all((r['train_samples'], r['validation_samples']) == (8373, 2345) for r in rows))
        selected = min(rows, key=lambda r: (r['validation_full_H20_masked_raw_MAE_rad'], r['step']))
        self.assertEqual(selected['step'], 17000)
        self.assertEqual(selected['validation_full_H20_masked_raw_MAE_rad'], 0.009497999548919436)
        self.assertEqual(rows[-1]['validation_full_H20_masked_raw_MAE_rad'], 0.011823793857057317)
        self.assertEqual(t['new_updates'], 20000)
        self.assertTrue(t['normal_owner_closed'] and t['own_full_CPU_equivalence_passed'])
        self.assertFalse(t['offline_target_error_is_physical_acceptance'])

    def test_fixed_case_matrix_and_untested_fresh_gate(self):
        d = self.e['known_development']
        self.assertEqual([(c['body'],c['split'],c['target'],c['physical_pass']) for c in d['cases']], [('p001120','TRAIN','R3',True),('p001120','TRAIN','L3',False),('p000923','selection_VAL','R3',True),('p000923','selection_VAL','L3',False)])
        self.assertEqual((d['fixed_denominator'],d['executed_cases'],d['independently_qualified']), (4,4,2))
        self.assertTrue(d['selection_VAL_used_for_checkpoint_selection'])
        self.assertFalse(d['aggregate_improvement_claim'])
        self.assertEqual(self.e['fresh_benchmark'], {'declared_scenarios':32,'tested':0,'gate_open':False,'payloads_opened':False})

    def test_distinct_failures_do_not_turn_CPU_candidate_into_physical_result(self):
        a = self.e['failure_diagnoses']['TRAIN_L3']; l = self.e['failure_diagnoses']['selection_VAL_L3']
        self.assertEqual((a['reconstructed_valid_IK_knots'],a['reconstructed_total_knots'],a['original_CPU_checker_calls']), (12,13,563))
        self.assertFalse(a['original_knots_saved'] or a['original_command_bit_exact_reconstruction_claim'] or a['executed_in_Isaac'] or a['repair_physically_qualified'] or a['physical_limits_changed'])
        self.assertEqual(a['internal_IK_reserve_candidate_rad'],0.0001)
        self.assertEqual((l['accepted_model_queries'],l['refused_model_queries'],l['issued_actions'],l['STOP_setters']), (241,0,2399,0))
        self.assertGreater(l['best_tangential_error_mm'],l['coarse_tube_mm'])
        self.assertFalse(l['coarse_reached'] or l['parking_entered'] or l['ALIGN_entered'] or l['L3_started'] or l['selection_VAL_labels_exported_or_trained'])
        s = self.e['data_scope']
        self.assertEqual((s['native_Base29_corridor_windows'],s['native_Base29_corridor_bodies'],s['measured_recovery_corridor_windows'],s['measured_recovery_corridor_bodies'],s['new_DART_corridor_windows']), (880,4,53,1,0))
        self.assertFalse(self.e['next_experiment']['new_enrichment_collected'] or self.e['next_experiment']['model_retrained'])

    def test_media_bindings_frames_and_native_coverage(self):
        expected = [('p000923-r3',1075,22),('p000923-l3',1210,1),('p001120-l3',293,6)]
        items = Elements(self.html).items
        for n,(m,(case,frames,trailing)) in enumerate(zip(self.e['verified_media'],expected),1):
            self.assertEqual(m['case_id'],case)
            v = m['video']; p = m['poster']
            self.assertEqual((v['frames'],v['fps'],v['recorded_size'],v['trailing_unrendered_native_ticks']), (frames,10,[640,360],trailing))
            self.assertAlmostEqual(v['encoded_s'],frames/10)
            self.assertAlmostEqual(v['native_s'],v['native_physics_ticks']/240)
            self.assertTrue(v['native_encoded_bytes_identical'] and v['full_decode_passed'] and p['exact_decoded_frame'])
            self.assertFalse(v['preencode_RGB_identity_claim'] or p['phase_inferred'])
            for asset in [v,p]:
                raw=(DRAFT/asset['file']).read_bytes()
                self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(asset['bytes'],asset['sha256']))
            self.assertEqual(struct.unpack('>II',(DRAFT/p['file']).read_bytes()[16:24]),(640,360))
            video = next(a for tag,a in items if tag=='video' and a.get('id')==f'iteration10-closed-video-{n}')
            self.assertIn('controls',video); self.assertNotIn('autoplay',video)
            self.assertEqual((video['preload'],video['poster']),('metadata',p['file']))
        fig=self.e['curve_figure']; raw=(DRAFT/fig['file']).read_bytes()
        self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(150254,'3e6e0bf037c1822531bb22343765932ac79319e282404a501a18cf305b535bb8'))
        self.assertEqual(struct.unpack('>II',raw[16:24]),(fig['width'],fig['height']))
        self.assertEqual(fig['actual_evaluation_rows'],40)
        self.assertFalse(fig['physical_acceptance_claim'])

    def test_all_prior_18_sections_are_byte_identical_and_new_anchor_is_ordered(self):
        self.assertEqual(len(HISTORICAL_SECTIONS),18)
        for name,sha in HISTORICAL_SECTIONS.items():
            section=re.search(r'<section\b[^>]*\bid="'+name+r'".*?</section>',self.html,re.S).group(0)
            self.assertEqual(hashlib.sha256(section.encode()).hexdigest(),sha,name)
        self.assertLess(self.html.index('id="iteration10"'),self.html.index('id="iteration10-closed"'))
        self.assertLess(self.html.index('id="iteration10-closed"'),self.html.index('id="evaluation"'))
        self.assertIn('href="#iteration10-closed"',self.html)

    def test_compact_public_summary_has_no_private_paths_or_raw_payloads(self):
        text=json.dumps(self.e)
        self.assertLess(len(text),23000)
        for private in ['/Users/','/opt/','/workspace/','.npz','.npy','ssh','credential']:
            self.assertNotIn(private,text)
        self.assertIsNone(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',text))
        def check(x):
            if isinstance(x,dict):
                self.assertFalse(set(x)&{'states','actions','joint_positions','raw_arrays','weights','environment'})
                for v in x.values(): check(v)
            elif isinstance(x,list):
                for v in x: check(v)
        check(self.e)
        self.assertTrue(all(v is False for v in self.e['limits'].values()))

if __name__=='__main__': unittest.main()
