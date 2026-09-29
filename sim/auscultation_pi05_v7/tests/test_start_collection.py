"""Synthetic CPU checks only; no simulator, acquired data, or qualification."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

import numpy as np
from sim.auscultation_pi05_v7 import start_teacher as teacher_module
from sim.auscultation_pi05_v7 import start_collection as collection
from sim.auscultation_pi05_v6.collect_demo import ObservedTeacher
from sim.auscultation_expert_layout_v1.controller import EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase

Q=np.array([.12,-.06,.06,.12,-.04,.04,0.])


def fixture():
    t=teacher_module.StartObservedTeacher.__new__(teacher_module.StartObservedTeacher)
    t.start_manifest=dict(requested_start_q_rad=Q.tolist(),start_family='train_yaw_positive_v1',
                         target='R4',parent_plan_sha256='a'*64)
    t.start_manifest_sha256='b'*64;t.experiment_protocol_sha256='c'*64;t.episode_id='start_R4_v1'
    t.start_receipt=dict(first_eligible=None);t.start_role='start_observation';t.target=S(name='R4')
    t.manager=S(state=S(phase=Phase.APPROACH));t.env=S(rows=[]);t.sensor=S(rows=[])
    def capture(state,phase,target,row,**kw):
        t.sensor.rows.append(dict(index=len(t.sensor.rows),time_s=state['time_s'],q=state['q'].tolist(),
            dq=state['dq'].tolist(),phase=phase,target=target,physics_row=row,**kw))
    t.sensor.capture=capture;t.q=Q.copy()
    return t


class StartCollectionTests(unittest.TestCase):
    def test_parent_acceptance_field_is_preserved_without_claiming_qualification(self):
        for accepted in (False,True):
            t=fixture()
            with patch.object(ObservedTeacher,'run',return_value=dict(success=accepted,training_accepted=accepted)):
                result=t.run([(None,None)])
            self.assertIs(result['training_accepted'],accepted)
            self.assertFalse(result['qualification_passed']);self.assertFalse(result['dataset_sealed'])

    def test_existing_contact_and_return_methods_are_inherited_exactly(self):
        for name in ('_contact_cycle','desired_normal_velocity','_observed_released','_return_home'):
            self.assertIs(getattr(teacher_module.StartObservedTeacher,name),getattr(ObservedTeacher,name))

    def test_natural_cadence_and_real_first_state_are_preserved(self):
        t=fixture()
        for index in range(24):
            q=Q+index*1e-7;dq=np.arange(7)*.0003
            state=dict(time_s=(index+1)/240,q=q,dq=dq)
            t.env.rows.append(dict(q_command=Q.tolist(),subphase='transfer',q=q.tolist(),dq=dq.tolist()))
            t._capture_observed_tick(state,'transfer',False,False)
        self.assertEqual([r['physics_row'] for r in t.sensor.rows],[11,23])
        first=t.start_receipt['first_eligible'];self.assertEqual(first['frame_index'],0)
        self.assertEqual(first['physics_row'],11)
        self.assertEqual(first['measured_q_rad'],(Q+11e-7).tolist())
        self.assertEqual(first['measured_dq_rad_s'],(np.arange(7)*.0003).tolist())
        self.assertEqual(first['retained_command_q_rad'],Q.tolist())

    def test_preparation_is_excluded_without_relabeling_raw_subphase(self):
        t=fixture();t.start_role='move_to_declared_start'
        for index in range(12):
            t.env.rows.append(dict(q_command=Q.tolist(),subphase='initial_home'))
            t._capture_observed_tick(dict(time_s=(index+1)/240,q=Q,dq=np.zeros(7)),
                                     'initial_home',False,False)
        self.assertFalse(t.sensor.rows[-1]['eligible']);self.assertIsNone(t.start_receipt['first_eligible'])
        self.assertEqual(t.env.rows[-1]['subphase'],'initial_home')
        self.assertEqual(t.env.rows[-1]['preparation_role'],'move_to_declared_start')

    def test_first_frame_must_be_genuinely_nonzero_and_track_request(self):
        for q in (np.zeros(7),Q+.031):
            with self.subTest(q=q),self.assertRaises((ValueError,EpisodeRejected)):
                teacher_module.checked_first_state(dict(time_s=1.,q=q,dq=np.zeros(7)),Q)

    def test_eligible_command_washout_rejected_before_parent_execution(self):
        t=fixture();t.start_role='eligible_start_route'
        with patch.object(ObservedTeacher,'_block') as parent:
            with self.assertRaises(EpisodeRejected):t._block(np.zeros((12,7)),'transfer')
            parent.assert_not_called()

    def test_preparation_can_include_zero_and_final_return_is_unmodified(self):
        t=fixture()
        with patch.object(ObservedTeacher,'_block',return_value='ok') as parent:
            for role,phase in [('move_to_declared_start','initial_home'),(None,'return_home')]:
                t.start_role=role;self.assertEqual(t._block(np.zeros((12,7)),phase),'ok')
            self.assertEqual(parent.call_count,2)

    def test_original_recovery_is_not_blocked_by_eligible_no_home_rule(self):
        t=fixture();t.start_role='eligible_start_route'
        with patch.object(ObservedTeacher,'_block',return_value='recovery') as parent:
            self.assertEqual(t._block(np.zeros((12,7)),'fault_retract',recovery=True),'recovery')
            parent.assert_called_once()

    def test_actually_imported_geometry_cannot_hide_behind_other_source_project(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sim/example.py';path.parent.mkdir();path.write_text('# changed')
            module=S(__file__=str(path))
            with patch.dict(collection.sys.modules,{'sim.test_geometry_binding':module}):
                with self.assertRaisesRegex(ValueError,'Actually imported geometry source drift'):
                    collection.verify_imported_geometry_sources({'sim/example.py':'a'*64})
                collection.verify_imported_geometry_sources({'sim/example.py':collection.sha(path)})

    def test_route_preserves_retained_measured_connector_and_original_suffix(self):
        canonical=np.array([np.zeros(7),Q+.4,Q+.5,Q+.6]);measured=Q+.001
        bridge=np.array([measured,Q+.2,canonical[1]])
        route,index=teacher_module.join_actual_route(Q,measured,bridge,canonical,1)
        np.testing.assert_array_equal(route[0],Q);np.testing.assert_array_equal(route[1],measured)
        np.testing.assert_array_equal(route[index:],canonical[1:])
        with self.assertRaises(ValueError):teacher_module.join_actual_route(Q,measured,
            np.array([measured,np.zeros(7),canonical[1]]),canonical,1)

    def test_receipt_requires_real_first_frame_and_final_sensor_hash(self):
        t=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'frames.jsonl';p.write_text('{}\n')
            with self.assertRaises(ValueError):t.finalize_start_receipt(p)
            t.start_receipt['first_eligible']=dict(frame_index=0,physics_row=11)
            r=t.finalize_start_receipt(p)
            self.assertEqual(r['sensor_frames_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertFalse(r['qualification_passed']);self.assertFalse(r['dataset_sealed'])

    def test_first_entry_never_calls_old_home_entry_and_keeps_original_return(self):
        t=fixture();t.start_role=None;t.sequence_index=0;t.q=np.zeros(7)
        state=dict(time_s=0.,q=np.zeros(7),dq=np.zeros(7),hall_N=0.,compression_m=0.)
        t.env.state=lambda:state;t.env.last_contacts={};t.checker=object();events=[]
        canonical=np.array([np.zeros(7),Q+.4,Q+.5,Q+.6])
        row=dict(success=True,transfer_q=canonical.tolist(),lower_staging_q=canonical[1].tolist())
        def execute(destination,phase,ticks):
            events.append((phase,np.asarray(destination).copy()))
            for _ in range(ticks):
                t.q=np.asarray(destination).copy();state.update(time_s=state['time_s']+1/240,q=t.q.copy())
                t.env.rows.append(dict(q_command=t.q.tolist(),subphase=phase))
                t._capture_observed_tick(state,phase,False,False)
        t._joint_segment=lambda q,phase:execute(q,phase,12)
        t._hold=lambda seconds,phase:execute(t.q,phase,round(seconds*240))
        t._observed_released=lambda state:True
        t._plan_inter_point_bridge=lambda q,stage,deadline:(np.array([q,stage]),dict(passed=True))
        t._execute_staged=lambda route,index,initial:execute(route[-1],'lower_approach',12)
        with patch.object(teacher_module,'checked_edge',return_value={'clear':True}),patch.object(
                ObservedTeacher,'_move_to_prepose',side_effect=AssertionError('old home entry called')):
            actual=t._move_to_prepose(row)
        self.assertEqual([e[0] for e in events],['initial_home','settle','transfer','lower_approach','align'])
        np.testing.assert_array_equal(events[0][1],Q)
        np.testing.assert_array_equal(t.canonical_transfer,canonical)
        np.testing.assert_array_equal(actual[0],Q)
        self.assertEqual(t.start_receipt['preparation_physics_range'],[0,12])
        self.assertEqual(t.start_receipt['settle_physics_range'],[12,252])
        self.assertEqual(t.start_receipt['first_eligible']['physics_row'],263)
        self.assertEqual(t.sensor.rows[21]['phase'],'FREE')
        self.assertEqual(t.sensor.rows[21]['subphase'],'transfer')

    def test_bound_start_rejects_protocol_profile_source_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan=root/'plan.json';plan.write_text('{}')
            source=root/'source.py';source.write_text('# frozen')
            inventory=root/'source_sha256.json';inventory.write_text(json.dumps({str(source):collection.sha(source)}))
            curve=root/'curve.json';curve.write_text('{}')
            start=dict(schema='wam_v7_robot_start_geometry_manifest_v1',geometry_preflight_passed=True,
                actual_acquisition_started=False,physics_accepted=False,training_accepted=False,
                actual_measured_start_q_rad=None,target='R4',parent_plan_sha256=collection.sha(plan),
                requested_physics_profile=dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02),
                requested_start_q_rad=Q.tolist(),start_family='train_yaw_positive_v1',split='train',
                proposed_starts_sha256='d'*64,source_sha256=collection.sha(inventory),
                original_contact_curve_file='curve.json',original_contact_curve_sha256=collection.sha(curve),
                limits=dict(free_clearance_m=.003,contact_clearance_m=.0003,joint_edge_sample_step_rad=.015,
                    urdf_planning_margin_deg=3.,command_velocity_rad_s=.175,command_acceleration_rad_s2=1.,
                    command_jerk_rad_s3=1.5,raw_measured_acceleration_rad_s2=2.,raw_measured_jerk_rad_s3=20.))
            path=root/'start.json';path.write_text(json.dumps(start))
            protocol=dict(schema='wam_v7_frozen_start_family_protocol_v1',actual_start_tolerance_rad=.03,
                start_spec_sha256='d'*64,families=[dict(family_id=start['start_family'],split='train',requested_q_rad=Q.tolist())],
                cases={start['start_family']+'::R4':dict(start_manifest_sha256=collection.sha(path),
                    source_plan_sha256=collection.sha(plan),target='R4',start_family=start['start_family'])})
            pp=root/'protocol.json';pp.write_text(json.dumps(protocol,sort_keys=True,separators=(',',':'))+'\n')
            args=S(start_manifest=path,start_manifest_sha256=collection.sha(path),protocol=pp,
                protocol_sha256=collection.sha(pp),episode_id='R4_start_v1',source_project=root,point='R4')
            collection.bound_start(args,S(plan_path=plan))
            source.write_text('# drift')
            with self.assertRaisesRegex(ValueError,'Geometry source drift'):collection.bound_start(args,S(plan_path=plan))
            source.write_text('# frozen');args.protocol_sha256='0'*64
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):collection.bound_start(args,S(plan_path=plan))
            args.protocol_sha256=collection.sha(pp);start['requested_physics_profile']['armature_kg_m2']=.01
            path.write_text(json.dumps(start));args.start_manifest_sha256=collection.sha(path)
            with self.assertRaisesRegex(ValueError,'.02 geometry'):collection.bound_start(args,S(plan_path=plan))

    def test_new_entry_does_not_import_isaac_or_physics_at_module_import(self):
        tree=ast.parse(Path(collection.__file__).read_text())
        imports=[]
        for node in tree.body:
            if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module)
        self.assertFalse(any(x.startswith(('sim.','isaac','omni','pxr')) for x in imports))
        text=Path(collection.__file__).read_text()
        self.assertLess(text.index("app=SimulationApp("),text.index('run_scoped_capture(app,capture'))

    def test_sources_include_original_teacher_and_both_real_v7_files(self):
        sources=collection.collector_source_paths()
        self.assertIn('pi05_v6/collect_demo.py',sources)
        for name in ('start_teacher.py','start_collection.py'):
            self.assertTrue(sources['pi05_v7/'+name].is_file())

    def test_cli_has_no_physics_override_and_needs_bound_start_protocol(self):
        parser=collection.argument_parser()
        with self.assertRaises(SystemExit):parser.parse_args([])
        options={opt for action in parser._actions for opt in action.option_strings}
        self.assertNotIn('--armature',options);self.assertNotIn('--position-iterations',options)
        self.assertIn('--start-manifest-sha256',options);self.assertIn('--protocol-sha256',options)


if __name__=='__main__':unittest.main()
