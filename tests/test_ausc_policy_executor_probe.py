"""CPU protocol sentinels; fake trajectories are not native Ruckig evidence."""
import importlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class ProbeTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('sim.ausc_policy.executor_probe')
        except ModuleNotFoundError:
            self.fail('The CPU executor compatibility probe is not implemented')

    def source(self, n=32):
        actions = np.zeros((n, 7)); actions[:, 0] = np.arange(1, n+1)*.0005
        state = np.zeros((n, 41)); state[:, 0] = np.arange(n)*.0005
        return dict(actions=actions, state=state, times=np.arange(n, dtype=np.float64)*.05,
                    phases=np.full(n, 'FREE'), targets=np.full(n, 'R4'),
                    target_sources=np.full(n, 'rgbd_surface_tracking'))

    def test_complete_h10_e5_never_pads_or_bridges_missing_origins(self):
        m=self.module(); data=self.source()
        segments=m.make_segments(data, list(range(8))+list(range(18, 23)))
        self.assertEqual([s['query_origins'] for s in segments], [[0, 5], [18]])
        self.assertEqual([s['executed_indices'] for s in segments], [list(range(10)),list(range(18,23))])
        self.assertEqual([s['unused_tail_actions'] for s in segments], [7,9])
        self.assertEqual([s['unused_tail_valid_window_origins'] for s in segments], [2,4])
        self.assertEqual(segments[0]['source_stop_exclusive'], 17)

    def test_phase_time_target_and_nan_corruption_are_rejected(self):
        m=self.module()
        for field,index,value in [('phases',4,'APPROACH'),('targets',3,'L4'),
                                  ('target_sources',5,'expert'),('times',6,.301)]:
            data=self.source(); data[field][index]=value
            with self.subTest(field=field),self.assertRaises(ValueError): m.make_segments(data,[0])
        data=self.source();data['actions'][2,0]=np.nan
        with self.assertRaises(ValueError):m.make_segments(data,[0])
        with self.assertRaises(ValueError):m.make_segments(self.source(),[23])

    def test_forward_velocity_uses_same_chunk_lookahead_and_never_clips(self):
        m=self.module(); chunk=self.source()['actions'][:10]
        velocity=m.derive_velocity(chunk)
        np.testing.assert_allclose(velocity[:,0],.01,atol=1e-14)
        np.testing.assert_array_equal(velocity[:,1:],0.)
        bad=chunk.copy();bad[5,0]=1.
        with self.assertRaisesRegex(ValueError,'velocity'):m.derive_velocity(bad)
        with self.assertRaises(ValueError):m.derive_velocity(chunk[:5])

    def trajectory(self, instances, reject=False):
        class Trajectory:
            def __init__(self,q,**limits):
                self.q=np.asarray(q).copy();self.velocity=np.zeros(7);self.acceleration=np.zeros(7)
                self.input=SimpleNamespace(target_velocity=[0.]*7,target_acceleration=[0.]*7)
                self.validations=[]
                def validate(*args,**kwargs):
                    self.validations.append(kwargs);return not reject
                self.generator=SimpleNamespace(validate_input=validate)
                self.targets=[];self.ticks=0;self.limits=limits;instances.append(self)
            def set_target(self,q):self.target=np.array(q);self.targets.append(self.target.copy())
            def step(self):
                self.ticks+=1;self.q=self.target.copy();return self.q.copy()
        return Trajectory

    def test_replay_keeps_one_trajectory_and_labels_at_t_plus_50ms_once(self):
        m=self.module();data=self.source();segments=m.make_segments(data,list(range(8)))
        instances=[]
        result=m.replay(data,segments,'zero_terminal_velocity',trajectory_factory=self.trajectory(instances))
        self.assertTrue(result['completed']);self.assertEqual(len(instances),1)
        self.assertEqual(instances[0].ticks,120)
        self.assertEqual(instances[0].limits,dict(dt=1/240,velocity=.175,acceleration=.5,jerk=7.99999))
        np.testing.assert_allclose(result['arrays']['label_time_s'],np.arange(1,11)*.05)
        np.testing.assert_array_equal(result['arrays']['teacher_target_q'],data['actions'][:10])
        np.testing.assert_array_equal(result['arrays']['command_q'],data['actions'][:10])
        np.testing.assert_array_equal(result['arrays']['teacher_measured_q_at_label'],data['state'][1:11,:7])

    def test_ff_only_changes_target_velocity_and_rejects_invalid_native_input(self):
        m=self.module();data=self.source();segments=m.make_segments(data,[0])
        instances=[]
        result=m.replay(data,segments,'adjacent_target_velocity',trajectory_factory=self.trajectory(instances))
        self.assertTrue(result['completed'])
        self.assertEqual(instances[0].validations[0],dict(check_current_state_within_limits=False,
                                                         check_target_state_within_limits=True))
        np.testing.assert_allclose(instances[0].input.target_velocity,[.01,0,0,0,0,0,0],atol=1e-14)
        np.testing.assert_array_equal(instances[0].input.target_acceleration,np.zeros(7))
        result=m.replay(data,segments,'adjacent_target_velocity',trajectory_factory=self.trajectory([],reject=True))
        self.assertFalse(result['completed']);self.assertEqual(result['arrays']['source_index'].size,0)
        self.assertIn('validation',result['failure']['reason'])

    def test_metrics_separate_startup_and_subsequent_endpoints(self):
        m=self.module();data=self.source(45);segments=m.make_segments(data,list(range(31)))
        result=m.replay(data,segments,'zero_terminal_velocity',trajectory_factory=self.trajectory([]))
        class Kin:
            def tcp(self,q):
                pose=np.eye(4);pose[:3,3]=q[:3];return pose
        stats=m.metrics(result['arrays'],Kin())
        self.assertEqual(stats['all']['samples'],35)
        self.assertEqual(stats['first_1s']['samples'],20)
        self.assertEqual(stats['after_1s']['samples'],15)
        self.assertEqual(stats['all']['command_vs_teacher_target']['joint_mae_rad'],0.)
        self.assertEqual(stats['all']['command_vs_teacher_target']['tcp_position_m']['max'],0.)

    def test_loader_opens_only_pinned_r4_cache_and_refuses_changed_bytes(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);data=self.source();bindings={}
            for key,value in data.items():
                path=folder/(key+'.npy');np.save(path,value)
                bindings[key]=dict(path=str(path),sha256=m.a.sha256(path))
            row=dict(episode_id=m.EPISODE,split='train',cache=bindings,starts=list(range(8)),
                boundary=dict(stationary_handoff_proved=True,boundary_source='raw align to contact_cycle, before all contact actions',
                              first_contact_physics_row=999,verified_label_pairs=32,l2_label_pairs=32),
                source_record=dict(episode_id=m.EPISODE,split='train',target='R4'))
            receipt=dict(schema='ausc_m0_act_preparation_v1',horizon=10,execute_steps=5,period_s=.05,
                action_representation='absolute_joint_radians',heldout_arrays_opened=False,image_arrays_opened=False,
                episodes=[row,dict(episode_id='R5_never_open',cache={'state':{'path':'/must/not/open'}})])
            path=folder/'prepared.json';path.write_text(json.dumps(receipt))
            config=dict(prepared_manifest=str(path),prepared_manifest_sha256=m.a.sha256(path),episode_id=m.EPISODE)
            arrays,chosen,provenance=m.load_source(config)
            self.assertEqual(chosen['episode_id'],m.EPISODE)
            self.assertTrue(all(isinstance(x,np.memmap) and not x.flags.writeable for x in arrays.values()))
            self.assertFalse(provenance['heldout_arrays_opened'])
            with (folder/'actions.npy').open('ab') as stream:stream.write(b'x')
            with self.assertRaisesRegex(ValueError,'cache'):m.load_source(config)

    def test_native_dependency_hash_is_checked_before_import_and_path_must_match(self):
        m=self.module()
        self.assertTrue(callable(getattr(m,'load_native',None)), 'Native dependency loader is missing')
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);library=folder/'ruckig.so';library.write_bytes(b'fake test module')
            config=dict(dependency_paths=[str(folder)],dependency_files={str(library):m.a.sha256(library)})
            fake=SimpleNamespace(__version__='0.12.2',__file__=str(library))
            with patch.dict(sys.modules,{'ruckig':fake}),patch.object(sys,'path',sys.path.copy()):
                self.assertIs(m.load_native(config),fake)
                fake.__file__=str(folder/'other.so')
                with self.assertRaisesRegex(ValueError,'resolved'):m.load_native(config)
                fake.__file__=str(library);library.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'dependency'):m.load_native(config)

    def test_finite_run_writes_small_arrays_and_never_claims_physics_or_native_fake_success(self):
        m=self.module();data=self.source();row=dict(starts=list(range(8)));instances=[]
        self.assertTrue(callable(getattr(m,'load_native',None)), 'Native dependency loader is missing')
        real_replay=m.replay
        def fake_replay(*args,**kwargs):
            return real_replay(*args,**kwargs,trajectory_factory=self.trajectory(instances))
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);library=folder/'fake.so';library.write_bytes(b'fixture')
            native=SimpleNamespace(__version__='0.12.2',__file__=str(library))
            with patch.object(m,'load_source',return_value=(data,row,{'fixture':True})),\
                    patch.object(m,'load_native',return_value=native),patch.object(m,'replay',side_effect=fake_replay):
                report=m.run(dict(experiment='executor_probe',mode='teacher_replay',device='cpu',time_limit_seconds=600),folder)
            self.assertTrue(report['comparison']['full_identical_schedule_completed'])
            self.assertEqual(len(instances),2)
            for value in m.FLAGS:self.assertFalse(report[value])
            self.assertTrue(all(not b['native_trajectory'] for b in report['branches'].values()))
            self.assertLess((folder/'endpoints.npz').stat().st_size,20000)
            with np.load(folder/'endpoints.npz') as arrays:
                self.assertEqual(arrays['zero_terminal_velocity__command_q'].shape,(10,7))


if __name__=='__main__':unittest.main()
