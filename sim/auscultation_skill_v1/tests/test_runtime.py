"""Timeout cleanup must reap its owned worker even without process enumeration."""
import subprocess
import unittest
from unittest.mock import Mock, call, patch

import psutil

from sim.auscultation_skill_v1.runtime import stop_process_tree, exit_after_cleanup


class RuntimeCleanupTests(unittest.TestCase):
    def test_enumeration_denied_still_terminates_and_reaps_owned_child(self):
        for error in (PermissionError("process enumeration denied"), psutil.AccessDenied(123)):
            with self.subTest(error=type(error).__name__):
                child=Mock(pid=123)
                child.poll.return_value=None
                child.wait.return_value=-15
                with patch("psutil.Process",side_effect=error), patch("psutil.wait_procs",return_value=([],[])):
                    stop_process_tree(child)
                child.terminate.assert_called_once_with()
                child.wait.assert_called_once_with(timeout=5.)
                child.kill.assert_not_called()

    def test_enumeration_denied_and_term_ignored_still_kills_and_reaps(self):
        child=Mock(pid=123)
        child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired("owned-worker",5.),-9]
        with patch("psutil.Process",side_effect=PermissionError("process enumeration denied")), \
                patch("psutil.wait_procs",return_value=([],[])):
            stop_process_tree(child)
        self.assertEqual(child.mock_calls,[call.poll(),call.terminate(),call.wait(timeout=5.),call.kill(),call.wait()])


class PostCleanupExitTests(unittest.TestCase):
    def test_both_task_codes_flush_before_controlled_exit(self):
        for code in (0,1):
            events=[]
            def exit_process(value):
                events.append(('exit',value)); raise SystemExit(value)
            with patch('sys.stdout.flush',side_effect=lambda:events.append('stdout')), \
                    patch('sys.stderr.flush',side_effect=lambda:events.append('stderr')), \
                    patch('os._exit',side_effect=exit_process):
                with self.assertRaises(SystemExit) as caught: exit_after_cleanup(code)
            self.assertEqual(caught.exception.code,code)
            self.assertEqual(events,['stdout','stderr',('exit',code)])

    def test_invalid_codes_never_call_process_exit(self):
        with patch('os._exit') as process_exit:
            for code in (True,None,2,0.):
                with self.assertRaises(ValueError): exit_after_cleanup(code)
        process_exit.assert_not_called()


if __name__=="__main__": unittest.main()
