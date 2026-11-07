"""Normal close for shared compact jobs; no Kit import until invoked.

Installed Isaac 6 APIs: SimulationApp.disable_viewport_updates, Replicator
orchestrator stop/wait, and public ViewportWindow.destroy. Owned camera products
must already be finalized and the world stopped. Never deletes global graphs.
"""
import json
import os
from pathlib import Path

CLEANUP_UPDATES=4
SHARED_CPU_THREADS=6  # SDK tasking/TBB/USD/BLAS budget, not direct PhysX control.


def shared_app_settings():
    return dict(headless=True,fast_shutdown=False,disable_viewport_updates=True,limit_cpu_threads=SHARED_CPU_THREADS)


def quiesce_before_close(app,*,context=None,replicator=None,get_viewport_window=None,progress=None):
    mark=progress or (lambda event:None)
    if context is None:
        import omni.usd
        context=omni.usd.get_context()
    if replicator is None:
        import omni.replicator.core as replicator
    if get_viewport_window is None:
        from omni.kit.viewport.utility import get_active_viewport_window
        get_viewport_window=get_active_viewport_window
    stage=context.get_stage()
    if stage is None:raise RuntimeError('Shared native cleanup requires a live stage')
    orchestrator=replicator.orchestrator
    orchestrator.set_capture_on_play(False)
    mark('rep_stop_enter')
    orchestrator.stop()
    mark('rep_stop_return');mark('rep_wait_enter')
    orchestrator.wait_until_complete()
    mark('rep_wait_return')
    window=get_viewport_window()
    if window is not None:
        viewport=window.viewport_api
        if viewport is not None:viewport.updates_enabled=False
        mark('viewport_destroy_enter')
        window.destroy()
        mark('viewport_destroy_return')
    for index in range(CLEANUP_UPDATES):
        if context.get_stage()!=stage:raise RuntimeError('Shared native stage changed before close')
        mark('app_update_'+str(index)+'_enter')
        app.update()
        mark('app_update_'+str(index)+'_return')
    if context.get_stage()!=stage:raise RuntimeError('Shared native stage changed during cleanup')
    return dict(viewport_present=window is not None,viewport_destroyed=window is not None,
        replicator_stop_wait_completed=True,cleanup_app_updates=CLEANUP_UPDATES)


def close_shared_runtime(app,*,progress_path=None,**dependencies):
    """Attempt normal close once; a cleanup error can never produce success."""
    from .scene_store import runtime_stage_leases
    error=None;receipt={};stream=None;journal_errors=[];events=[];leases=[]
    if progress_path is not None:
        try:stream=Path(progress_path).open('x')
        except Exception as caught:journal_errors.append(caught)
    def mark(event):
        if stream is None or journal_errors:return
        try:
            if len(events)>=32:raise RuntimeError('Shutdown journal event bound exceeded')
            stream.write(json.dumps(dict(schema='m2_native_shutdown_progress_v1',index=len(events),event=event))+'\n')
            stream.flush();os.fsync(stream.fileno());events.append(event)
        except Exception as caught:journal_errors.append(caught)
    try:
        mark('cleanup_enter');runtime_stage_leases()
        if progress_path is not None:dependencies=dict(dependencies,progress=mark)
        receipt=quiesce_before_close(app,**dependencies)
        leases=runtime_stage_leases();mark('leases_valid_before_close')
    except Exception as caught:error=caught
    try:
        mark('app_close_enter');app.close();mark('app_close_return')
    except Exception as caught:
        if error is None:error=caught
        else:error=RuntimeError(str(error)+'; app.close: '+repr(caught))
    finally:
        if stream is not None:stream.close()
    if journal_errors:
        error=RuntimeError(str(error)+'; shutdown progress journal: '+repr(journal_errors[0]))
    if error is not None:raise RuntimeError('Shared native cleanup failed: '+str(error)) from error
    return dict(receipt,app_close_returned=True,fast_shutdown=False,
        disable_viewport_updates=True,
        configured_limit_cpu_threads=SHARED_CPU_THREADS,
        stage_cache_leases=leases,stage_cache_release_policy='owned_process_exit',
        stage_cache_validation_phase='before_app_close',
        progress_journal=None if progress_path is None else str(progress_path),
        interpreter_exit_policy='success_only_os_exit_after_native_close_and_fsynced_receipt')
