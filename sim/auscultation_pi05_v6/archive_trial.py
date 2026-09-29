"""Use the verified archive orchestration on separately versioned V6 outputs."""
from sim.auscultation_pi05_v5 import archive_trial
archive_trial.SRC_ROOT='/home/robot/wam-sim-media/pi05-v6-results'
if __name__=='__main__':archive_trial.main()
