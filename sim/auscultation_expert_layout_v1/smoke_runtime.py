"""Short physics smoke. Invoke as python -m sim.auscultation_training.smoke_runtime."""
import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
import json,numpy as np
from .env import IsaacContactEnv
env=IsaacContactEnv()
for i in range(480):
    state=env.step(np.zeros(7),'settle')
    if i%120==0:print('SMOKE',i,state['q'],env.last_contacts,env.last_pairs,flush=True)
env.save('smoke')
print('SMOKE_DONE',state['tcp_world'],flush=True)
app.close()
