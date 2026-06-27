import os, subprocess, sys
key=None
for line in open('.env'):
    line=line.strip()
    if line.startswith('WB_API_KEY='):
        key=line.split('=',1)[1].strip().strip('"').strip("'"); break
if not key: sys.exit('WB_API_KEY not found')
env=dict(os.environ); env['WANDB_API_KEY']=key; env['PYTHONPATH']=os.getcwd()
r=subprocess.run([sys.executable,'scratchpad/wandb_push_wavlm.py',
                  '--project','slashh-stress','--name','wavlm-npu-int8'], env=env)
sys.exit(r.returncode)
