import os
from subprocess import call
import sys, time

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = '5'
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    print('Missing argument: job_id and gpu_id.')
    quit()

# Executables
executable = 'python3'

# Arguments
architecture = ['conv4_sn', 'conv4_sn_nobn', 'dcgan4_sn_nobn', 'dcgan4_sn_nobn', 'resnet_sn_v1', 'resnet_sn_v2']
gantype = ['standard', 'standard', 'standard', 'standard', 'standard', 'standard']
opt_type = ['adam', 'adam', 'adam', 'rmsprop', 'adam', 'adam']
lr = ['1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4']

bs = '64'
z_dim = '128'
beta1 = '0.5'
beta2 = '0.999'
seed = '125'

# Paths
rootdir = '..'
scriptname = 'run.py'
cwd = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(cwd, 'out', time.strftime("%Y%m%d"),
                      'simgd_{}_{}_{}_bs{}_zdim{}_lr{}_beta1#{}_beta2#{}_seed{}'.format(
                          architecture[job_id], gantype[job_id],
                          opt_type[job_id], bs, z_dim,
                          lr[job_id], beta1, beta2, seed))

args = [
    # Architecture
    '--image-size', '128',
    '--output-size', '32',
    '--beta1', beta1,
    '--beta2', beta2,
    '--c-dim', '3',
    '--z-dim', z_dim,
    '--gf-dim', '64',
    '--df-dim', '64',
    '--reg-param', '10.',
    '--g-architecture', architecture[job_id],
    '--d-architecture', architecture[job_id],
    '--gan-type', gantype[job_id],
    # Training
    '--seed', seed,
    '--optimizer', 'simgd',
    '--opt-type', opt_type[job_id],
    '--nsteps', '500000',
    '--ntest', '5000',
    '--learning-rate', lr[job_id],
    '--batch-size', bs,
    '--log-dir', os.path.join(outdir, 'tf_logs'),
    '--sample-dir', os.path.join(outdir, 'samples'),
    '--is-inception-scores',
    '--fid-type', '1',
    '--inception-dir', './inception',
    # Data set
    '--dataset', 'cifar-10',
    '--data-dir', './data',
    '--split', 'train'
]

# Run
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
