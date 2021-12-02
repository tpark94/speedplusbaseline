import argparse

PROJROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplusbaseline',
               'linux': '/media/shared/Jeff/SLAB/speedplusbaseline'}

DATAROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplus/data/datasets',
               'linux': '/home/jeffpark/SLAB/Dataset'}

parser = argparse.ArgumentParser('Configurations for SPEED+ Baseline Study')

# ------------------------------------------------------------------------------------------
# Basic directories and names
parser.add_argument('--seed',     type=int, default=2021)
parser.add_argument('--projroot', type=str, default=PROJROOTDIR['linux'])
parser.add_argument('--dataroot', type=str, default=DATAROOTDIR['linux'])
parser.add_argument('--dataname', type=str, default='speedplus')
parser.add_argument('--savedir',  type=str, default='checkpoints/synthetic/krn')
parser.add_argument('--resultfn', type=str, default='')
parser.add_argument('--logdir',   type=str, default='log/synthetic/krn')
parser.add_argument('--pretrained', type=str, default='')

# ------------------------------------------------------------------------------------------
# Model config.
parser.add_argument('--model_name',      type=str,   default='krn')
parser.add_argument('--input_shape',     nargs='+',  type=int, default=(224, 224))
parser.add_argument('--num_keypoints',   type=int,   default=11)   # KRN-specific
parser.add_argument('--num_classes',     type=int,   default=5000) # SPN-specific
parser.add_argument('--num_neighbors',   type=int,   default=5)    # SPN-specific
parser.add_argument('--keypts_3d_model', type=str,   default='src/utils/tangoPoints.mat')
parser.add_argument('--attitude_class',  type=str,   default='src/utils/attitudeClasses.mat')

# ------------------------------------------------------------------------------------------
# Training config.
parser.add_argument('--start_over',        dest='auto_resume', action='store_false', default=True)
parser.add_argument('--randomize_texture', dest='randomize_texture', action='store_true', default=False)
parser.add_argument('--perform_dann',      dest='dann', action='store_true', default=False)
parser.add_argument('--texture_alpha',   type=float, default=0.5)
parser.add_argument('--texture_ratio',   type=float, default=0.5)
parser.add_argument('--use_fp16',          dest='fp16', action='store_true', default=False)
parser.add_argument('--batch_size',      type=int,   default=32)
parser.add_argument('--max_epochs',      type=int,   default=75)
parser.add_argument('--num_workers',     type=int,   default=8)
parser.add_argument('--test_epoch',      type=int,   default=-1)
parser.add_argument('--optimizer',       type=str,   default='rmsprop')
parser.add_argument('--lr',              type=float, default=0.001)
parser.add_argument('--momentum',        type=float, default=0.9)
parser.add_argument('--weight_decay',    type=float, default=5e-5)
parser.add_argument('--lr_decay_alpha',  type=float, default=0.96)
parser.add_argument('--lr_decay_step',   type=int,   default=1)

# ------------------------------------------------------------------------------------------
# Dataset-related inputs
parser.add_argument('--train_domain', type=str, default='synthetic')
parser.add_argument('--test_domain',  type=str, default='lightbox')
parser.add_argument('--train_csv',    type=str, default='train.csv')
parser.add_argument('--test_csv',     type=str, default='lightbox.csv')

# ------------------------------------------------------------------------------------------
# Other miscellaneous settings
parser.add_argument('--gpu_id',  type=int, default=0)
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', default=True)

# End
cfg = parser.parse_args()