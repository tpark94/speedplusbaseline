import argparse

PROJROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplusbenchmarks',
               'linux': '/media/shared/Jeff/SLAB/speedplusbenchmarks',
               'vm': '/home/ubuntu/speedplusbenchmarks'}

DATAROOTDIR = {'mac':  '/Users/taehapark/SLAB/speedplus/data/datasets/speedplus',
               'linux': '/home/jeffpark/SLAB/Dataset/speedplus',
               'vm': '/home/ubuntu/Dataset/speedplus'}

parser = argparse.ArgumentParser('Configurations for SPEED+ Benchmarks')

# ------------------------------------------------------------------------------------------
# Basic directories and names
parser.add_argument('--projroot', type=str, default=PROJROOTDIR['linux'])
parser.add_argument('--dataroot', type=str, default=DATAROOTDIR['linux'])
parser.add_argument('--savedir',  type=str, default='checkpoints/park2019/dann_odn/speedplus_lightbox_1')

# ------------------------------------------------------------------------------------------
# Park et al. (2019) config.
parser.add_argument('--input_shape_odn',   type=tuple, default=(416, 416))
parser.add_argument('--input_shape_krn',   type=tuple, default=(224, 224))
parser.add_argument('--num_keypoints',     type=int,   default=11)
parser.add_argument('--batch_size',        type=int,   default=16)
parser.add_argument('--num_workers',       type=int,   default=8)
parser.add_argument('--max_epochs',        type=int,   default=20)
parser.add_argument('--save_epoch',        type=int,   default=1)
parser.add_argument('--test_epoch',        type=int,   default=1)
parser.add_argument('--lr',                type=float, default=0.001)
parser.add_argument('--lr_decay_alpha',    type=float, default=0.95)
parser.add_argument('--lr_decay_step',     type=int,   default=1)

# ------------------------------------------------------------------------------------------
# Dataset-related inputs
parser.add_argument('--source_train_csv', type=str, default='synthetic/splits_park2019/train.csv')
parser.add_argument('--target_train_csv', type=str, default='lightbox/splits_park2019/lightbox.csv')
parser.add_argument('--target_val_csv',   type=str, default='lightbox/splits_park2019/lightbox.csv')

# ------------------------------------------------------------------------------------------
# Other miscellaneous settings
parser.add_argument('--no_cuda',         dest='use_cuda',        action='store_false', default=True)
parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true',  default=False)

# End
cfg = parser.parse_args()