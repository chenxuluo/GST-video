import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of action recognition models")
parser.add_argument('--dataset', type=str, choices=['somethingv1','somethingv2','diving48'],
                   default = 'somethingv1')
parser.add_argument('--root_path', type = str, default = '../',
                    help = 'root path to video dataset folders')
parser.add_argument('--store_name', type=str, default="")

# ========================= Model Configs ==========================
parser.add_argument('--type', type=str, default="GST",choices=['GST','R3D','S3D'],
                    help = 'type of temporal models, currently support GST,Res3D and S3D')
parser.add_argument('--arch', type=str, default="resnet50",choices=['resnet50','resnet101'],
                    help = 'backbone networks, currently only support resnet')
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--alpha', type=int, default=4, help = 'spatial temporal split for output channels')
parser.add_argument('--beta', type=int, default=2, choices=[1,2], help = 'channel splits for input channels, 1 for GST-Large and 2 for GST')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 60], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--dropout', '--dp', default=0.3, type=float,
                    metavar='dp', help='dropout ratio')

#========================= Optimizer Configs ==========================
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=3e-4, type=float,   
                    metavar='W', help='weight decay (default: 3e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: 20)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--checkpoint_dir',type=str,  required=True,
                    help = 'folder to restore checkpoint and training log')



