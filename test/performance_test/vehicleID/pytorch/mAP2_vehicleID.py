import time
import numpy as np
from models.triplet_loss_model import Tripletnet
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio
import os
import sys
import cv2
import argparse
from models.VGGM_10 import VGGM_10
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Train a verification model')
parser.add_argument('--feature_layer', dest='feature_layer',
                    default=0, type=int)
parser.add_argument('--list_file', dest='list_file',
                    help='test list file',
                    default='', type=str)
parser.add_argument('--ext', dest='ext',
                    help='image file extension',
                    default='.jpg', type=str)
parser.add_argument('--image_dir', dest='image_dir',
                    help='image directory',
                    default='', type=str)
parser.add_argument('--repeat', dest='repeat',
                    help='repeat times',
                    default=1, type=int)
parser.add_argument('--maxg', dest='maxg',
                    help='max number of a class id in gallery',
                    default=1000, type=int)
parser.add_argument('--net', dest='net',
                    help='',
                    default='', type=str)
parser.add_argument('--save', dest='save',
                    help='save to file',
                    default='map_vggm_compress_2.txt', type=str)
parser.add_argument('--save_dir', dest='save_dir',
                    default='checkpoints/car_compression_test/', type=str)
parser.add_argument('--which_epoch', dest='which_epoch',
                    default='latest', type=str)
parser.add_argument('--im_height', dest='im_height',
                    help = 'im_height',
                    default=224, type=int)
parser.add_argument('--im_width', dest='im_width',
                    help = 'im_width',
                    default=224, type=int)
args = parser.parse_args()

def gen_gallery_probe_ori(samples, k=1):
    """
    k: k samples for each id in gallery, 1 for reid
    TODO: generate gallery and probe sets.

    """
    cls_ids = samples.keys()
    gallery = {}
    probe = {}
    for cls_id in cls_ids:
        cls_samples = samples[cls_id]
        if len(cls_samples)<=1:
            continue
        gallery[cls_id] = []
        probe[cls_id] = []
        n = len(cls_samples)
        #  gid = np.random.randint(0, n)
        gids = np.random.permutation(np.arange(n))[:min(n-1, k)]
        for i in xrange(len(cls_samples)):
            if i in gids:
                gallery[cls_id].append(cls_samples[i])
            else:
                probe[cls_id].append(cls_samples[i])
    return gallery, probe

def load_cls_samples_ori(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = []
        cls_samples[int(t[1])].append(t[0])
    return cls_samples


net = VGGM_10()
tnet = Tripletnet(net)
tnet.cuda()
print("=> loading checkpoint '{}'".format(args.net.strip()))
checkpoint = torch.load(args.net.strip())
start_epoch = checkpoint['epoch']
tnet.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
        .format(args.net.strip(), checkpoint['epoch']))

tnet.eval()
transform_list = []
transform_list += [transforms.Scale([args.im_width, args.im_height], Image.BICUBIC)]
transform_list += [transforms.ToTensor()]
#transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
transform_list += [transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
transformer = transforms.Compose(transform_list)
    
ext = args.ext
length = 1024 
mean_avg_prec = np.zeros([args.repeat, ])
mean_avg_prec_p = np.zeros([args.repeat, ])
mean_avg_prec_e = np.zeros([args.repeat, ])    
FEAT_SIZE = 1024
f = open('result.txt','w')
for r_id in xrange(args.repeat):
    cls_samples = load_cls_samples_ori(args.list_file)
    gallery, probe = gen_gallery_probe_ori(cls_samples, args.maxg)
    if r_id==0:
        print 'Gallery size: %d' % (len(gallery.keys()))
    FEAT_SIZE = 1024 
    g_n = 0
    p_n = 0
    for gid in gallery:
        g_n += len(gallery[gid])
    for pid in probe:
        p_n += len(probe[pid])
    g_feat = np.zeros([g_n, FEAT_SIZE], dtype=np.float32)
    g_ids = np.zeros([g_n, ], dtype=np.float32)
    k = 0

    for gid in gallery.keys():
        for s in gallery[gid]:
            img = Image.open(os.path.join(args.image_dir, s + args.ext))
            im = transformer(img)
            im = torch.unsqueeze(im, 0)
            im = im.cuda()
            im = Variable(im)
            out = net(im)[0]
            g_feat[k] = torch.squeeze(out, 0).data.cpu().numpy()
            g_ids[k] = gid
            k += 1
    if r_id == 0:
        print 'Gallery feature extraction finished'

    for pid in probe:
        for psample in probe[pid]:
            g_dist = np.zeros([g_n, ], dtype=np.float32)
            img  = Image.open(os.path.join(args.image_dir, psample + args.ext))
            im = transformer(img)
            im = torch.unsqueeze(im, 0)
            im = im.cuda()
            im = Variable(im)
            out = net(im)[0]
            p_feat = torch.squeeze(out, 0).data.cpu().numpy()
            for i in xrange(g_n):
                g_dist[i] = np.linalg.norm(g_feat[i] - p_feat)
            g_sorted = np.array([g_ids[i] for i in g_dist.argsort()])
            n = np.sum(g_sorted == pid)
            hit_inds = np.where(g_sorted == pid)[0]
            map_ = 0
            for i, ind in enumerate(hit_inds):
                map_ += (i + 1) * 1.0 / (ind + 1)
            map_ /= n
            mean_avg_prec[r_id] += map_
    mean_avg_prec[r_id] /= p_n
    print '============================= ITERATION %d =============================' % (r_id + 1)
    f.write('============================= ITERATION %{} ============================='.format(r_id + 1))
    print mean_avg_prec[r_id]
    f.write(str(mean_avg_prec[r_id]))
print 'Average MAP:', np.mean(mean_avg_prec)
f.write( 'Average MAP:{}'.format(np.mean(mean_avg_prec)))
if save != '':
    with open(save, 'a') as fd:
        fd.write('%.6f\n' % np.mean(mean_avg_prec))
