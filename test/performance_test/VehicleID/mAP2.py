import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio
import os
import sys
import cv2
import argparse
from models import VGGM
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
parser.add_argument('--save', dest='save',
                    help='save to file',
                    default='map_vggm.txt', type=str)
parser.add_argument('--save_dir', dest='save_dir',
                    default='checkpoints/', type=str)
parser.add_argument('--which_epoch', dest='which_epoch',
                    default='latest', type=str)
parser.add_argument('--im_height', dest='im_height',
                    help = 'im_height',
                    default=224, type=int)
parser.add_argument('--im_width', dest='im_width',
                    help = 'im_width',
                    default=224, type=int)
args = parser.parse_args()

def gen_gallery_probe_ori(samples, k=10000):
    """
    k: k samples for each id in gallery, 1 for reid
    TODO: generate gallery and probe sets.

    """
    cls_ids = samples.keys()
    gallery = {}
    extansion = {}
    probe = {}
    for cls_id in cls_ids:
        cls_samples = samples[cls_id]
        if len(cls_samples)<=1:
            continue
        gallery[cls_id] = []
        probe[cls_id] = []
        extansion[cls_id] = []
        n = len(cls_samples)
        #  gid = np.random.randint(0, n)
        gids = np.random.permutation(np.arange(n))[:min(n-1, k)]
        for i in xrange(len(cls_samples)):
            if i in gids:
                gallery[cls_id].append(cls_samples[i])
            else:
                probe[cls_id].append(cls_samples[i])
                extansion[cls_id].append(cls_samples[i])
    return gallery, extansion, probe

def gen_gallery_probe_random(samples, k=1):
    cls_ids = samples.keys()
    gallery = {}
    probe = {}
    extansion = {}
    for cls_id in cls_ids:
        cls_samples = samples[cls_id]
        n = len(samples[cls_id][0]) + len(samples[cls_id][1])
        if n <=1:
            continue
        gallery[cls_id] = []
        probe[cls_id] = []
        extansion[cls_id] = []
        pids = np.random.randint(0,n)
        if(pids < len(samples[cls_id][0])):
            pid_v = 0
        else:
            pid_v = 1
            pids = pids - len(samples[cls_id][0])

        exts = np.random.randint(0,n)
        if(exts < len(samples[cls_id][0])):
            ext_v = 0
        else:
            ext_v = 1
            exts = exts - len(samples[cls_id][0])
        
        if(len(samples[cls_id][ext_v]) > 0):
            exts = np.random.randint(0,len(samples[cls_id][ext_v]))
            extansion[cls_id].append(cls_samples[ext_v][exts])
        for i in xrange(len(cls_samples[0])):
            if i == pids and pid_v == 0:
                probe[cls_id].append(cls_samples[0][i])
            else:
                gallery[cls_id].append(cls_samples[0][i])

        for i in xrange(len(cls_samples[1])):
            if i == pids and  pid_v == 1:
                probe[cls_id].append(cls_samples[1][i])
            else:
                gallery[cls_id].append(cls_samples[1][i])
    return gallery, extansion, probe


def gen_gallery_probe(samples, k=1):
    cls_ids = samples.keys()
    gallery = {}
    probe = {}
    extansion = {}
    for cls_id in cls_ids:
        cls_samples = samples[cls_id]
        n = len(samples[cls_id][0]) + len(samples[cls_id][1])
        if n <=1:
            continue
        gallery[cls_id] = []
        probe[cls_id] = []
        extansion[cls_id] = []
        pids = np.random.randint(0,n)
        if(pids < len(samples[cls_id][0])):
            pid_v = 0
        else:
            pid_v = 1
            pids = pids - len(samples[cls_id][0])
        ext_v = (pid_v + 1)%2
        
        if(len(samples[cls_id][ext_v]) > 0):
            exts = np.random.randint(0,len(samples[cls_id][ext_v]))
            extansion[cls_id].append(cls_samples[ext_v][exts])
        else:
            extansion[cls_id].append(cls_samples[pid_v][pids])
        for i in xrange(len(cls_samples[0])):
            if i == pids and pid_v == 0:
                probe[cls_id].append(cls_samples[0][i])
            else:
                gallery[cls_id].append(cls_samples[0][i])

        for i in xrange(len(cls_samples[1])):
            if i == pids and  pid_v == 1:
                probe[cls_id].append(cls_samples[1][i])
            else:
                gallery[cls_id].append(cls_samples[1][i])
    return gallery, extansion, probe


def load_cls_samples(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = {}
            cls_samples[int(t[1])][0] = []
            cls_samples[int(t[1])][1] = []
        cls_samples[int(t[1])][int(t[2])%2].append(t[0])
    return cls_samples

def load_cls_samples_ori(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = []
        cls_samples[int(t[1])].append(t[0])
    return cls_samples

def load_network(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(args.save_dir.strip(), save_filename)
    network.load_state_dict(torch.load(save_path))

def load_network_ID(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(args.save_dir.strip(), save_filename)
    print save_path
    checkpoint = torch.load(save_path,map_location={'cuda:1':'cuda:0'})
    network.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    print("->loading checkpoint '{}' (epoch{})".format(save_filename,start_epoch))
    params=checkpoint['state_dict']


net_ID = VGGM.VGGM()
load_network(net_ID, 'ID', args.which_epoch.strip())
net_ID.cuda()
net_ID.eval()
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

f = open('result.txt','w')
for r_id in xrange(args.repeat):
    cls_samples = load_cls_samples(args.list_file)
    gallery, extansion, probe = gen_gallery_probe_random(cls_samples, args.maxg)
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
    g_imgs = []
    k = 0
    for gid in gallery.keys():
        for s in gallery[gid]:
            img = Image.open(os.path.join(args.image_dir.strip(), s + ext)).convert('RGB')
            im = transformer(img)
            im = torch.unsqueeze(im, 0)
            im = im.cuda()
            im = Variable(im)
            fea = net_ID(im)[args.feature_layer].data
            g_feat[k] = fea
            g_ids[k] = gid
            g_imgs.append(s)
            k += 1

    if r_id==0:
        print 'Gallery feature extraction finished'

    for pid in probe:# for every probe only has one sample
        #probe
        psample = probe[pid][0]
        p_dist = np.zeros([g_n,], dtype=np.float32)
        img = Image.open(os.path.join(args.image_dir.strip(), psample + ext)).convert('RGB')
        im = transformer(img)
        im = torch.unsqueeze(im, 0)
        im = im.cuda()
        im = Variable(im)
        fea = net_ID(im)[args.feature_layer].data
        p_feat = fea
    
        ##compute distance
        dist = np.zeros([g_n,], dtype=np.float32)
        for i in xrange(g_n):
            p_dist[i] = np.linalg.norm(g_feat[i]-p_feat)

        ##probe
        p_sorted = np.array([g_ids[i] for i in p_dist.argsort()])
        p_sortimg = np.array([g_imgs[i] for i in p_dist.argsort()])
        n = np.sum(p_sorted==pid)
        hit_inds_p = np.where(p_sorted==pid)[0]
        map_p = 0
        for i, ind in enumerate(hit_inds_p):
            map_p += (i+1)*1.0/(ind+1)
        map_p /= n
        mean_avg_prec_p[r_id] += map_p

    mean_avg_prec_p[r_id] /= p_n
    print '============================= ITERATION %d =============================' % (r_id+1)
    print mean_avg_prec_p[r_id]

print 'Average MAP:', np.mean(mean_avg_prec_p)
if args.save != '':
    with open(args.save, 'a') as fd:
        save_filename = '%s_net_%s.pth' % ('ID',args.which_epoch.strip())
        fd.write("->loading checkpoint '{}')".format(save_filename))
        fd.write('%.6f\n' % np.mean(mean_avg_prec_p))
