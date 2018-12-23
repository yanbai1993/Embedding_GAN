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


parser = argparse.ArgumentParser(description='CMC')
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
                    default=10, type=int)
parser.add_argument('--maxg', dest='maxg',
                    help='max number of a class id in gallery',
                    default=1000, type=int)
parser.add_argument('--save', dest='save',
                    help='save to file',
                    default='', type=str)
parser.add_argument('--save_dir', dest='save_dir',
                    default='checkpoints/car_cyclegan_VGGM/', type=str)
parser.add_argument('--which_epoch', dest='which_epoch',
                    default='latest', type=str)
parser.add_argument('--im_height', dest='im_height',
                    help = 'im_height',
                    default=224, type=int)
parser.add_argument('--im_width', dest='im_width',
                    help = 'im_width',
                    default=224, type=int)
args = parser.parse_args()

def gen_gallery_probe(samples, k=1):
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

def load_cls_samples_cls(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = {}
            cls_samples[int(t[1])][0] = []
            cls_samples[int(t[1])][1] = []
        cls_samples[int(t[1])][int(t[2])%2].append(t[0])
    return cls_samples

def load_cls_samples(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = []
        cls_samples[int(t[1])].append(t[0])
    return cls_samples

def load_network(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(args.save_dir, save_filename)
    network.load_state_dict(torch.load(save_path))


net_ID = VGGM.VGGM()
load_network(net_ID, 'ID', args.which_epoch)
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
RANK_LIST = [1,3, 5,7,9,10,12,14, 16,18, 20,22,24,26,28, 30,32,34,36,38, 40,42,44,46,48, 50]


average_rank_rate = np.zeros([len(RANK_LIST), ])
for r_id in xrange(args.repeat):
    cls_samples = load_cls_samples(args.list_file)
    gallery, probe = gen_gallery_probe(cls_samples)
    if r_id==0:
        print 'Gallery size: %d' % (len(gallery.keys()))

    FEAT_SIZE = length
    gids = gallery.keys()
    g_feat = np.zeros([len(gids), FEAT_SIZE], dtype=np.float32)
    for i in xrange(len(gids)):
        #input_ = transformer.preprocess(in_, caffe.io.resize_image(caffe.io.load_image(os.path.join(args.image_dir, gallery[gids[i]][0]+ext)), (in_shape[2], in_shape[3])))
        #out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
        #  g_feat[i] = out[args.fc].flatten()
        img = Image.open(os.path.join(args.image_dir.strip(), gallery[gids[i]][0] + ext)).convert('RGB')
        im = transformer(img)
        im = torch.unsqueeze(im, 0)
        im = im.cuda()
        im = Variable(im)
        # print(type(im))
        out = net_ID(im)[args.feature_layer].data
        g_feat[i] = out
        #  print np.linalg.norm(g_feat[i])
    if r_id==0:
        print 'Gallery feature extraction finished'

    rank_rate = np.zeros([len(RANK_LIST), ])
    cnt = 0
    for pid in probe:
        for psample in probe[pid]:
            #  gids = gallery.keys()
            g_dist = np.zeros([len(gids),])
            p_feat = np.zeros([FEAT_SIZE,], dtype=np.float32)
            img  = Image.open(os.path.join(args.image_dir.strip(), psample + ext))
            im = transformer(img)
            im = torch.unsqueeze(im, 0)
            im = im.cuda()
            im = Variable(im)
            out = net_ID(im)[args.feature_layer].data
            p_feat = out
            #input_ = transformer.preprocess(in_, caffe.io.resize_image(caffe.io.load_image(os.path.join(args.image_dir, psample+ext)), (in_shape[2], in_shape[3])))
            #out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
            #  p_feat = out[args.fc].flatten()
            #p_feat = net.blobs[args.fc].data[0].flatten()

            for i in xrange(len(gids)):
                #  input_ = transformer.preprocess(in_, caffe.io.resize_image(caffe.io.load_image(os.path.join(args.image_dir, gallery[gids[i]][0]+ext)), (in_shape[2], in_shape[3])))
                #  out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
                #  g_feat[i] = out[args.fc].flatten()
                g_dist[i] = np.linalg.norm(g_feat[i]-p_feat)
            g_sorted = [gids[i] for i in g_dist.argsort()]
            for k, r in enumerate(RANK_LIST):
                if pid in g_sorted[:r]:
                    rank_rate[k] += 1

            cnt += 1
            #  print '%s finished(%d)' % (psample, cnt), rank_rate/cnt

    rank_rate /= cnt
    print '============================= ITERATION %d =============================' % (r_id+1)
    print RANK_LIST
    print rank_rate
    #  print '========================================================================'
    average_rank_rate += rank_rate
average_rank_rate /= args.repeat
print 'Average rank rate: '
print average_rank_rate
if args.save != '':
    with open(args.save, 'a') as fd:
        fd.write('%.6f\n' % average_rank_rate)
