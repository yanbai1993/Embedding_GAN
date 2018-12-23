import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio
import os
import sys
import cv2
import argparse
from models.VGGM import VGGM
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.autograd import Variable
from models.triplet_loss_model import Tripletnet
from data.car_dataset import load_triplet_samples, gen_gallery_probe

parser = argparse.ArgumentParser(description='CMC')
parser.add_argument('--feature_layer', dest='feature_layer',
                    default=0, type=int)
parser.add_argument('--list_file', dest='list_file',
                    help='test list file',
                    default='', type=str)
parser.add_argument('--query_list_file', dest='query_list_file',
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
                    default='', type=str)
parser.add_argument('--save_dir', dest='save_dir',
                    default='checkpoints/car_cyclegan_VGGM/', type=str)
parser.add_argument('--im_height', dest='im_height',
                    help = 'im_height',
                    default=224, type=int)
parser.add_argument('--im_width', dest='im_width',
                    help = 'im_width',
                    default=224, type=int)
args = parser.parse_args()


def load_query_reference(imagelist_file,query_file):
    gallery = {}
    probe = {}
    for line in open(imagelist_file).readlines():
        line = line.strip()
        t = line.split('/')
        if int(t[0]) not in gallery:
            gallery[int(t[0])] = []
        gallery[int(t[0])].append(line)
    for line in open(query_file).readlines():
        line = line.strip()
        t = line.split('/')
        if int(t[0]) not in probe:
            probe[int(t[0])] = []
        probe[int(t[0])].append(line)
    return gallery, probe

net = VGGM()
tnet = Tripletnet(net)
tnet.cuda()
print("=> loading checkpoint '{}'".format(args.save_dir.strip()))
checkpoint = torch.load(args.save_dir.strip())
start_epoch = checkpoint['epoch']
tnet.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
        .format(args.save_dir.strip(), checkpoint['epoch']))

tnet.eval()
transform_list = []
transform_list += [transforms.Scale([args.im_width, args.im_height], Image.BICUBIC)]
transform_list += [transforms.ToTensor()]
#transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
transform_list += [transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
transformer = transforms.Compose(transform_list)

ext = args.ext
length = 1024 
RANK_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]


average_rank_rate = np.zeros([len(RANK_LIST), ])
for r_id in xrange(args.repeat):
    cls_samples = load_triplet_samples(args.list_file)
    gallery, probe = gen_gallery_probe(cls_samples)
    #print 'gallery', gallery
    print '____________________________________'
    #print 'probe', probe
    #gallery, probe = load_query_reference(args.list_file, args.query_list_file)
    if r_id==0:
        print 'Gallery size: %d' % (len(gallery.keys()))

    FEAT_SIZE = length
    gids = gallery.keys()
    g_feat = np.zeros([len(gids), FEAT_SIZE], dtype=np.float32)
    for i in xrange(len(gids)):
        img = Image.open(os.path.join(args.image_dir.strip(), gallery[gids[i]][0] + ext))#.convert('RGB')
        print(gids[i], gallery[gids[i]][0])
        im = transformer(img)
        im = torch.unsqueeze(im, 0)
        im = im.cuda()
        im = Variable(im)
        out = net(im)[0]
        g_feat[i] = torch.squeeze(out, 0).data.cpu().numpy()

        #  print np.linalg.norm(g_feat[i])
    if r_id==0:
        print 'Gallery feature extraction finished'

    rank_rate = np.zeros([len(RANK_LIST), ])
    cnt = 0
    for pid in probe:
        for psample in probe[pid]:
            #  gids = gallery.keys()
            print pid, psample
            g_dist = np.zeros([len(gids),])
            p_feat = np.zeros([FEAT_SIZE,], dtype=np.float32)
            img  = Image.open(os.path.join(args.image_dir.strip(), psample + ext))
            im = transformer(img)
            im = torch.unsqueeze(im, 0)
            im = im.cuda()
            im = Variable(im)
            out = net(im)[0]
            p_feat = torch.squeeze(out, 0).data.cpu().numpy()
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
