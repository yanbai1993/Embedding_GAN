import os
import argparse
import sys
import numpy as np
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT_DIR, 'caffe_root', 'caffe', 'python'))
import caffe


parser = argparse.ArgumentParser(description='CMC')
parser.add_argument('--gpu_id', dest='gpu_id',
                    help='GPU device to use [0]',
                    default=0, type=int)
parser.add_argument('--net_def', dest='net_def',
                    help='net_def',
                    default='', type=str)
parser.add_argument('--weights', dest='weights',
                    help='pretrained weights',
                    default='', type=str)
parser.add_argument('--list_file', dest='list_file',
                    help='test list file',
                    default='', type=str)
parser.add_argument('--mean', dest='mean',
                    help='mean file',
                    default='', type=str)
parser.add_argument('--fc', dest='fc',
                    help='fc layer',
                    default='fc7', type=str)
parser.add_argument('--ext', dest='ext',
                    help='image file extension',
                    default='', type=str)
parser.add_argument('--image_dir', dest='image_dir',
                    help='image directory',
                    default='', type=str)
parser.add_argument('--repeat', dest='repeat',
                    help='repeat times',
                    default=1, type=int)
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

def load_cls_samples(fname):
    cls_samples = {}
    for line in open(fname).readlines():
        t = line.strip().split()
        if int(t[1]) not in cls_samples:
            cls_samples[int(t[1])] = []
        cls_samples[int(t[1])].append(t[0])
    return cls_samples

ext = '' if args.ext=='' else '.'+args.ext
RANK_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
#RANK_LIST = [1, 5, 10, 15, 20, 30, 40, 50]

caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()
net = caffe.Net(args.net_def,
                args.weights,
                caffe.TEST)
print '\nLoaded network {:s}'.format(args.weights)
# configure pre-processing
#  in_ = 'pair_data'
in_ = net.inputs[0]
in_shape = net.blobs[in_].data.shape
transformer = caffe.io.Transformer({in_: in_shape})
transformer.set_transpose(in_, (2, 0, 1))
transformer.set_raw_scale(in_, 255)
transformer.set_channel_swap(in_, (2, 1, 0))
transformer.set_mean('data', np.array([102.7,115.8,123.5]))
#  if args.mean is not None:
    #  transformer.set_mean(in_, args.mean)

average_rank_rate = np.zeros([len(RANK_LIST), ])
for r_id in xrange(args.repeat):
    cls_samples = load_cls_samples(args.list_file)
    gallery, probe = gen_gallery_probe(cls_samples)
    if r_id==0:
        print 'Gallery size: %d' % (len(gallery.keys()))

    FEAT_SIZE = net.blobs[args.fc].data.shape[1]

    gids = gallery.keys()
    g_feat = np.zeros([len(gids), FEAT_SIZE], dtype=np.float32)
    for i in xrange(len(gids)):
        input_ = transformer.preprocess(in_, caffe.io.resize_image(caffe.io.load_image(os.path.join(args.image_dir, gallery[gids[i]][0]+ext)), (in_shape[2], in_shape[3])))
        out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
        #  g_feat[i] = out[args.fc].flatten()
        g_feat[i] = net.blobs[args.fc].data[0].flatten()
        #  print np.linalg.norm(g_feat[i])
    if r_id==0:
        print 'Gallery feature extraction finished'

    rank_rate = np.zeros([len(RANK_LIST), ])
    cnt = 0
    for pid in probe:
        for psample in probe[pid]:
            #print pid,psample 
            g_dist = np.zeros([len(gids),])
            p_feat = np.zeros([FEAT_SIZE,], dtype=np.float32)

            input_ = transformer.preprocess(in_, caffe.io.resize_image(caffe.io.load_image(os.path.join(args.image_dir, psample+ext)), (in_shape[2], in_shape[3])))
            out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
            #  p_feat = out[args.fc].flatten()
            p_feat = net.blobs[args.fc].data[0].flatten()

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
    print "probe size", cnt
    rank_rate /= cnt
    print '============================= ITERATION %d =============================' % (r_id+1)
    print RANK_LIST
    print rank_rate
    #  print '========================================================================'
    average_rank_rate += rank_rate
average_rank_rate /= args.repeat
print 'Average rank rate: '
print average_rank_rate
