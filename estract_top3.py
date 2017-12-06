import json
import sys
sys.path.append('/home/frelam/caffe-master/python')
import caffe
import numpy as np
import os
from pprint import pprint
#with open('/home/frelam/ai_challenger_scene_train_20170904/train/scene_train_annotations_20170904.json','r') as f:
#    data = json.load(f)
caffe.set_mode_gpu()
caffe.set_device(1)
#pprint(data[1])
net1 = caffe.Net(
    '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/\
aichallenger_arug20171116/deploy_resnet152_places365.prototxt',
    1,
    weights = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/snapshot1123/1123_5_95.1/_iter_12000.caffemodel'
)

net2 = caffe.Net(
    '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/\
aichallenger_arug20171116/deploy_resnet152_places365.prototxt',
    1,
    weights = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/snapshot1123/1123_6/_iter_2400.caffemodel'
)

net3 = caffe.Net(
    '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/\
aichallenger_arug20171116/deploy_resnet152_places365.prototxt',
    1,
    weights = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/snapshot1123/1123_93.2-94.7-95/_iter_1200.caffemodel'
)

net = caffe.Net(
    '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/VGG16_ai/deploy_vgg16_places365.prototxt',
    1,
    weights = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/VGG16_ai/wt_vgg16_train_iter_12000.caffemodel'
)
def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean,'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_mean = arr[0]
    np.save(npyMean,npy_mean)

binMean = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/places365_deploy_weight/\
restnet152/places365CNN_mean.binaryproto'
npyMean = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/\
ai_challange_scene_dataset_lmdb/places365.npy'
convert_mean(binMean,npyMean)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.load(npyMean).mean(1).mean(1))
transformer.set_raw_scale("data",255)
transformer.set_channel_swap('data',(2,1,0))
dims = transformer.inputs['data'][1:]
path = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/ai_challenger_scene_test_b_20170922/scene_test_b_images_20170922'
files = os.listdir(path)
files_name = files;
batchsize = 16
out = None
out2 = None
out3 = None
out4 = None
i = 0
for chunk in [files[x:x+batchsize] for x in xrange(0,len(files),batchsize)]:
    new_shape = (len(chunk),) + tuple(dims)
    if net.blobs['data'].data.shape != new_shape:
        net.blobs['data'].reshape(*new_shape)
    for index,file in enumerate(chunk):
        image_files = path+'/'+file
        input_image = caffe.io.load_image(image_files)
        net.blobs['data'].data[index] = transformer.preprocess('data',input_image);
    output= net.forward()[net.outputs[-1]];
    if out is None:
        out = np.copy(output)
        print 'a'
    else:
        out = np.vstack((out,output))
        print 'b'
    i = i+1;
    print i
    #top_k3 = net.blobs['prob'].data[0].flatten().argsort()[-1:-4:-1];

#indices = (-out).argsort()[:,:3]
np.save("./out4.npy",out)