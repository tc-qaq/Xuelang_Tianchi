import sys
import os
sys.path.insert(0, '..')
import zipfile
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, init,autograd
from mxnet.gluon import nn,data as gdata, loss as gloss, utils as gutils
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms
from mxnet import image
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
test_augs = transforms.Compose([transforms.Resize(1536),transforms.ToTensor(),normalize])

def transform(data,label,augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    #data = nd.transpose(data,(2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')

def batch_net(dir, suffix, net,train=True):
    res_list1 = []
    res_list2 = []    
    file_dir = 'data/xgb/dense/'
    if(train == True):
        fw1 = open(file_dir+'net_data.csv','w',encoding='utf-8')
        fw2 = open(file_dir+'net_label.csv','w',encoding='utf-8')
    else:
        fw1 = open(file_dir+'net_test.csv','w',encoding='utf-8')
        fw2 = open(file_dir+'net_test_name.csv','w',encoding='utf-8')
    for root,dirs,files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            filesuffix = os.path.splitext(filepath)[1][1:]
            if(filesuffix in suffix):
                with open(filepath,'rb') as f:
                    img = image.imdecode(f.read())
                data, _ = transform(img,-1,test_augs)
                data = data.expand_dims(axis=0)
                out = net(data.as_in_context(mx.gpu()))
                print(out.shape)
                out = out.reshape((16384))
                out = out.as_in_context(mx.cpu())
                out = out.asnumpy()       
                res = ''
                for item in out:
                    res += str(item) +' '   
                #print(res)
                fw1.write(res+'\n')
                label = root.strip().split('/')[-1]
                if(train == True):
                    fw2.write(label+'\n')
                    print(label)
                else:
                    fw2.write(file+'\n')
                    print(file)
    fw1.close()
    fw2.close()
    print('net done!!!')
    

pretrained_net = models.densenet121(pretrained=True)
net = nn.HybridSequential()
for layer in pretrained_net.features:
    net.add(layer)
print(net)
ctx = mx.gpu()
net.collect_params().reset_ctx(ctx)
net.hybridize()
dir = 'data/test_b/'
suffix=['jpg']
batch_net(dir,suffix,net,train=False)
