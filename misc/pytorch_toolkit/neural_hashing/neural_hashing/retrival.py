import numpy as np
import torch
import os
import operator
import random
from vectorHandle import precision
from utils.lenet import Encoder
from parameter import zSize
np.seterr(divide='ignore', invalid='ignore')

#print("Output vector size: ", zSize)
model = Encoder(zSize)
# print(model)
mdl_map = {}
model_path = "/storage/asim/trainingDmcCD/extra/encoder-100.pkl"
model.load_state_dict(torch.load(model_path))

model.cuda()

print(model_path)
galleryfolderpath = "/storage/asim/Hashing_MedMNISTV2/gallery"
queryfolderpath = "/storage/asim/Hashing_MedMNISTV2/query"
print(galleryfolderpath)
print(queryfolderpath)
#gimage  = random.sample(os.listdir(galleryfolderpath),20000)
gallery = {}
print("\n\n Building Gallery .... \n")
with torch.no_grad():
    for img in os.listdir(galleryfolderpath):
        np_img = np.load(os.path.join(galleryfolderpath, img))
        # print(pil_im)
        im = np.resize(np_img, (28, 28))
        # print(im.shape)
        numpy_image = np.array(im)
        # print(numpy_image.shape)
        if(len(numpy_image.shape) < 3):
            numpy_image = np.stack((numpy_image,)*1, axis=-1)
        # print(numpy_image.shape)
        numpy_image = numpy_image.transpose((2, 0, 1))
        # print(numpy_image.shape)
        #numpy_image = imagenormalize(numpy_image)
        numpy_image = np.array([numpy_image])
        # print(numpy_image.shape)
        torch_image = torch.from_numpy(numpy_image)
        torch_image = torch_image.type('torch.cuda.FloatTensor')

        h, _ = model(torch_image)
        # print(h)
        # h = torch.ceil(h)[0]
        gallery[img] = h
        del(torch_image)
    print("\n Building Complete. \n")
    ap1in1 = 0
    ap1in3 = 0
    ap1in5 = 0
    ap3in15 = 0
    ap5in15 = 0
    count = 0
    q_prec = 0
    q_prec_100 = 0
    q_prec_1000 = 0
    q_prec_10000 = 0
    # print(len(queryfolderpath))
    qNimage = random.sample(os.listdir(queryfolderpath), 500)
    # print(qNimage[0:500])
    querynumber = len((qNimage))
    print(querynumber)
    # print(len(qNimage[0:100]))
    for q_name in qNimage:
        count = count+1

        q_class = q_name.split(".")[0].split("_")[0]
        query_image = os.path.join(queryfolderpath, q_name)
        np_im_q = np.load(query_image)
        im_q = np.resize(np_im_q, (28, 28))
        numpy_image_q = np.array(im_q)
        if(len(numpy_image_q.shape) < 3):
            numpy_image_q = np.stack((numpy_image_q,)*1, axis=-1)
        numpy_image_q = (numpy_image_q.transpose((2, 0, 1)))
        numpy_image_q = np.array([numpy_image_q])
        torch_image_q = torch.from_numpy(numpy_image_q)
        torch_image_q = torch_image_q.type("torch.cuda.FloatTensor")
        h_q, _ = model(torch_image_q)

        dist = {}
        for key in gallery.keys():
            h1 = gallery[key]
            h1norm = torch.div(h1, torch.norm(h1, p=2))
            h2norm = torch.div(h_q, torch.norm(h_q, p=2))
            dist[key] = torch.pow(torch.norm(h1norm - h2norm, p=2), 2)*zSize/4
        print(count)
        print(q_class)
        sorted_pool = sorted(dist.items(), key=operator.itemgetter(1))[0:10]
        ret_classes = [sorted_pool[i][0].split(".")[0].split(
            "_")[0] for i in range(len(sorted_pool))]
        q_prec += precision(q_class, ret_classes)
        '''
        sorted_pool_100 = sorted(dist.items(), key=operator.itemgetter(1))[0:100]
        ret_classes_100 = [sorted_pool_100[i][0].split(".")[0].split("_")[0] for i in range(len(sorted_pool_100))]
        q_prec_100 += precision(q_class, ret_classes_100)
        sorted_pool_1000 = sorted(dist.items(), key=operator.itemgetter(1))[0:1000]
        ret_classes_1000 = [sorted_pool_1000[i][0].split(".")[0].split("_")[0] for i in range(len(sorted_pool_1000))]
        q_prec_1000 += precision(q_class, ret_classes_1000)
        sorted_pool_10000 = sorted(dist.items(), key=operator.itemgetter(1))[0:10000]
        ret_classes_10000 = [sorted_pool_10000[i][0].split(".")[0].split("_")[0] for i in range(len(sorted_pool_10000))]
        q_prec_10000 += precision(q_class, ret_classes_10000)'''

        if q_class == ret_classes[0]:
            ap1in1 += 1
        if q_class in ret_classes[0:3]:
            ap1in3 += 1
        if q_class in ret_classes[0:5]:
            ap1in5 += 1
        sorted_pool_15 = sorted(dist.items(), key=operator.itemgetter(1))[0:15]
        ret_classes_15 = [sorted_pool_15[i][0].split(".")[0].split(
            "_")[0] for i in range(len(sorted_pool_15))]
        print(ret_classes)
        print(ret_classes_15)
        if ret_classes_15.count(q_class) >= 5:
            ap5in15 += 1
        if ret_classes_15.count(q_class) >= 3:
            ap3in15 += 1
        print(count)
        querynumber = len((qNimage))
        if count % int(querynumber/10) == 0:
            print("Mean Average Precision 1 @ 1", ap1in1/count)
            print("Mean Average Precision 1 @ 3", ap1in3/count)
            print("Mean Average Precision 1 @ 5", ap1in5/count)
            print("Mean Average Precision 3 @ 15", ap3in15/count)
            print("Mean Average Precision 5 @ 15", ap5in15/count)
        '''if count % int(querynumber/100) == 0:    
            print("Model" + " ::  mAP@10 :", q_prec/count)
            print("Model" + " ::  mAP@100 :", q_prec_100/count)
            print("Model" + " ::  mAP@1000 :", q_prec_1000/count)
            print("Model" + " ::  mAP@10000 :", q_prec_10000/count)'''

    print("Mean Average Precision 1 @ 1", ap1in1/querynumber)
    print("Mean Average Precision 1 @ 3", ap1in3/querynumber)
    print("Mean Average Precision 1 @ 5", ap1in5/querynumber)
    print("Mean Average Precision 3 @ 15", ap3in15/querynumber)
    print("Mean Average Precision 5 @ 15", ap5in15/querynumber)
    #print("Model" + " ::  mAP@10 :", q_prec/querynumber)
    #print("Model" + " ::  mAP@100 :", q_prec_100/querynumber)
    #print("Model" + " ::  mAP@1000 :", q_prec_1000/querynumber)
    #print("Model" + " ::  mAP@10000 :", q_prec_10000/querynumber)
    #storepath = '/storage/asim/trainingDmcCD/train(0.0001)'
    #log_path = os.path.join(storepath, 'map_log.pkl')
    #mdl_map = q_prec/querynumber
   # with open(log_path, 'wb') as handle:
    #pickle.dump(mdl_map, handle)
