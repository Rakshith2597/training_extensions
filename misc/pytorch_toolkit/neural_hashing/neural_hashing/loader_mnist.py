import os
import random
import itertools


def dataset(fpath):
    # fpath = trainingDataPath
    if not os.path.exists(fpath):
        raise Exception("Invalid path destination")
    if fpath.split("/")[-1] == "train1":
        im_thresh = 5000  # optimum 5000
        type2_num = 50000  # optimum 200000
        type1_num = 30000  # optimum 100000
        type0_num = 30000  # optimum 100000
    elif fpath.split("/")[-1] in ["test1", "valid"]:
        im_thresh = 2000  # optimum 1500
        type0_num = 10000  # optimum 10000
        type1_num = 10000  # optimum 10000
        type2_num = 10000  # optimum 10000

    classes = [d for d in os.listdir(fpath)]
    classes = sorted(classes)
    cls_to_idx = {classes[i]: i for i in range(len(classes))}
    print(classes)
    # print(cls_to_idx)
    cls_num_imgs = {cls_to_idx[i]: len(os.listdir(
        os.path.join(fpath, i))) for i in classes}
    print(cls_num_imgs)
    images = []
    images1 = []
    images2 = []
    dataset2 = []
    for cls_name in classes:
        images_temp = [(im, cls_name)
                       for im in os.listdir(os.path.join(fpath, cls_name))]
        random.shuffle(images_temp)
        # images.extend(images_temp)
        images1.extend(images_temp[:200])
        images2 = images_temp[:min(im_thresh, len(images_temp))]
        dataset2.extend(list(itertools.combinations(images2, 2)))

    # dataset2.extend(list(itertools.combinations(images,2)))
    dataset1 = list(itertools.combinations(images1, 2))
    # print(len(images_temp))
    '''print(len(images1))
    print(len(dataset1))
    print(len(dataset2))'''
    count = [0]*3
    newdataset = []
    for data in dataset1:
        if not data[0][1] == data[1][1]:
            t = 2
            count[2] += 1
            img_path1 = os.path.join(
                os.path.join(fpath, data[0][1]), data[0][0])
            img_path2 = os.path.join(
                os.path.join(fpath, data[1][1]), data[1][0])
            item = (img_path1, img_path2, t,
                    cls_to_idx[data[0][1]], cls_to_idx[data[1][1]])
            newdataset.append(item)
            if count[2] == type2_num:
                break

    random.shuffle(dataset2)
    # print(dataset2)
    for data in dataset2:
        img_path1 = os.path.join(os.path.join(fpath, data[0][1]), data[0][0])
        img_path2 = os.path.join(os.path.join(fpath, data[1][1]), data[1][0])
        '''print(data[0][0])
        print(data[0][0].split("_")[-2])
        print(data[1][0]) 
        print(data[1][0].split("_")[-2]) '''
        # if (((data[0][0].split("_")[0]+'_' + data[0][0].split("_")[1]) == (data[1][0].split("_")[0]+'_' + data[1][0].split("_")[1])) and (data[0][0].split("_")[-2]==data[1][0].split("_")[-2])):
        if (((data[0][0].split("_")[0]+'_' + data[0][0].split("_")[1]) == (data[1][0].split("_")[0]+'_' + data[1][0].split("_")[1])) and (data[0][0].split("_")[0] == data[1][0].split("_")[0])):
            t = 0
            if count[0] < type0_num:
                # if (t==0):
                count[0] += 1
                item = (img_path1, img_path2, t,
                        cls_to_idx[data[0][1]], cls_to_idx[data[1][1]])
                # item = (data[0][0],data[1][0],t)
                newdataset.append(item)
        else:
            t = 1
            if count[1] < type1_num:
                # if (t==1):
                count[1] += 1
                item = (img_path1, img_path2, t,
                        cls_to_idx[data[0][1]], cls_to_idx[data[1][1]])
                newdataset.append(item)

    random.shuffle(newdataset)
    print("Count", count)
    # print(newdataset[:5])
    return newdataset

#fpath = "/storage/asim/medHashing/test1"
#newdata = dataset(fpath)
# print(newdata)
