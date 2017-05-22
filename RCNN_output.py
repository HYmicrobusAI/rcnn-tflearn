from __future__ import division, print_function, absolute_import
import pickle
import numpy as np
import selectivesearch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path
import skimage
from sklearn import svm
import preprocessing_RCNN as prep
import os
import nms
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# Load testing images
def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def image_proposal(img_path):
    img = skimage.io.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        if r['size'] < 220:
            continue
        # resize to 224 * 224 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        im = Image.fromarray(proposal_img)
        resized_proposal_img = resize_image(im, 224, 224)
        candidates.add(r['rect'])
        img_float = pil_to_nparray(resized_proposal_img)
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


# Load training images
def generate_single_svm_train(one_class_train_file):
    trainfile = one_class_train_file
    savepath = one_class_train_file.replace('txt', 'pkl')
    images = []
    Y = []
    if os.path.isfile(savepath):
        print("restoring svm dataset " + savepath)
        images, Y = prep.load_from_pkl(savepath)
    else:
        print("loading svm dataset " + savepath)
        images, Y = prep.load_train_proposals(trainfile, 2, threshold=0.3, svm=True, save=True, save_path=savepath)
    return images, Y


# Use a already trained alexnet with the last layer redesigned
def create_alexnet(num_classes, restore=False):
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


# Construct cascade svms

def train_svms(train_file_folder, model):
    listings = os.listdir(train_file_folder)
    svms = []
    for train_file in listings:
        if "pkl" in train_file:
            continue
        X, Y = generate_single_svm_train(train_file_folder + train_file)
        train_features = []
        for i in X:
            feats = model.predict([i])
            train_features.append(feats[0])
        print("feature dimension")
        print(np.shape(train_features))
        clf = svm.LinearSVC()
        print("fit svm")
        clf.fit(train_features, Y)
        svms.append(clf)
    return svms


if __name__ == '__main__':
    train_file_folder = 'svm_train/'
    img_path = 'testimg7.jpg'
    imgs, verts = image_proposal(img_path)
    net = create_alexnet(3)
    model = tflearn.DNN(net)
    model.load('fine_tune_model_save.model')
    if os.path.isfile('./svms.pkl'):
        svms=pickle.load(open('./svms.pkl', 'rb'))
    else:
        svms = train_svms(train_file_folder, model)
        print("saving svms")
        pickle.dump((svms), open('./svms.pkl', 'wb'))
    # pickle.load(open(dataset_file, 'rb'))
    print("Done fitting svms")
    features = model.predict(imgs)
    print("predict image:")
    print(np.shape(features))
    results = []
    results_label = []
    results_margin=[]
    count = 0
    for f in features:
        temp_margin=svms[0].decision_function(f)
        temp_pred=svms[0].predict(f)
        for i in svms:
            pred = i.predict(f)
            if pred[0] != 0:
                margin=i.decision_function(f)
                if margin>temp_margin:
                    temp_margin=margin
                    temp_pred=pred[0]
        if temp_pred!=0:       # 如果结果为一个特定分类而不是背景，筛选出来
            results.append(verts[count])  # 加入边框
            results_margin.append(temp_margin) # 加入最大的预测值
            results_label.append(temp_pred) # 加入对应分类
        count += 1
    print("result:")
    print(results)
    print("result label:")
    print(results_label)
    print("margins:")
    print(results_margin)

    expired = []
    indexs = []
    def is_expired(x):
        for item in expired:
            if x == item:
                return True
        return False
    for cls in results_label:
        if not is_expired(cls):
            index = []
            for i in range(len(results_label)):
                if results_label[i] == cls:
                    index.append(i)
            indexs.append(index)
            expired.append(cls)

    print(indexs)
    for index in indexs:

        sub_results = []
        sub_margins = []
        sub_labels = []
        input = {'boxes': [],
                 'labels': [],
                 'margins': []}
        for sub_index in index:
            input['boxes'].append(results[sub_index])
            input['labels'].append(results_label[sub_index])
            input['margins'].append(results_margin[sub_index])
        input_tuples = []
        for box, margin in zip(input['boxes'], input['margins']):
            temp_margin = []
            temp_margin.append(margin)
            input_tuples.append(box + tuple(temp_margin))
        pick = nms.nms(input_tuples, overlap=0.3)
        this_boxes=[]
        for ind in pick:
            temp_box=input['boxes'][ind]
            tag=True
            if temp_box[0]<1 and temp_box[1]<1 and temp_box[2]>490 and temp_box[3]>490:
                tag=False
            if tag:
                this_boxes.append(temp_box)
        img = skimage.io.imread(img_path)
        plt.figure('flg'+str(input['labels'][0]))
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)
        for x, y, w, h in this_boxes:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
        print("class : "+str(input['labels'][0])+"\n\n")
        plt.show()
