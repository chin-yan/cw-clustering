# encoding:utf-8
from annoy import AnnoyIndex
import numpy as np
np.random.seed(20200601)
import pickle
import sys
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
 
def load_ann(ann_path=None, index_to_name_dict_path=None, dim=64):
    ann = AnnoyIndex(dim)
    ann.load(ann_path)
 
    with open(index_to_name_dict_path, 'rb') as f:
        index_to_name_dict = pickle.load(f)
    return ann, index_to_name_dict
 
 
def query_ann(ann=None, index_to_name_dict=None, query_vec=None, topN=None):
    topN_item_idx_list = ann.get_nns_by_vector(query_vec, topN)
 
    topN_item_id_list = []
 
    for idx in topN_item_idx_list:
        item_id = index_to_name_dict[idx]
        topN_item_id_list.append(item_id)
 
    return topN_item_id_list
 
 
if __name__ == '__main__':
    index_to_name_dict_path = r"C:\Users\VIPLAB\Desktop\Yan\index_to_name_dict.pkl"
    ann_path = r"C:\Users\VIPLAB\Desktop\Yan\img_feature_list.ann"
    name_path = r"C:\Users\VIPLAB\Desktop\Yan\img_name_list.pkl"
    vec_path = r"C:\Users\VIPLAB\Desktop\Yan\img_feature_list.pkl"
    dim = 25088
    topN = 16
    
    name_path = open(name_path, 'rb')
    vec_path = open(vec_path, 'rb')
    img_name_list = pickle.load(name_path)
    img_vec_list = pickle.load(vec_path)
    
    idx = 4541
    query_name = img_name_list[idx]
    query_vec = img_vec_list[idx]
    
    ann, index_to_name_dict = load_ann(ann_path=ann_path, \
        index_to_name_dict_path=index_to_name_dict_path, \
        dim=dim)
 
    topN_item_list = query_ann(ann=ann, \
        index_to_name_dict=index_to_name_dict, \
        query_vec=query_vec, \
        topN=topN)
 
    # query object picture
    print("query_image: \n")
    fig, axes = plt.subplots(1, 1)
    #query_image = mpimg.imread(r"C:\Users\VIPLAB\Desktop\Yan\Lee'sFamilyReunion\AllFaces\\" + query_name)
    query_image = mpimg.imread(r"C:\Users\VIPLAB\Desktop\Yan\Lee'sFamilyReunion\FaceLibrary\L_347.jpg")
    axes.imshow(query_image/255)
    axes.axis('off')
    axes.set_title('%s' % query_name, fontsize=8, color='r')
    plt.show()
    plt.savefig("query_image.jpg")
 
    # Top-9 similar
    fig, axes = plt.subplots(4, 4)
    for idx, img_path in enumerate(topN_item_list):
 
        i = idx % 4   # Get subplot row
        j = idx // 4  # Get subplot column
        image = mpimg.imread(r"C:\Users\VIPLAB\Desktop\Yan\Lee'sFamilyReunion\AllFaces\\" + img_path)
        axes[i, j].imshow(image/255)
        axes[i, j].axis('off')
        axes[i, j].axis('off')
 
        axes[i, j].set_title('%s' % img_path, fontsize=8, color='b')
    plt.savefig("top-16 similar(L).jpg")
