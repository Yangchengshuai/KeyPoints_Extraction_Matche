from hloc import extract_features, match_features
from tqdm import tqdm
from pathlib import Path
import argparse
from hloc.utils.parsers import parse_retrieval
from hloc.utils.io import get_keypoints, get_matches

import cv2
import numpy as np

import time

if __name__ == '__main__':

    #添加参数，在运行时输入自己的--base_dir，比如我的运行代码是
    # python SPSGtest.py --base_dir /Users/chengshuai/Documents/work/test
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=Path, required=True)
    args = parser.parse_args()

    #图像所在路径
    images = args.base_dir / 'images/'
    #输出路径
    outputs=args.base_dir /'output/'
    #要匹配的图像对所在路径，里面每行的内容为：img1name img2name
    loc_pairs=args.base_dir / 'loc_pairs.txt'
    #计时
    start = time.time()
    #提取特征和匹配特征的配置文件
    feature_conf = extract_features.confs['superpoint_max']
    matcher_conf = match_features.confs['superglue']


    #提取特征和匹配特征，利用预训练模型
    features = extract_features.main(feature_conf, images, outputs)
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)
    retrieval_dict = parse_retrieval(loc_pairs)

    end = time.time()
    print('time:',end-start)

    #遍历每一对图像，画出匹配点对和匹配线
    for img1 in tqdm(retrieval_dict):
        img2 = retrieval_dict[img1]
        for img2name in img2:
            matches,_ = get_matches(loc_matches, img1, img2name)
            kpts0= get_keypoints(features, img1)
            kpts1= get_keypoints(features, img2name)
            #找出匹配点对的坐标
            kpts0 = kpts0[matches[:,0]]
            kpts1 = kpts1[matches[:,1]]
            #画出匹配点对
            img1=cv2.imread(str(images/img1))
            img2=cv2.imread(str(images/img2name))
            for i in range(len(kpts0)):
                cv2.circle(img1,(int(kpts0[i][0]),int(kpts0[i][1])),2,(0,0,255),-1)
                cv2.circle(img2,(int(kpts1[i][0]),int(kpts1[i][1])),2,(0,0,255),-1)
               
                img3=np.zeros((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3),np.uint8)
                img3[:img1.shape[0],:img1.shape[1]]=img1
                img3[:img2.shape[0],img1.shape[1]:]=img2
                #画出所有匹配线
                for i in range(len(kpts0)):
                    cv2.line(img3,(int(kpts0[i][0]),int(kpts0[i][1])),(int(kpts1[i][0])+img1.shape[1],int(kpts1[i][1])),(0,255,0),1)

            cv2.imwrite('/Users/chengshuai/Documents/work/test/Hierarchical-Localization/result.jpg',img3)

