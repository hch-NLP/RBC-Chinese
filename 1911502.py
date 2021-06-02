#!/usr/bin/env python
# coding: utf-8

# # 简介
# 本项目是参加飞桨常规赛：中文场景文字识别的项目，项目score为85.94141。
# 
# 生成的预测文件为work中的result.txt文件
# 
# 项目任务为识别包含中文文字的街景图片，准确识别图片中的文字
# 
# 本项目源于https://aistudio.baidu.com/aistudio/projectdetail/615795，在此基础上进行修改
# 
# 感谢开发者为开源社区做出的贡献

# # 赛题说明
# **赛题背景**
# 
# 中文场景文字识别技术在人们的日常生活中受到广泛关注，具有丰富的应用场景，如：拍照翻译、图像检索、场景理解等。然而，中文场景中的文字面临着包括光照变化、低分辨率、字体以及排布多样性、中文字符种类多等复杂情况。如何解决上述问题成为一项极具挑战性的任务。
# 
# 本次飞桨常规赛以 中文场景文字识别 为主题，由2019第二届中国AI+创新创业全国大赛降低难度而来，提供大规模的中文场景文字识别数据，旨在为研究者提供学术交流平台，进一步推动中文场景文字识别算法与技术的突破。
# 
# **比赛任务**
# 
# 要求选手必须使用飞桨对图像区域中的文字行进行预测，返回文字行的内容。
# 
# **数据集介绍**
# 
# 本次竞赛数据集共包括33万张图片，其中21万张图片作为训练集，12万张作为测试集。数据集采自中国街景，并由街景图片中的文字行区域（例如店铺标牌、地标等等）截取出来而形成。所有图像都经过一些预处理，将文字区域利用仿射变化，等比映射为一张高为48像素的图片，如下图1所示：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/fb3cf59747e04f0cb9adde6a5a1945b3d9ef82f3b7c14c98bf248eb1c3886a3f)
# 
# 
# (a) 标注：魅派集成吊顶
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/57d58a35e1f34278bdb013b3f945ab69cddacf37c7fe40deba3c124fa1249753)
# 
# 
# (b) 标注：母婴用品连锁
# 图1
# 
# **标注文件**
# 
# 平台提供的标注文件为.txt文件格式。样例如下：
# 
# 
# 
# | h | w | name | value |
# | -------- | -------- | -------- |-------- |
# | 128 | 48 | img_1.jpg | 文本1|
# | 56	| 48	| img_2.jpg|	文本2|
# 其中，文件中的四列分别是图片的宽、高、文件名和文字标注。

# # 安装第三方库
# 
# 将安装目录设置为external-libraries，这样项目重启后安装的库不会消失。

# In[8]:


get_ipython().system('mkdir /home/aistudio/external-libraries')
import sys
sys.path.append('/home/aistudio/external-libraries')
get_ipython().system(' pip install tqdm paddlepaddle-gpu==1.7.1.post97 -i https://mirror.baidu.com/pypi/simple')
get_ipython().system(' pip install pqi')
get_ipython().system(' pqi use aliyun')
get_ipython().system(' pip install tqdm imgaug lmdb matplotlib opencv-python Pillow python-Levenshtein PyYAML trdg anyconfig # -t /home/aistudio/external-libraries')


# # 解压文件
# 
# 压缩包内含训练集图片、训练集图片信息、测试集图片

# In[9]:


import os
os.chdir('/home/aistudio/data/data10879')
get_ipython().system(' tar -zxf train_img.tar.gz')
get_ipython().system(' unzip test_images.zip')


# # 预处理
# 
# * 文件 langconv(language convert)，这个文件用来把繁体字转成简体字<br>
# 
# * 函数 read_ims_list：读取train.list文件，生成图片的信息字典
# * 函数 modify_ch：对标签label进行修改，进行四项操作，分别是“繁体->简体”、“大写->小写”、“删除空格”、“删除符号”。
# * 函数 pipeline：调用定义的函数，对训练数据进行初步处理。

# In[16]:


from work.langconv import Converter
import codecs
import random
import sys
import os
from os.path import join as pjoin

os.chdir('/home/aistudio')
sys.path.append('/home/aistudio/work')
def read_ims_list(path_ims_list):
    """
    读取 train.list 文件
    """
    ims_info_dic = {}
    with open(path_ims_list, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=3)
            w, h, file, label = parts[0], parts[1], parts[2], parts[3]
            ims_info_dic[file] = {'label': label, 'w': int(w)}
    return ims_info_dic
    

def modify_ch(label):
    # 繁体 -> 简体
    label = Converter("zh-hans").convert(label)

    # 大写 -> 小写
    label = label.lower()

    # 删除空格
    label = label.replace(' ', '')

    # 删除符号
    for ch in label:
        if (not '\u4e00' <= ch <= '\u9fff') and (not ch.isalnum()):
            label = label.replace(ch, '')

    return label

def save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))

def pipeline(dataset_dir):
    path_ims        = pjoin(dataset_dir, "train_images")
    path_ims_list   = pjoin(dataset_dir, "train.list")
    path_train_list = pjoin('/home/aistudio/work', "train.txt")
    path_test_list  = pjoin('/home/aistudio/work', "test.txt")
    path_label_list = pjoin('/home/aistudio/work', "dict.txt")

    # 读取数据信息
    file_info_dic = read_ims_list(path_ims_list)

    # 创建 train.txt
    class_set = set()
    data_list = []
    for file, info in file_info_dic.items():
        label = info['label']
        label = modify_ch(label)

        # 异常: 标签为空
        if label == '':
            continue

        for e in label:
            class_set.add(e)
        data_list.append("{0}\t{1}".format(pjoin('/home/aistudio/',path_ims, file), label))
        
    # 创建 label_list.txt
    class_list = list(class_set)
    class_list.sort()
    print("class num: {0}".format(len(class_list)))
    with codecs.open(path_label_list, "w", encoding='utf-8') as label_list:
        for id, c in enumerate(class_list):
            # label_list.write("{0}\t{1}\n".format(c, id))
            label_list.write("{0}\n".format(c))

    # 随机切分
    random.shuffle(data_list)
    val_len = int(len(data_list) * 0.05)
    val_list = data_list[-val_len:]
    train_list = data_list[:-val_len]
    print('训练集数量: {}, 验证集数量: {}'.format(len(train_list),len(val_list)))
    save_txt(train_list,path_train_list)
    save_txt(val_list,path_test_list)
    
random.seed(0)
pipeline(dataset_dir="data/data10879")


# In[17]:


os.chdir('/home/aistudio/work/PaddleOCR/')
get_ipython().system('pwd')


# # 特别说明
# 
# 对于PaddleOCR提供的配置文件rec_r34_vd_none_bilstm_ctc.yml我做出了如下修改
# 
# 1.将epoch_num改为120
# 
# 2.将train_batch_size_per_card改为256
# 
# 3.将test_batch_size_per_card改为128
# 
# 4.将base_lr改为0.00001
# 
# 经测试这样能提高score
# 
# **由于将epoch_num改为120后使用单卡GPU训练很费时间，同时训练具有随机性不能保证训练后与我得到同样的结果，所以我将训练120epoch后生成的模型文件打包在data/data50975/文件夹中只需要将
# rec_CRNN_aug_341.zip文件解压（数据集地址：https://aistudio.baidu.com/aistudio/datasetdetail/50975），将解压后的文件替换work/PaddleOCR/output/rec_CRNN_aug_341文件夹中的文件即可跳过下面的训练步骤，可以直接进行预测。替换程序如下**

# In[18]:


#** 注意**
# 运行此段代码后无需运行下面的训练代码可直接进行预测（需要取消该段程序注释）
# 若要自己训练，不要运行此段代码
'''
%cd ~
import os
import shutil
!cd data/data50975 && unzip rec_CRNN_aug_341.zip
%cd ~/work/PaddleOCR/output/rec_CRNN_aug_341
!rm -r *.pdmodel
!rm -r *.pdopt
!rm -r *.pdparams
%cd ~
filelist = os.listdir('data/data50975/rec_CRNN_aug_341')
print(filelist)
for file in filelist:
    src = os.path.join('data/data50975/rec_CRNN_aug_341', file)
    dst = os.path.join('work/PaddleOCR/output/rec_CRNN_aug_341', file)
    shutil.move(src, dst)
os.chdir('/home/aistudio/work/PaddleOCR/')
!pwd
'''


# # 模型训练

# In[ ]:


get_ipython().system(' export PYTHONPATH=$PYTHONPATH:.')
get_ipython().system(' python tools/train.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml')


# # 模型预测

# In[13]:


get_ipython().system(' python3 tools/infer_rec.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml -o Global.checkpoints=output/rec_CRNN_aug_341/best_accuracy')


# In[ ]:




