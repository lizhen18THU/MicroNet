# import os
# import json
#
# import pandas as pd
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# def read_csv_classes(csv_dir: str, csv_name: str):
#     data = pd.read_csv(os.path.join(csv_dir, csv_name))
#     # print(data.head(1))  # filename, label
#
#     label_set = set(data["label"].drop_duplicates().values)
#
#     print("{} have {} images and {} classes.".format(csv_name,
#                                                      data.shape[0],
#                                                      len(label_set)))
#     return data, label_set
#
#
# def calculate_split_info(path: str, label_dict: dict, train_num: int = 500):
#     # read all images
#     image_dir = os.path.join(path, "images")
#     images_list = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
#     print("find {} images in dataset.".format(len(images_list)))
#
#     train_data, train_label = read_csv_classes(path, "train.csv")
#     val_data, val_label = read_csv_classes(path, "val.csv")
#     test_data, test_label = read_csv_classes(path, "test.csv")
#
#     # Union operation
#     labels = (train_label | val_label | test_label)
#     labels = list(labels)
#     labels.sort()
#     print("all classes: {}".format(len(labels)))
#
#     # create classes_name.json
#     classes_label = dict([(index, [label, label_dict[label]]) for index, label in enumerate(labels)])
#     json_str = json.dumps(classes_label, indent=4)
#     with open('./data/classes_name.json', 'w') as json_file:
#         json_file.write(json_str)
#
#     # concat csv data
#     data = pd.concat([train_data, val_data, test_data], axis=0)
#     print("total data shape: {}".format(data.shape))
#
#     # split data on every classes
#     num_every_classes = []
#     split_train_data = []
#     split_val_data = []
#     for index, label in enumerate(labels):
#         class_data = data[data["label"] == label]
#         class_data.loc[:, "label"] = index
#         num_every_classes.append(class_data.shape[0])
#
#         # shuffle
#         shuffle_data = class_data.sample(frac=1, random_state=1)
#         num_train_sample = train_num
#         split_train_data.append(shuffle_data[:num_train_sample])
#         split_val_data.append(shuffle_data[num_train_sample:])
#
#         # imshow
#         imshow_flag = False
#         if imshow_flag:
#             img_name, img_label = shuffle_data.iloc[0].values
#             img = Image.open(os.path.join(image_dir, img_name))
#             plt.imshow(img)
#             plt.title("class: " + classes_label[img_label][1])
#             plt.show()
#
#     # plot classes distribution
#     plot_flag = False
#     if plot_flag:
#         plt.bar(range(1, 101), num_every_classes, align='center')
#         plt.show()
#
#     # concatenate data
#     new_train_data = pd.concat(split_train_data, axis=0)
#     new_val_data = pd.concat(split_val_data, axis=0)
#
#     # save new csv data
#     new_train_data.to_csv(os.path.join(path, "split_train.csv"), index=False)
#     new_val_data.to_csv(os.path.join(path, "split_val.csv"), index=False)
#
#
# def main():
#     data_dir = "./data"  # 指向数据集的根目录
#     json_path = "./data/imagenet_class_label.json"  # 指向imagenet的索引标签文件
#
#     # load imagenet labels
#     label_dict = json.load(open(json_path, "r"))
#     label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])
#
#     calculate_split_info(data_dir, label_dict)
#
#
# if __name__ == '__main__':
#     main()
#
import os
import zipfile
import shutil

filepath = "/home2/jhj/lizhen_MicroNet_temper/data/0/images"
# zip_file = zipfile.ZipFile(filepath) #获取压缩文件
# print(filepath)
newfilepath = "/home2/jhj/lizhen_MicroNet_temper/data"
shutil.move(filepath, newfilepath)
# print(newfilepath)
# if os.path.isdir(newfilepath): # 根据获取的压缩文件的文件名建立相应的文件夹
#     pass
# else:
#     os.mkdir(newfilepath)
# for name in zip_file.namelist():# 解压文件
#     zip_file.extract(name,newfilepath)
# zip_file.close()
# Conf = os.path.join(newfilepath,'conf')
# if os.path.exists(Conf):#如存在配置文件，则删除（需要删则删，不要的话不删）
#     shutil.rmtree(Conf)
# print("解压{0}成功".format(filepath))
