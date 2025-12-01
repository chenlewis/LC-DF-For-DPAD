import os
import random
import json
import shutil
rootPath = r'/home/data1/lyj/ROD/LQ_OCR_new//images/'

# 图像文件夹路径和类别标签
folder_paths = ["0", "1"]
folder_labels = [0, 1]

# 读取文件夹中的图像路径
image_data1 = {}
image_data0 = {}
for folder_path, folder_label in zip(folder_paths, folder_labels):
    for filename in os.listdir(os.path.join(rootPath,folder_path)):
        if filename.endswith(".jpg") and folder_label == 1:
            image_path = os.path.join(folder_path, filename)
            image_label = folder_label
            image_data1[image_path] = image_label
        if folder_label == 0:
            image_path = os.path.join(folder_path, filename)
            image_label = folder_label
            image_data0[image_path] = image_label

# 将图像路径和标签打包成元组列表并随机打乱
data1 = list(image_data1.items())
data0 = list(image_data0.items())
print(len(data1), len(data0))
random.shuffle(data1)
random.shuffle(data0)

# 将数据分成训练集、验证集和测试集
#change
val_size1 = int(0.1 * len(data1))
train_size1 = int(0.8 * len(data1))
test_size1 = len(data1) - train_size1 - val_size1

val_size0 = int(0.1 * len(data0))
train_size0 = int(0.8 * len(data0))
test_size0 = len(data0) - train_size0 - val_size0

print(train_size1, train_size0)

test_mode = 1
if test_mode == 1:
    train_data = data1[:1] + data0[:1]
    val_data = data1[:1] + data0[:1]
    test_data = data1 + data0
else:
    # train_data = data1[:train_size1] + data0[:train_size0]
    # val_data = data1[train_size1:train_size1+val_size1] + data0[train_size0:train_size0+val_size0]
    # test_data = data1[train_size1+val_size1:] + data0[train_size0+val_size0:]
    test_data = data1[:test_size1] + data0[:test_size0]
    val_data = data1[test_size1:test_size1+val_size1] + data0[test_size0:test_size0+val_size0]
    train_data = data1[test_size1+val_size1:] + data0[test_size0+val_size0:]

print(train_data)
random.shuffle(train_data)
print(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

train_dataDict = dict(train_data)
val_dataDict = dict(val_data)
d = dict(test_data)
test_dataDict = d

# 将字典保存为json文件
# with open("E:\Data\SRDID162\CMAP\DM_ROD/train.json", "w") as f:
#     train_dataDict = {key.replace("\\", "/"): value for key, value in train_dataDict.items()}
#     json.dump(train_dataDict, f)
# with open("E:\Data\SRDID162\CMAP\DM_ROD/val.json", "w") as f:
#     val_dataDict = {key.replace("\\", "/"): value for key, value in val_dataDict.items()}
#     json.dump(val_dataDict, f)
# with open("E:\Data\SRDID162\CMAP\DM_ROD/test.json", "w") as f:
#     test_dataDict = {key.replace("\\", "/"): value for key, value in test_dataDict.items()}
#     json.dump(test_dataDict, f)

# 定义路径
data_dir = "/home/data1/lyj/ROD/LQ_OCR_new/"
train_json_path = os.path.join(data_dir, "train.json")
val_json_path = os.path.join(data_dir, "val.json")
test_json_path = os.path.join(data_dir, "test.json")

# 要创建的目标文件夹路径
train_folder = os.path.join(data_dir, "train")
val_folder = os.path.join(data_dir, "val")
test_folder = os.path.join(data_dir, "test")

# 确保目录存在
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 假设train_dataDict, val_dataDict, test_dataDict已经定义并填充数据
# 修改路径并保存为JSON文件
for dataDict, json_path, folder in zip(
        [train_dataDict, val_dataDict, test_dataDict],
        [train_json_path, val_json_path, test_json_path],
        [train_folder, val_folder, test_folder]):

    # 替换路径分隔符
    dataDict = {key.replace("\\", "/"): value for key, value in dataDict.items()}

    # 保存为JSON
    with open(json_path, "w") as f:
        json.dump(dataDict, f)

#     # 移动/拷贝图片到相应的文件夹
#     for img_path in dataDict.keys():
#         # 获取图片文件名
#         img_filename = os.path.basename(img_path)
#         image_path = rootPath + '/' + img_path
#         # 目标路径
#         dest_path = os.path.join(folder, img_filename)
#
#         # 拷贝图片
#         shutil.copy2(img_path, dest_path)
#
# print("操作完成！")


# 计算训练集中标签为0和1的数量
train_count_0 = sum(1 for _, label in train_data if label == 0)
train_count_1 = sum(1 for _, label in train_data if label == 1)

# 计算验证集中标签为0和1的数量
val_count_0 = sum(1 for _, label in val_data if label == 0)
val_count_1 = sum(1 for _, label in val_data if label == 1)

# 计算测试集中标签为0和1的数量
test_count_0 = sum(1 for _, label in test_data if label == 0)
test_count_1 = sum(1 for _, label in test_data if label == 1)

print("Train - Label 0:", train_count_0, "Label 1:", train_count_1)
print("Validation - Label 0:", val_count_0, "Label 1:", val_count_1)
print("Test - Label 0:", test_count_0, "Label 1:", test_count_1)
print(len(data1))