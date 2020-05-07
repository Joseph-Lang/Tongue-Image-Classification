# encoding: utf-8
import pandas as pd
import os
import re
import zipfile
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# 若目录尚不存在，则创建指定目录
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 将压缩包解压缩于指定目录中
def unzip_directory(zip_path, dest_dir):
    # 创建解压缩指定目录
    create_directory(dest_dir)
    # 默认模式r，读取压缩包目录
    with zipfile.ZipFile(zip_path, 'r') as azip:
        # 返回所有目录和文件
        for zip_file in azip.namelist():
            # 将乱码先编码为'cp437'，然后再以'gbk'的形式解码，即可得到简体中文
            gbkfilename = zip_file.encode('cp437').decode('gbk')
            # 删除文件名称字符串中全部的空格
            gbkfilename = gbkfilename.replace(" ", "")
            # 删除前缀'Tongue_pic/'
            gbkfilename = gbkfilename.split('/')[-1]
            # 解压缩
            azip.extract(zip_file, dest_dir, pwd=None)
            # 定义分割符
            pattern = r'[.|/]'
            # 分割字符串zip_path，提取压缩包的名称
            result = re.split(pattern, zip_path)
            os.rename(os.path.join(dest_dir, zip_file), 
                     os.path.join(dest_dir, result[-2], gbkfilename))
            
# 使用标准化的命名方式重命名指定目录中的文件
def rename_directory(data_dir):
    for f in os.listdir(data_dir):
        # 筛选形如'D246.2.jpg'的文件
        if len(f.split('.')) == 3:
            # 元素f分割后首尾相加，存储于字符串newfilename，即新的文件名称
            newfilename = f.split('.')[0] + '.' + f.split('.')[-1]
            # 判断字符串newfilename是否已经存在于目录中
            if newfilename in os.listdir(data_dir):
                # 若已经存在，则证明该文件是重复的，直接删除
                os.remove(os.path.join(data_dir, newfilename))
            # 对文件进行重命名
            os.rename(os.path.join(data_dir, f), 
                        os.path.join(data_dir, newfilename))
        elif len(f.split('.')) == 2:
            # 筛选文件名称中包含中文的文件，形如'G18小于60岁.jpg'
            # 初始化循环变量i，记录字符串中中文字符的数目
            i = 0
            # 元素f赋值给newfilename
            newfilename = f
            for ch in newfilename:
                # 只要编码在此范围内即可判断为中文字符
                if u'\u4e00' <= ch <= u'\u9fff':
                    # 记录变量ch最后一次出现的位置
                    index = newfilename.index(ch)
                    # 删除元素f中的中文字符，生成新的文件名称newfilename
                    newfilename = newfilename.replace(ch, "")
                    i += 1
            # 修改形如'G18小于60岁.jpg'或'I45改为43.jpg'
            if (i == 2)|(i == 3):
                # 待删除的字符串rep，即index前两个字符
                rep = newfilename[index - 2] + newfilename[index - 1]
                newfilename = newfilename.replace(rep, "")
                os.rename(os.path.join(data_dir, f),
                         os.path.join(data_dir, newfilename))
                
# 以排序列表的形式返回指定目录下的文件名称
def get_data_files(data_dir):
    # 格式化输出，寻找全部以'.jpg'作为结尾的文件
    fs = glob("{}/*.jpg".format(data_dir))
    # 路径中的文件名称存储于列表中
    fs = [os.path.basename(filename) for filename in fs]
    # 排序
    return sorted(fs)

# 获取指定目录下的电子病历信息，以数据框(dataframe)的形式返回
def get_img_info(data_dir, columns):
    # 读取.csv文件
    EMR = pd.read_csv(data_dir, encoding='gbk')
    # 只选取dataframe中所需要的列，在本例中是'Unnamed: 0', '编号'和'高血压'
    info = EMR[list(columns.keys())].copy()
    # 重新命名info
    info.rename(columns=columns, inplace=True)
    # 删除dataframe中任何含有Nan的行
    info.dropna(axis=0, how='any', inplace=True)
    # 修改dataframe的'id'列元素的类型转换为无浮点的字符型
    info[columns['编号']] = info[columns['编号']].astype('int').astype('str')
    
    # 将dataframe的'Hyper'列元素‘是’和‘否’转换为'HYPER'和'NORMAL'
    # 定义函数convert_to_letter()
    def convert_to_letter(hyper):
        if hyper == '是':
            return 'HYPER'
        elif hyper == '否':
            return 'NORMAL'
        else:
            return False
    # 在dataframe中新添加一列，命名为'Status'
    info['Status'] = info[columns['高血压']].map(convert_to_letter)
    
    # 定义空列表id_list，用于存储dataframe中第1、2列合并的元素
    id_list = []
    # 遍历dataframe，合并第1、2列元素，并存储于列表patient_ID
    for i in range(info.shape[0]):
        patient_ID = info.iloc[i, 0] + info.iloc[i, 1]
        id_list.append(patient_ID)
    # 在dataframe中新添加一列，命名为'patient_ID'
    info['patient_ID'] = id_list
    
    # 定义空列表status_list，用于存储dataframe中第3、4列合并的元素
    status_list = []
    # 遍历dataframe，使用'_'合并第3、4列元素，并存储于列表status_list
    for i in range(info.shape[0]):
        patient_status = '_'.join([info.iloc[i, 4], info.iloc[i, 3]])
        status_list.append(patient_status)
    # 在dataframe中新添加一列，命名为'patient_status'
    info['patient_status'] = status_list
    
    return info

# 根据数据框(dataframe)中的内容重命名指定目录中的文件
def rename_directory_info(data_dir, info):
    for f in get_data_files(data_dir):
        # 获取文件'.jpg'前的名称，存储于变量ch
        ch = f.split('.')[0]
        # 当字符串中存在'HYPER'或'NORMAL'两个关键字时，表示修改已完成，跳出本次循环
        if ('HYPER' in ch)|('NORMAL' in ch):
            continue
        else:
            # 根据数据框info中'patient_status'列，对文件进行重命名
            newfilename = ''.join(info[info['patient_ID'] == ch]['patient_status'].values.tolist()) + '.jpg'
            os.rename(os.path.join(data_dir, f),os.path.join(data_dir, newfilename))

# 以test_size的比例，将data_dir目录下的数据集分割为训练集train和测试集test
# 以元组(tuple)的形式，返回训练集train和测试集test的目录路径
def split_img(data_dir, test_size, random_state):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    # 若train和test两个目录均存在
    if os.path.exists(train_dir) & os.path.exists(test_dir):
        return train_dir, test_dir
    # 若train和test两个目录均不存在，需要进行创建
    elif (not os.path.exists(train_dir)) & (not os.path.exists(test_dir)):
        # 创建train目录
        os.makedirs(train_dir)
        # 创建test目录
        os.makedirs(test_dir)
        # 存储目录data_dir下的全部文件名称
        dataset = []
        for f in get_data_files(data_dir):
            dataset.append(f)
        # 分割训练集train与测试集test
        train_test = train_test_split(dataset, test_size=test_size, 
                                      random_state=random_state)
        # 定义函数move
        def move(split, data_dir, split_dir):
            # 遍历数据集split
            for i in split:
                # 源目录
                src = os.path.join(data_dir, i)
                # 目标目录
                dst = os.path.join(split_dir, i)
                # 移动文件
                shutil.move(src, dst)
        # 将训练集train全部文件移动到train_dir目录下
        move(train_test[0], data_dir, train_dir)
        # 将测试集test全部文件移动到test_dir目录下
        move(train_test[1], data_dir, test_dir)
        return train_dir, test_dir
    else:
        print("Only one directory exist.")
        return False
            
# 根据class1和class2对目录下的全部文件分类
def classify_img(data_dir, class1, class2):
    class1_dir = os.path.join(data_dir, class1)
    class2_dir = os.path.join(data_dir, class2)
    # 若class1和class2两个目录均存在
    if os.path.exists(class1_dir) & os.path.exists(class2_dir):
        return class1_dir, class2_dir
    # 若class1和class2两个目录均不存在，需要进行创建
    elif (not os.path.exists(class1_dir)) & (not os.path.exists(class2_dir)):
        # 创建class1目录
        os.makedirs(class1_dir)
        # 创建class2目录
        os.makedirs(class2_dir)
        for f in get_data_files(data_dir):
            # 获取文件'.jpg'前的名称，存储于变量ch
            ch = f.split('.')[0]
            if class1 in ch:
                # 源目录
                src = os.path.join(data_dir, f)
                # 目标目录
                dst = os.path.join(class1_dir, f)
                # 移动文件
                shutil.move(src, dst)
            elif class2 in ch:
                src = os.path.join(data_dir, f)
                dst = os.path.join(class2_dir, f)
                shutil.move(src, dst)
            else:
                print("Wrong File:", f)
        return class1_dir, class2_dir
    else:
        print("Only one classification exist.")
        return False