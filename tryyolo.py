

import os
import re

# 设置目录路径和小数位数
dir_path = 'C:\\yolov7\\yolov7dataset\\valid\\labels'
# 遍历目录下所有txt文件
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r') as f:
            filedata = f.read()

        # 使用正则表达式匹配小数，并进行格式化
        formatted_data = re.sub("2","0",filedata)
        # 将格式化后的数据写回文件
        with open(filepath, 'w') as f:
            f.write(formatted_data)

