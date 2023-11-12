import os
import re

# 设置目录路径和小数位数
dir_path = 'C:\\Users\\az567\\Desktop\\yolov7datasetw0123\\valid\\labels'
# 遍历目录下所有txt文件
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r') as f:
            filedata = f.read()

            num_lines = filedata
            for line in f:
                num_lines += 1
            print("文件中有", num_lines, "行。")


"""
        if __name__ == '__main__':
            strs = filedata
            item = "0"
            freq = strs.count(item)
            print(freq)
"""




"""
        contentLines=''
        characters=[]
        rate={}
        for line in filedata:
            line=line.strip()
            if len(line)==0:
                continue
        contentLines = contentLines + line
        for x in range(0,len(line)):
            if not line[x] in characters:
                characters.append(line[x])
            if line[x] not in rate:
                rate[line[x]]=1
            rate[line[x]]+=1
        rate=sorted(rate.items(), key=lambda e:e[1], reverse=True)
        print('全文共有%d個字'%len(contentLines))
        print('一共有%d個不同的字'%len(characters))
        print()
        for i in rate:
            print("[",i[0],"]共出現",i[1], "次")
        f.close()
"""
