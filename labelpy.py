"""

import cv2
import os
from os import getcwd
from xml.etree import ElementTree as ET

# 此處設置相關的文件路徑
weightsPath = "C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\weights\\yolov4-fire_last.weights"
configPath = "C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\yolov4-fire.cfg"
labelsPath = "C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\fire.names"

# 讀取names文件中的類別名
LABELS = open(labelsPath).read().strip().split("\n")

# 使用opencv加載Darknet模型
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# 下麵是通過檢測獲取坐標的函數
def coordinate_get(img):
    coordinates_list = []  # 創建坐標列表
    boxes = []
    confidences = []
    classIDs = []
    (H, W) = img.shape[:2]
    # 得到 YOLO需要的輸出層
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # 從輸入圖像構造一個blob，然後通過加載的模型，給我們提供邊界框和相關概率
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # 在每層輸出上循環
    for output in layerOutputs:
        # 對每個檢測進行循環
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # 過濾掉那些置信度較小的檢測結果
            if confidence > 0.01:
                # 框後接框的寬度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 邊框的左上角
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新檢測出來的框
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            xmin = int(x)
            ymin = int(y)
            xmax = int(x + w)
            ymax = int(y + h)
            coordinates_list.append([xmin, ymin, xmax, ymax, classIDs[i]])

    return coordinates_list


# 定義一個創建一級分支object的函數
def create_object(root, xi, yi, xa, ya, obj_name):  # 參數依次，樹根，xmin，ymin，xmax，ymax
    # 創建一級分支object
    _object = ET.SubElement(root, 'object')
    # 創建二級分支
    name = ET.SubElement(_object, 'name')
    print(obj_name)
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    # 創建bndbox
    bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


# 創建xml文件的函數
def create_tree(image_name, h, w):
    global annotation
    # 創建樹根annotation
    annotation = ET.Element('annotation')
    # 創建一級分支folder
    folder = ET.SubElement(annotation, 'folder')
    # 添加folder標簽內容
    folder.text = (imgdir)

    # 創建一級分支filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    # 創建一級分支path
    path = ET.SubElement(annotation, 'path')

    path.text = getcwd() + '\{}\{}'.format(imgdir, image_name)  # 用於返回當前工作目錄

    # 創建一級分支source
    source = ET.SubElement(annotation, 'source')
    # 創建source下的二級分支database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # 創建一級分支size
    size = ET.SubElement(annotation, 'size')
    # 創建size下的二級分支圖像的寬、高及depth
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # 創建一級分支segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'


def main():
    for image_name in IMAGES_LIST:
        # 判斷後綴只處理jpg文件
        if image_name.endswith('jpg'):
            image = cv2.imread(os.path.join(imgdir, image_name))
            coordinates_list = coordinate_get(image)
            (h, w) = image.shape[:2]
            create_tree(image_name, h, w)

            for coordinate in coordinates_list:
                label_id = coordinate[4]
                create_object(annotation, coordinate[0], coordinate[1], coordinate[2], coordinate[3], LABELS[label_id])
                # if coordinates_list==[]:
                #     break

            # 將樹模型寫入xml文件
            tree = ET.ElementTree(annotation)
            tree.write('.\{}\{}.xml'.format(imgdir, image_name.strip('.jpg')))


if __name__ == '__main__':
    main()
"""


import cv2
import os
import numpy as np

weightsPath = "C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\weights\\yolov4-fire_best.weights"
configPath = "C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\yolov4-fire.cfg"
labelsPath = "C:\\Users\\az567\\darknet1\\Fire_detection\\cfg\\fire.names"
imgdir = "C:\\output\\17"
save_result_dir = "C:\\output\\result10"
save_txt_dir = "C:\\output\\txt10"

if not os.path.exists(save_result_dir):
    os.mkdir(save_result_dir)

if not os.path.exists(save_txt_dir):
    os.mkdir(save_txt_dir)

LABELS = open(labelsPath).read().strip().split("\n")
# Set according to the number of categories
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4

net = cv2.dnn.readNet(weightsPath, configPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
# Set according to the size of input
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


for image_name in os.listdir(imgdir):
    print("detect " + image_name + " ...")
    name = image_name.split('.jpg')[0]

    img = cv2.imread(os.path.join(imgdir, image_name))
    image_size = [img.shape[1], img.shape[0]]

    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s" % (LABELS[classid[0]])

        xmin = int(box[0])
        xmax = int(box[0]) + int(box[2])
        ymin = int(box[1])
        ymax = int(box[1]) + int(box[3])

        bndbox = [xmin, xmax, ymin, ymax]
        x, y, w, h = convert(image_size, bndbox)

        cv2.rectangle(img, box, color, 2)
        text = label + ":" + str(score[0])
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with open(os.path.join(save_txt_dir, name + '.txt'), 'a+') as f:
            # yolo txt
            f.write('%s %s %s %s %s\n' % (classid[0], x, y, w, h))

    cv2.imwrite(os.path.join(save_result_dir, name + ".jpg"), img)