"""
import requests

import playsound
playsound.playsound('C:\\Users\\az567\\darknet1\\Fire_detection\\mp3.mp3')


def lineNotifyMessage(token, msg):
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type" : "application/x-www-form-urlencoded"
    }
    payload = {'message': msg }
    r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
    return r.status_code

if __name__ == "__main__":
  token = 'OcLOSG5be45VGtH5mN2Thd7a6fIrzIZFrzDVqQ7VVET'
  message = '偵測到火焰'
  lineNotifyMessage(token, message)

import pyfirmata
board=pyfirmata.Arduino(COM3)
board.digital[13].write(LOW)

"""

import os
import cv2
import glob

video_path = 'C:\\Users\\az567\\Downloads\\555.mp4'
output_folder = 'C:\\output\\'

if os.path.isdir(output_folder):
    print("Delete old result folder: {}".format(output_folder))
    os.system("rm -rf {}".format(output_folder))
os.system("mkdir {}".format(output_folder))
print("create folder: {}".format(output_folder))

vc = cv2.VideoCapture(video_path)
fps = vc.get(cv2.CAP_PROP_FPS)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
video = []

for idx in range(frame_count):
    vc.set(1, idx)
    ret, frame = vc.read()
    height, width, layers = frame.shape
    size = (width, height)

    if frame is not None:
        file_name = '{}{:08d}.jpg'.format(output_folder,idx)
        cv2.imwrite(file_name, frame)

    print("\rprocess: {}/{}".format(idx+1 , frame_count), end = '')
vc.release()


'''

url = 'https://notify-api.line.me/api/notify'
token = '123'
headers = {
    'Authorization': 'Bearer ' + token    # 設定權杖
}
data = {
    'message':'測試一下！'     # 設定要發送的訊息
}
data = requests.post(url, headers=headers, data=data)   # 使用 POST 方法

'''