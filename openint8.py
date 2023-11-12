
import cv2
import numpy as np
import openvino.quantization
import openvino.inference_engine as ie
from openvino.inference_engine import IENetwork, IECore, TensorDesc
from openvino.inference_engine import Blob as ie_blob

# 加载FP32或FP16精度的模型
model_xml = "C:\\Intel2\\openvino_2021.4.752\\deployment_tools\\model_optimizer\\best.xml"
model_bin = "C:\\Intel2\\openvino_2021.4.752\\deployment_tools\\model_optimizer\\best.bin"
net = IENetwork(model=model_xml, weights=model_bin)

# 加载量化器和量化参数
ie_core = ie.IECore()
config = {"CPU_THREADS_NUM": "4"}
device = "CPU"
ie_core.set_config(config, device)

input_shape = net.inputs["images"].shape
tensor_desc = TensorDesc(net.inputs["images"].precision, input_shape, "NHWC")
calibrator = ie_blob(tensor_desc, np.zeros((1, *input_shape[1:]), dtype=np.float32))
quantize_params = {"quantization_precision": "INT8", "calibration_mode": "calibrate", "calibrator": calibrator}

# 进行量化
quantized_net = ie_core.quantize(net, **quantize_params)

# 定义图像预处理函数
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # 对图像进行预处理
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))  # 转换图像格式
    image = image.astype(np.float32)  # 转换数据类型
    image -= [104, 117, 123]  # 减去均值
    return image

# 保存量化后的模型
output_xml_path = 'C:\\Users\\az567\\Desktop\\openvino\\716cpu\\output_model.xml'
output_bin_path = 'C:\\Users\\az567\\Desktop\\openvino\\716cpu\\output_model.bin'
ie_core.export_network(quantized_net, output_xml_path, output_bin_path)

# 指定模型路径和精度级别（FP32或FP16）
#model_path = 'C:\\Users\\az567\\Desktop\\openvino\\732cpu\\best.xml'
#bin_path = 'C:\\Users\\az567\\Desktop\\openvino\\732cpu\\best.bin'
#output_xml_path = 'C:\\Users\\az567\\Desktop\\openvino\\732cpu\\output_model.xml'
#output_bin_path = 'C:\\Users\\az567\\Desktop\\openvino\\732cpu\\output_model.bin'

"""

model_xml = "C:\\Users\\az567\\Desktop\\openvino\\732cpu\\best.xml"
model_bin = "C:\\Users\\az567\\Desktop\\openvino\\732cpu\\best.bin"
ie = IECore()
net = IENetwork(model=model_xml, weights=model_bin)

input_layer_names = [input_name for input_name in net.input_info]
print(f"Input layer names: {input_layer_names}")
"""


