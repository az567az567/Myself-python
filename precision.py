import openvino.inference_engine as ie

model_xml = "C:\\Users\\az567\\Desktop\\openvino\\732cpu\\best.xml"
model_bin = "C:\\Users\\az567\\Desktop\\openvino\\732cpu\\best.bin"


ie_core = ie.IECore()
net = ie_core.read_network(model=model_xml, weights=model_bin)

input_precision = net.inputs['images'].precision
output_precision = net.outputs['/model.105/m.0/Conv'].precision

print(f"Input precision: {input_precision}")
print(f"Output precision: {output_precision}")



#net = ie.IENetwork(model=model_xml, weights=model_bin)
#output_layer_name = next(iter(net.outputs))
#print("Output layer name:", output_layer_name)