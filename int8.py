class YOLLOV5DataLoader(DataLoader):
    def _init_dataloader(self):
	    dataloader = create_dataloader(self._data_source['val'],
		    imgsz=self._imgsz, batch_size=self._batch_size,
		    stride=self._stride, single_cls=self,_single_cls,
		    pad=self._pad, rect=self._rect, workers=self._workers)[0]
	    return dataloader
	def _getitem_(self):
		try:
			batch_data = next(self._data_iter)
		except StopIteration:
			self._data_iter = iter(self._data_loader)
			batch_data = next(self._data_iter)
		im, target, path, shape = batch_data
		im = im.float()
		im /= 255
		nb, _, height, width = im.shape
		img = im.cpu().detach().numpy()

		annotation = dict()
		annotation['image_path'] = path
		annotation['target'] = target
		annotation['batch_size'] = nb
		annotation['shape']= shape
		annotation['width'] = width
		annotation['height'] = height
		annotation['img'] = img
		return (item,annotation), img
"""
class COCOMetric(Metric):
	def _process_status(self, stats):
		mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        stats = [np.concatenate(x, 0)for x in zip (*stats)]
	    if len(stats) and stats[0].any():
			tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=Flase, save_dir)
"""
    def get_cinfig():
		algorithms = [
			{
				"name":"DefaultQuantization",
				"params":{
					"target_device":"CPU",
					"preset":"mixed",
					"stat_subset_size": 300
				}
			}
		]
		config["algorithms"] = algorithms
		return config
config = get_cinfig()
init_logger(level='INFO')
Logger = get_logger(__name__)
save_di = increment_path(path("C:\yolov5"), exist_ok = True)
save_dir.mkir(parents=True, exist_ok =True)
model = load_model(config["model"])
data_loader = YOLLOV5DataLoader(config["dataset"])
engine = IEEngine(config=config["engine"],data_loader=data_loader, metric=metric)
pipeline = create_pipeline(config["algorithms"], engine)
metric_result = None
metric_result_fp32 = pipelline.evaluate(model)
Logger.info("FP32 mode metric_result:{}".format(metric_result_fp32))
compressed_model = pipeline.run(model)
compressed_model_weights(compressed_model)
optimized_save_dir = Path(save_dir).joinpath("optimized")
save_model(compressed_model, path(Path.cwd()).joinpath(optimized_save_dir), config["model"]["model_name"])
metric_results_i8 = pipeline.evaluate(compressed_model)

Logger.info("save quantized model in {}".format(optimized_save_dir))
Logger.info("Quantized INT8 model metric_results: {}".format(metric_results_i8))
