文件说明:

​	ckpt: 保存模型训练权重

​	data:标注数据

​	export: 部署模型export保存



​	export_onnx.py: 生成部署模型

​	loss.py: 训练用loss定义

​	test.py: 对任意一视频进行预测可视化

​	train.py: 主训练代码

​	utils.py: 工具函数以及config定义



用法:

​	训练:

​		`python train.py`

​	预测可视化:

​		`python test.py --video_path your_video_path`

​	生成部署模型:
​		`python export_onnx.py`



