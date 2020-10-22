# 使用 SSD 算法检测一二维码

## 环境依赖

* pytorch >= 1.3.0
* opencv-python

## 源码结构

```
dataset         // 读取数据
  |--barcode_dataset.py  // 将数据封装为 pytorch 可读取格式
  |--preprocess.py       // 数据预处理
  |--transforms.py       // 数据预处理
models          // 训练权重文件存储
config.py       // SSD 的 priors 设置，anchors 设置
eval.py         // 评估模型准确率
model.py        // 模型(network)定义
run_example.py  // 测试脚本
run_train.ipynb // 训练命令
train.py        // 训练
utils.py        // 工具模块
```

**Hint：** 详细注解见实际代码

## 标注工具

`labelImg`([链接](https://github.com/tzutalin/labelImg))

## 检测

检测使用 `SSD` 算法。

## 识别

识别使用 `pyzbar`(python 的 [zbar 库](https://pypi.org/project/pyzbar/))。

识别示例：

* https://www.jianshu.com/p/9c922274ed8d
* https://www.cnblogs.com/dongxiaodong/p/10216579.html

## 训练自己的数据

### 扩展数据集(不新增加类别)

使用标注工具标注后，将数据(jpg + xml)放置到 `barcode_dat/{train, val}` 即可。

代码部分不需要进行特别的调整。

### 新增类别

对于新增类别，加入新增类别为另一个二维码码制：`pdf417`。

反映到标注文件 xml 上的变化为：object 中的 name。同样将新增的图片和标注(jpg + xml)移动到相应的位置 `barcode_dat/{train, val}` 即可。

```
	<object>
		<name>pdf417</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>956</xmin>
			<ymin>280</ymin>
			<xmax>1443</xmax>
			<ymax>750</ymax>
		</bndbox>
	</object>
```

新增类别后代码层面的修改：

* `config.py`

```python
class_names = ['background', 'bar_code', 'qr_code']
# 添加新增类别
class_names = ['background', 'bar_code', 'qr_code', 'pdf417']
```

* `dataset/barcode_dataset.py`

```python
# class BarCodeDataset 中
self.class_names = ('background', 'bar_code', 'qr_code', 'pdf417')
```

注意上述两处调整，类别对应顺序要一致。

### 训练

```bash
python train.py -d ../barcode_dat \
                -v ../barcode_dat \
                --batch_size 32 \
                --num_epochs 200 \
                --debug_steps 30 \
                --scheduler "multi-step" \
                --lr 0.01 \
                --milestones "120,160"
```

上述参数中：

* batch_size：根据 GPU 的现存大小可以调整(16, 32, 64)，CPU 下调小一些(2, 4, 8)
* num_epochs：根据实际情况可适当进行调整
* milestones：如果 num_epochs 调整，该值也可以相应作出调整

另一种学习率调整策略，可作为尝试：

```bash
python train.py -d ../barcode_dat \
                -v ../barcode_dat \
                --batch_size 32 \
                --num_epochs 200 \
                --scheduler cosine \
                --lr 0.01 \
                --t_max 200
```

### 评估

评估模型准确率：

```bash
python eval.py -m models/tiny-Epoch-199-Loss-1.0858.pth \
               --dataset ../barcode_dat
```

参数说明：

* m：训练完成的模型权重，存储在 models 文件夹下
* dataset：读取验证集，提供 train 和 val 的父目录就可以，代码自动读取 val 中的数据进行验证

### 测试

实际使用部分，输入图片，输出检测框坐标。

```bash
python run_example.py tiny \
                      models/tiny-Epoch-199-Loss-1.0858.pth \
                      images/qr_6.jpg
```

## 参考

* [PyTorch Tutorial to Object Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
