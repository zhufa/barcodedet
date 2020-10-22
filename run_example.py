import random

import cv2
import sys
import torch

from utils import Timer
from model import create_tiny_ssd, create_predictor
import config as C

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_in_jupyter = False  # for run in IDE, or Python Console like jupyter
if run_in_jupyter:
    if len(sys.argv) < 4:
        print('Usage: python run_ssd_example.py <net type> <model path> <image path>')
        sys.exit(0)
    net_type = sys.argv[1]
    model_path = sys.argv[2]
    image_path = sys.argv[3]
else:
    net_type = "tiny"
    model_path = "models/tiny-Epoch-199-Loss-1.0257.pth"
    image_path = "val_img/0211.jpg"

class_names = C.class_names

if net_type == 'tiny':
    # 若是拿灰度图训练好的模型去预测，此处要指明输入通道为1
    net = create_tiny_ssd(len(class_names), is_test=True, input_imgChannel=3)
else:
    print("The net type is wrong")
    sys.exit(1)
net.load(model_path)
# 若是拿灰度图训练好的模型去预测，此处要指明cvt2gray表示对输入的待预测图像做灰度化处理
predictor = create_predictor(net, cvt2gray=False, candidate_size=200, device=DEVICE)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)  # 每个类提取前10个且置信度大于0.4的预测结果
random_color = lambda: random.randint(0, 255)
for i in range(boxes.size(0)):
    color = (random_color(), random_color(), random_color())
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 2)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                color,
                1)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
result = ''
for j in range(len(class_names)):
    condi = labels == j
    this_class_num = condi.long().sum(dim=0, keepdim=True)
    if this_class_num != 0:
        result += C.class_names[j] + f' found {this_class_num.numpy()[0]}, '
print(f"Found {len(probs)} objects: " + result + f"The output image is {path}")
'''
测试结果及问题优化
1.条码容易出现多个重复的预测框，自认为这与条码本身就是具有重复的属性有关，条码竖向分成两半时，这两部分各自依然具有完整的条码特征，所以一个条码容易
出现多个预测框，而二维码裁剪成两半后，两部分都丢失了特征，因此二维码不会出现重复的预测框，此问题可以通过调整测试时的IOU阈值来去处多余框
2.条码预测框位置准确，但其长宽(通常是长)未达到最佳效果，没有完整包裹住目标，优化方向？
3.将二维码识别成条码，优化方向？
4.小目标检测问题优化
5.转灰度去训练，效果并没有提升，自认为是因为灰度其实就是3通道值计算得到的
'''
