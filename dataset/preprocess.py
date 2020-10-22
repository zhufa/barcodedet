from .transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0, cvt2gray=False):
        """
        训练时的数据增强操作
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),      # img, np.ndarray
            PhotometricDistort(),
            Expand(self.mean),      # img, boxes
            RandomSampleCrop(),     # img, boxes, labels
            RandomMirror(),
            ToPercentCoords(),      # boxes
            Resize(self.size),      # img
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])
        if cvt2gray:
            self.augment.replace(7, Cvt2Gray())  # 转灰度去训练,SubtractMeans是BGR减去均值处理，转灰度后不需要SubtractMeans

    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0, cvt2gray=False):
        """
        test 时的数据增强处理
        """

        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])
        if cvt2gray:
            self.transform.replace(2, Cvt2Gray())  # 转灰度去训练,SubtractMeans是BGR减去均值处理，转灰度后不需要SubtractMeans

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0, cvt2gray=False):
        """
        预测时的数据增强处理
        """
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])
        if cvt2gray:
            self.transform.replace(1, Cvt2Gray())  # 转灰度去训练,SubtractMeans是BGR减去均值处理，转灰度后不需要SubtractMeans

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image