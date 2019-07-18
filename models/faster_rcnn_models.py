import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

def get_mobilenet_model(num_classes):
    """
        Seguir ejemplo en https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    """

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7,sampling_ratio=2)

    # stats for test images
    #Original Width avg 172.58  std_dev 122.58 min 31 max 1083
    #Original Height avg 105.00 std_dev 52.75 min 13 max 516

    model = FasterRCNN(backbone, num_classes=num_classes, min_size=100, max_size=300, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

def get_fasterrcnn_model(num_classes):
	"""
	"""
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	return model