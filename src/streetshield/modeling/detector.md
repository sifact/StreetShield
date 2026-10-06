# detector.py

Builds, saves and reloads the two-stage Faster R-CNN detector the chapter chooses.

- Wraps torchvision's Faster R-CNN, which has the chapter's three parts: convolutional backbone (feature map), Region Proposal Network (candidate regions) and a classifier/box-regression head.
- Defaults to the MobileNetV3-FPN backbone so it trains on a laptop GPU (Apple MPS); ResNet50-FPN(v2) is one config change away when accuracy matters more than speed.
- Loads COCO-pretrained weights and replaces only the final predictor with `num_classes = 2 + background`, which is what makes a ~1.4k-image dataset enough.
- Resizes every image to one fixed size (`model.fixed_size`, default 1024×640): Apple MPS caches a compiled graph per input shape, so variable sizes leaked ~1 GB of RAM every 15 s; boxes are mapped back to original coordinates automatically.
- Checkpoints store the model config and class list next to the weights, so `load_detector` rebuilds the exact model without the training config.
