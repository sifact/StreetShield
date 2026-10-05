# CH03: Google Street View Blurring System

## What problem we're actually solving
- Street-level photos expose bystanders' faces and car plates, which is a privacy and legal (e.g. GDPR) liability.
- Every face and plate must be hidden before an image is published; one missed face is a user complaint.
- Volume is too high for manual review, so detection has to be automatic and run offline at scale.
- Mistakes must be catchable after release: users report bad images and those reports must improve the system.
- Latency is not the constraint; reliable coverage (recall) and cost-efficient batch processing are.

## What solution we'll provide (non-technical)
- A tool that takes a street photo and returns the same photo with faces and license plates blurred.
- A batch mode that processes whole folders of images unattended, while old images stay live.
- A quality score showing how reliably the system finds faces and plates before we ship a new version.
- A feedback loop: reported misses are turned into new training examples so the system keeps improving.
- A clear path to scale: swap in a faster model or more data without redesigning the pipeline.

## How we'll approach it (technical)
- Frame as 2-class object detection (face, license plate) with boxes `[x, y, w, h]`; start with a two-stage Faster R-CNN (torchvision), one-stage (YOLO/SSD) as the faster fallback.
- Data: small public datasets from GitHub for faces and plates, merged into one annotation format; resize, normalize and box-aware augmentation (flip, crop, color jitter).
- Train with combined loss `L = L_cls + λ·L_reg` (cross-entropy + box regression), seeded and config-driven.
- Evaluate offline with IoU-based precision, per-class AP and mAP (COCO/VOC style); online proxy is user reports and spot-check rate.
- Serve as a batch pipeline: CPU preprocessing → GPU detection → NMS → Gaussian blur → write to storage; hard negative mining from reports feeds retraining.
