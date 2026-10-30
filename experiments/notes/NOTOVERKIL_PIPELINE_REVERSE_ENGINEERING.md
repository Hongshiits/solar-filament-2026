# notoverkil pipeline reverse-engineering (negative results)

Goal: reproduce the public 0.41 pipeline from its released weights. Two components were attempted.

## Refiner (refiner/fold-{0..4}.pt)
- Architecture recovered exactly: ResNet34(3,4,6,3) backbone + U-Net decoder, 2-channel 256x256 input, 24.44M params, strict load OK (decoder BNs use track_running_stats=False).
- Preprocessing convention could not be recovered. Candidate-mask vs ref-mask mean IoU = 0.894 unrefined; refined output under 15 tested conventions (crop padding 1.1/1.2/1.5/2.0/2.5, image normalization div255/ImageNet/pm1, channel order image-mask vs mask-image, mask representation binary/distance/signed-distance, full-image 256 mode) peaks at 0.56 -> the refiner output is not a simple mask refinement of our instances.
- Conclusion: abandoned; their refinement step uses an undocumented input representation.

## Mask R-CNN family (p1/p2 maskrcnn1536)
- Checkpoint contains EMA weights + config; torchvision maskrcnn_resnet50_fpn_v2(1536) loads with 0 missing keys (in_ch=3, n_cls=2).
- Calibration on their own fold0 val (141 images, honest): PQ 0.248 (score 0.3), 0.285 (score 0.15) with div255 input; ImageNet double-normalization yields 0 detections (model already normalizes internally).
- Our 15-model YOLO consensus uses only the three YOLO families; adding maskrcnn at this quality would not help.

## Implication
The 0.41 public score is not reachable from the published artifacts alone; our best independent reproduction is 0.39 (15-model YOLO consensus, 1259-instance vote filter), consistent with the leaderboard evidence that ~0.39-0.41 is the legitimate ceiling range and >0.45 implies leaked test annotations.
