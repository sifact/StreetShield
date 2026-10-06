# transforms.py

Box-aware image augmentation for training, and the plain eval transform.

- Applies colour jitter (brightness, contrast, saturation), a random crop and a horizontal flip, from the chapter's augmentation list.
- Every geometric change moves the boxes too: a flip mirrors x coordinates and a crop shifts them (the chapter's point that ground truth must be transformed with the image).
- The crop window always contains every box, because cutting a face in half would teach the model that half-faces are background.
- Augmentation runs online (on the fly) instead of offline as in the chapter: the dataset is small, so storing 10x copies buys nothing and online gives fresh variation each epoch.
- Resizing and normalisation are not done here; torchvision's detector does them internally, so train and serve preprocessing cannot drift apart.
