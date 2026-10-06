# trainer.py

Fine-tunes the two-stage detector and keeps the checkpoint that scores best on the validation split.

- Each step is the chapter's training loop: forward pass → combined loss `L_cls + λ·L_reg` (`losses.py`) → backward pass → SGD update.
- Uses linear LR warmup for the first iterations, then a 10x step decay at the configured epochs, so the freshly replaced heads don't destabilise the pretrained backbone.
- Skips any batch whose loss is not finite instead of letting one bad batch corrupt the weights.
- After every epoch it runs the evaluator on the val split and saves the checkpoint with the best mAP@0.5 (more stable than mAP@[.5:.95] on a small val set).
- Saves the full state (weights, optimizer, scheduler, history) after every epoch to `artifacts/last_state.pt`. The CLI runs each epoch in a fresh process from that state, because PyTorch's Apple MPS backend leaks host RAM until the process exits; `--resume` continues an interrupted run. Shuffling and augmentation are seeded per epoch, so this doesn't change results.
