# losses.py

Combines the detector's loss terms into the chapter's overall loss, `L = L_cls + λ · L_reg`.

- Classification loss `L_cls` is the sum of the ROI classifier's cross-entropy (face / plate / background) and the RPN's objectness loss (object vs not).
- Regression loss `L_reg` is the sum of the ROI and RPN box-regression losses, which measure how well predicted boxes line up with ground truth.
- `λ` (`train.lambda_reg` in config) balances the two, exactly the balancing parameter in the chapter.
- torchvision uses smooth-L1 for box regression instead of the chapter's MSE; it behaves like MSE for small errors but is less dominated by outlier boxes.
- Returns a float breakdown of every term, so training logs show whether classification or localisation is the bottleneck.
