### Notable Results

During implementation of the convolutional neural network, I first implemented it using regular python loops, and then optimized/vectorized it using im2col and GEMM. These are my results when training the LeNet network architecture. Note that all of these results are done using CPU:

- Python Loop Implementation: 3585.65 seconds
- Im2Col + GEMM Implementation: 705.46 seconds

This is approximately a **5.1x** speedup.

### Hyperparameter Progression

Starting hyperparameters (from 9/26/2026):
- **Epochs**: 50
- **Learning Rate**: 0.002
- **Stop Factor LR**: 0.00001
- **Batch Size**: 256
- **LR Decay Factor**: 0.98
- **Weight initialization method**: Kaiming
- **Bias initialization method**: None

Achieved a 98.97% accuracy.

History:
- Changed **lr** to **0.001**. Increased accuracy to 99.11%
- Changed **batch size** to **128**. Increased accuracy to 99.29%