### Notable Results

During implementation of the convolutional neural network, I first implemented it using regular python loops, and then optimized/vectorized it using im2col and GEMM. These are my results when training the LeNet network architecture. Note that all of these results are done using CPU:

- Python Loop Implementation: 3585.65 seconds
- Im2Col + GEMM Implementation: 705.46 seconds

This is approximately a **5.1x** speedup.