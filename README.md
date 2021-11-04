# CMSC742
Folder structure:
- Root
  - Datasets
  - logs
  - BMCNNwHFCs

Training:
1. call go function from python/train.py
2. go function sets up parameters and call loops class from constructs/loops.py to run training loops
3. training loops call model forward functions to compute estimates
4. forward function utilize functions in nn_ops to compute output, which include tf.nn.conv2d
