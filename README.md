# Faster, lighter Facenet for face recognition using pytorch
This code is based on Pytorch implementation of the paper: "FaceNet: A Unified Embedding for Face Recognition and Clustering".
at https://github.com/tbmoon/facenet

The current model uses triplet loss and reduced number of levels.
Final recognition rates are comparable to the original numbers (+- 2%) but the resulting model is faster and smaller.


# Designed for deployment

As of 2019.05 the OpenCV can load Pytorch models but the support for certain layers is limited.
Therefore this model is using only layers which are supported by ONNX and OpenCV DNN.

The export.py script is there to convert the trained model into the opencv-compatible onnx format.


# References
- https://github.com/tbmoon/facenet
- https://github.com/liorshk/facenet_pytorch 
- https://github.com/davidsandberg/facenet
- https://arxiv.org/abs/1503.03832
