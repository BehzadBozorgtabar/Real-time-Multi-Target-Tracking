# Real-time-Multi-Target-Tracking

In this work, a Real-time Multi-Target Tracking using region-based fully convolutional neural network has been presented. 

## Usage

Download MOT16 dataset and pre-trained models from the following links. Copy the model in `data`, then build and run the code. 

```bash
pip install -r requirements.txt
sh make.sh
python perform_tracker.py
```


## Dependencies
- [Python2.7](https://www.anaconda.com/download/#linux)
- [PyTorch](http://pytorch.org/)
- [torchvision](http://pytorch.org/docs/master/torchvision)
- [OpenCV](https://opencv.org/)


![](https://github.com/BehzadBozorgtabar/Real-time-Multi-Target-Tracking/blob/master/Tracker_Screenshot.png)


### Pre-trained Model

Region-based FCN Model: https://drive.google.com/file/d/1sLGZ95gMVvpOl2wzW8yj99FkLETMt7W1/view?usp=sharing
MOT 2016 dataset : https://motchallenge.net/data/MOT16/
