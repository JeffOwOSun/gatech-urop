# dance data

## 1. prepare the data
`dance/prep-data.py` will process the images put in `dance/data` directory. 

It will produce a `todo.json` recording the source image and 256x256 thumbnails 
to be generated, as well as class information. `todo.json` will be read if it
exists when `prep-data.py` runs.

The dataset is split into a 70/30 train/val set based on videos.

Thumbnails are stored in `dance/data/thumbs`.

absolute paths of training and validation set and corresponding class
label specifications required by caffe are stored in `dance/data/(train|val).txt`.

## 2. Train the network
`dance/train.py` will do the job. Basically it loads the `solver.prototxt` and
`train_val.prototxt` in `dance/models/caffenet` and run single step through a 
number of iterations while recording the training and testing accuracy along 
the way. 

Snapshots are carried out by caffe as specified in solver.prototxt.

In `solver.prototxt` a dummy test interval is supplied such that in practice no
testing is carried out by caffe. All tests are taken care of by
`dance/train.py`.

When everything is done, the script will save a
`best_weight_<iteration>.caffemodel` inside `dance/data/models/caffenet`, which
corresponds to the highest testing accuracy model. A learning curve is plotted
in `dance/data/models/caffenet/train_val.png`. All the training accuracy, losses
and those of the testing epochs are stored in log.json for future analysis.
