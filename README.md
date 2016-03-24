# deeptrackpy

Implementation of [Deep Tracking: Seeing Beyond Seeing Using Recurrent Neural Networks](http://arxiv.org/abs/1602.00991), Ondruska & Posner 2016, using Python.

Doesn't completely work yet, it reproduces the early parts of training pretty well but runs into numerical instability in later stages.


####Requires:
+ numpy
+ theano
+ lasagne
+ h5py

If using original dataset:

`chmod +x get_data.sh && ./get_data.sh`

If generating data:

`python generate_moving_objects.py --dest DATA_FILE --filestype hdf5 --frames 3000`

To train:

`python train.py`

Options:

`--num_epochs`, default 10

`--seq_len`, default 100

`--batch_size`, default 2
