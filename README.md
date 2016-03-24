# deeptrackpy
=======

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
