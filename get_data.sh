#!/bin/bash

mkdir -p data
wget https://dl.dropboxusercontent.com/u/63070080/deeptrack.hdf5.bz2 -P data
bzip2 -d data/deeptrack.hdf5.bz2
