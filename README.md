# MLPerf-on-A100
Benchmarking MLPerf on A100

# RECOMMENDATION Benchmark
# Applies **_Neural Collaborative Filtering (NCF)_** model on **_MovieLens 20 Million (ml-20m)_** dataset for recommendation
Instructions followed from: https://github.com/mlperf/training/tree/master/recommendation/pytorch

Step 1: Clone *mlperf/training* from github
git clone https://github.com/mlperf/training

cd training

## Step 2: Generate Expanded Dataset
Follow instructions from: https://github.com/mlperf/training/tree/master/data_generation/fractal_graph_expansions
let the data direcotry be **MY_DATA_DIR**
## Step 3: Build docker image for this benchmark
```bash
# Build from Dockerfile
cd training/recommendation/pytorch
sudo docker build -t mlperf/recommendation:v0.6 .
```

## Step 4: Start container by attaching data directory to it.
```bash
nvidia-docker run --rm -it --ipc=host --network=host -v /MY_DATA_DIR:/data/cache mlperf/recommendation:v0.6 /bin/bash
```
## Step 5: Generate Negative test samples within docker using
```bash
# Let, DATA Directory is visible in container as: /data/cache/ml-20mx16x32
# Generate -ve test sample by running below command within docker. 
python convert.py /data/cache/ml-20mx16x32 --seed 0
```
## Step 6: Start Running Training

Once the Expanded dataset and Negative Test Samples are available at **/data/cache/ml-20mx16x32**
we start training within docker using:
```bash
./run_and_time.sh <SEED>
```
