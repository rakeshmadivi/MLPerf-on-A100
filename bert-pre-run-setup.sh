#!/bin/bash
# FOLLOWING IS IN REFERENCE TO: https://github.com/mlperf/training_results_v0.7/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch
# -----------------------------------------------------------------------------------------------------------------------------------------

# Download Input Files
download(){
curl -k https://storage.googleapis.com/pkanwar-bert/bs64k_32k_ckpt/bert_config.json -o bert_config.json
curl -k https://storage.googleapis.com/pkanwar-bert/bs64k_32k_ckpt/model.ckpt-28252.data-00000-of-00001 -o model.ckpt-28252.data-00000-of-00001
curl -k https://storage.googleapis.com/pkanwar-bert/bs64k_32k_ckpt/model.ckpt-28252.index -o model.ckpt-28252.index
curl -k https://storage.googleapis.com/pkanwar-bert/bs64k_32k_ckpt/model.ckpt-28252.meta -o model.ckpt-28252.meta
curl -k https://storage.googleapis.com/pkanwar-bert/bs64k_32k_ckpt/vocab.txt -o vocab.txt
}

# Download and preprocess datasets
preProcessDataset(){
cd cleanup_scripts  
mkdir -p wiki  
cd wiki  
wget https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2    # Optionally use curl instead  
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2  
cd ..    # back to bert/cleanup_scripts  
git clone https://github.com/attardi/wikiextractor.git  
python3 wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml    # Results are placed in bert/cleanup_scripts/text  
./process_wiki.sh '<text/*/wiki_??'  
}

# Checkpoint conversion
checkpointConversion(){
python convert_tf_checkpoint.py --tf_checkpoint /cks/model.ckpt-28252 --bert_config_path /cks/bert_config.json --output_checkpoint model.ckpt-28252.pt
}

# Generate the BERT input dataset
# The create_pretraining_data.py script duplicates the input plain text, replaces different sets of words with masks for each duplication, and serializes the output into the TFRecord file format.

createBERTInputDataset(){
python3 create_pretraining_data.py \
   --input_file=<path to ./results of previous step>/part-XX-of-00500 \
   --output_file=<tfrecord dir>/part-XX-of-00500 \
   --vocab_file=<path to vocab.txt> \
   --do_lower_case=True \
   --max_seq_length=512 \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
# The generated tfrecord has 500 parts, totalling to ~365GB.
}

# Running the model

# Building the Docker container
# docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
# docker push <docker/registry>/mlperf-nvidia:language_model

# To run this model, use the following command. Replace the configuration script to match the system being used (e.g., config_DGX1.sh, config_DGX2.sh, config_DGXA100.sh).
# source config_DGXA100.sh
# sbatch -N${DGXNNODES} --ntasks-per-node=${DGXNGPU} --time=${WALLTIME} run.sub
# Alternative launch with nvidia-docker. Replace the configuration script to match the system being used (e.g., config_DGX1.sh, config_DGX2.sh, config_DGXA100.sh).

# docker build --pull -t mlperf-nvidia:language_model .
launchTest(){
source config_DGXA100.sh
CONT=mlperf-nvidia:language_model DATADIR=<path/to/datadir> DATADIR_PHASE2=<path/to/datadir_phase2> EVALDIR=<path/to/evaldir> CHECKPOINTDIR=<path/to/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1> ./run_with_docker.sh
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# All configuration files follow the format config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>x<GRADIENT_ACCUMULATION_STEPS>.sh
# ----------------------------------------------------------------------------------------------------------------------------------------------------

download
preProcessDataset
checkpointConversion
#createBERTInputDataset
#launchTest
