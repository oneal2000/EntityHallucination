# DRAD
The source code of our submitted paper: DRAD

# Overview

[TODO]

# Install environment

```bash
conda create -n drad python=3.7
conda activate drad
pip install -r requirements.txt
```

# Run DRAD

## Build Wikipedia index

Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command:

```bash
mkdir data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

Use Elasticsearch to index the Wikipedia dump:

```bash
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
pushd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
popd
python prep.py --task build_elasticsearch --inp data/dpr/psgs_w100.tsv wiki  # build index
```

## Download Dataset

Take Natural Questions as an example:

```bash
mkdir data/nq
wget -O data/nq/nq-dev-all.jsonl.gz https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz
```

## Run DRAD

Let’s continue taking NQ as an example:

```bash
python3 src/main.py \
    --model_name_or_path model_name_or_path \
    --method entity \
    --hallucination_threshold 0.4 \
    --entity_solver avg \
    --sentence_solver avg \
    --dataset nq \
    --data_path data/nq \
    --generate_max_length 64 \
    --output_dir result \
    --fewshot 0 
```

You can modify the parameters to meet your running requirements. Note that if you want to use the GPT model to run, for example `text-davinci-003`, please use `gpt-text-davinci-003` in `model_name_or_path` to indicate that this is a GPT model.

# Citation

[TODO]