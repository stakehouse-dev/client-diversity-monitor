# `Blockswap Client Diversity`

Blocksap Client Diversity is a tool that adopts Blockprint for measuring client diversity of LSD and Open Index validators on the Ethereum beacon chain, based on trained classification models using block rewards and meta data.

This tool was forked from: https://github.com/sigp/blockprint

## Running `Blockswap Client Diversity`

Startup of the API can be done by executing the background_task.py script. This will start the API server and the background task that will fetch new blocks from the Lighthouse API and classify them if they are a block proposal from a LSD or Open Index validator.

### Lighthouse

The Blockswap Client Diversity API needs to be configured to connect to a Lighthouse node v2.1.2 or newer.

It uses the [`/lighthouse/analysis/block_rewards`][block_rewards_endpoint] endpoint.

[block_rewards_endpoint]: https://lighthouse-book.sigmaprime.io/api-lighthouse.html

### VirtualEnv

All Python commands should be run from a virtualenv with the dependencies from `requirements.txt`
installed.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### LSD Subgraph and Validator Indexing

The Blockswap Client Diversity API needs to be configured to connect to a Graph node running the LSD subgraph.
This API also connects to Blockswap labs' validator indexing ETL pipeline for LSD and Open Index validators.

### k-NN Classifier

The classifier is a k-nearest neighbours classifier in `knn_classifier.py`.

See `./knn_classifier.py --help` for command line options including cross
validation (CV) and manual classification.

### Training the Classifier

The classifier is trained from a directory of reward batches. You can fetch batches with the
`load_blocks.py` script by providing a start slot, end slot and output directory:

```
./load_blocks.py 2048001 2048032 testdata
```

The directory `testdata` now contains 1 or more files of the form `slot_X_to_Y.json` downloaded
from Lighthouse.

To train the classifier on this data, use the `prepare_training_data.py` script:

```
./prepare_training_data.py testdata testdata_proc
```

This will read files from `testdata` and write the graffiti-classified training data to
`testdata_proc`, which is structured as directories of _single_ block reward files for each
client.

```
$ tree testdata_proc
testdata_proc
├── Lighthouse
│   ├── 0x03ae60212c73bc2d09dd3a7269f042782ab0c7a64e8202c316cbcaf62f42b942.json
│   └── 0x5e0872a64ea6165e87bc7e698795cb3928484e01ffdb49ebaa5b95e20bdb392c.json
├── Nimbus
│   └── 0x0a90585b2a2572305db37ef332cb3cbb768eba08ad1396f82b795876359fc8fb.json
├── Prysm
│   └── 0x0a16c9a66800bd65d997db19669439281764d541ca89c15a4a10fc1782d94b1c.json
└── Teku
    ├── 0x09d60a130334aa3b9b669bf588396a007e9192de002ce66f55e5a28309b9d0d3.json
    ├── 0x421a91ebdb650671e552ce3491928d8f78e04c7c9cb75e885df90e1593ca54d6.json
    └── 0x7fedb0da9699c93ce66966555c6719e1159ae7b3220c7053a08c8f50e2f3f56f.json
```

You can then use this directory as the datadir argument to `./knn_classifier.py`:

```
./knn_classifier.py testdata_proc --classify testdata
```

If you then want to use the classifier to build an sqlite database:

```
./build_db.py --db-path block_db.sqlite --classify-dir testdata --data-dir testdata_proc
```


### Running the API server

```
gunicorn -b localhost:7000 --workers=4 api_server:app --timeout 1800 --reload
```

It will take a few minutes to start-up while it loads all of the training data into memory.

### License

Copyright 2023 Blockswap Labs, Sigma Prime and blockprint contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
