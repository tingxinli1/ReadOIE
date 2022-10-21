# ReadOIE
**SDM 2023 submission**: *Coarse-to-Fine Open Information Extraction via Relation Oriented Reading Comprehension*

To reproduce the experiments, please firstly download some files:
- BERT initial weight (`pytorch_model.bin`): https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin, put it under `./bert-base-uncased/`. Note that `vocab.txt` is modified, so please do not overwrite it with original one.
- OPIEC official sample (`OPIEC-Clean-example.avro`): http://data.dws.informatik.uni-mannheim.de/opiec/OPIEC-Clean-example.avro , put it under `./data/`

You may also need to configure an envirenment:
- run `conda create --name readoie python=3.7` to create a new environment
- install all the packages using `pip install -r requirements.txt`
- install spacy models using `python -m spacy download en_core_web_sm` and `python -m spacy download en_core_web_trs`

After that, prepare the training data:
- run `python preprocess1.py` to prepare data for intensive extrating module
- run `python preprocess2.py` to prepare data for extensive detecting module

Then, start training:
- run `python train_extensive.py` to train intensive extrating module
- run `python train_intensive.py` to train extensive detecting module

Finally, run `python infer.py --task_name TASK_NAME` to test on benchmarks, where `TASK_NAME` can be one of `web`, `nyt`, and `penn`.
