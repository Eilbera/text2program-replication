# Reproducing "Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records (CHIL 2022)"


This repository provides the official implementation of the [Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records](https://arxiv.org/abs/2203.06918).

## Requirements
- PyTorch == 1.7.1
- Python == 3.8.5
- transformers == 4.5.1
- numpy == 1.19.5
- pytorch-lightning == 1.3.2
- rdflib == 5.0.0

## Setup
git clone https://github.com/cyc1am3n/text2program-for-ehr.git
cd text2program-for-ehr/

### Changes made to initial code base
sed -i '/^class Text2TraceForTransformerModel/a\
\
    def transfer_batch_to_device(self, batch, device=None, dataloader_idx=0):\
        if device is None:\
            device = self.device\
        return {\
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)\
            for k, v in batch.items()\
        }' model/pl_model.py

sed -i 's/eval(json.loads(f.read()))/json.loads(f.read())/' model/evaluation.py

sed -i -E '
/^[[:space:]]*for model_id in range\(len\(model_list\)\):/,/^[[:space:]]*model_inputs\[key\] = None/ c\
            for model_id in range(len(model_list)):\
                device = next(model_list[model_id].parameters()).device  # model'\''s home GPU\
                for key, value in model_inputs.items():\
                    if value is None:\
                        continue\
                    if key != '\''past_key_values'\'':\
                        model_inputs[key] = value.to(device)\
                    else:\
                        model_inputs[key] = None
' model/ensemble_test.py



## Data
### Prepare Knowledge Graph
You should build knowledge graph for MIMICSPARQL* following instruction in [official MIMICSPARQL* github](https://github.com/junwoopark92/mimic-sparql).  
The KG(`mimic_sparqlstar_kg.xml`) file should be in `./data/db/mimicstar_kg` directory.

#python dep to run main repo
pip install uv
uv python pin 3.8
uv init
uv add transformers==4.5.1 numpy==1.19.5 pytorch-lightning==1.3.2 rdflib==5.0.0 pandas "packaging<21.0" sumeval torchmetrics==0.2.0 torch==2.2 scikit-learn

#preprocess step in main repo
cd /workspace/text2program-for-ehr/data && uv run preprocess.py
cd /workspace/text2program-for-ehr/

#aws CLI - to get S3 info over checkpoint (trained models)
apt update
apt install -y unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
mkdir -p ~/.aws


### Pre-process
Generate dictionary files for the recovery technique.
```shell script
$ cd data
$ python preprocess.py
```

## Train
```shell script
$ python main.py
```

## Test
```shell script
$ python main.py --test
```

## Citation
```
@article{kim2022uncertainty,
  title={Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records},
  author={Kim, Daeyoung and Bae, Seongsu and Kim, Seungho and Choi, Edward},
  journal={arXiv preprint arXiv:2203.06918},
  year={2022}
}
```
