# Reproducing “Uncertainty‑Aware Text‑to‑Program for Question Answering on Structured Electronic Health Records”

This repository accompanies the **CS 598: Deep Learning for Healthcare** reproduction study and extension of  
Kim *et al.* (AAAI 2022). It contains all code, data‑processing scripts, trained checkpoints, and instructions  
necessary to recreate our results and run the proposed **dynamic‑ensemble** extension.

---

## 1  Quick start

```bash
# 1 Clone the repo (submodule contains original authors’ code)
git clone --recursive https://github.com/<your‑handle>/text2program‑repro.git
cd text2program‑repro

# 2 Install UV and pinned Python 3.8 environment
pip install uv
uv python pin 3.8
uv init

# 3 Install all runtime dependencies (CUDA 11.8 assumed)
uv add \
  transformers==4.5.1 \
  torch==2.2 \
  torchmetrics==0.2.0 \
  pytorch-lightning==1.3.2 \
  rdflib==5.0.0 \
  pandas "packaging<21.0" \
  numpy==1.19.5 \
  scikit-learn

# 4 Fetch MIMIC‑III (credentialed users only) and build the KG
#    *This step can take >2 h and >200 GB RAM; see Section 3.*
make build-kg

# 5 Train and evaluate single‑seed model
uv run main.py train  \
  --seed 42 --batch_size 18 --epochs 100 --lr 1e-4
uv run main.py test

# 6 Run 5‑seed ensemble (paper default)
uv run main.py ensemble --seeds 1 12 123 1234 42 --num-samples 1

# 7 Run dynamic‑ensemble extension (adaptive beam)
uv run scripts/dyn_ensemble.py --top-frac 0.33 --max-beam 7

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


##python dep to run main repo
pip install uv
uv python pin 3.8
uv init
uv add transformers==4.5.1 numpy==1.19.5 pytorch-lightning==1.3.2 rdflib==5.0.0 pandas "packaging<21.0" sumeval torchmetrics==0.2.0 torch==2.2 scikit-learn

##preprocess step in main repo
cd /workspace/text2program-for-ehr/data && uv run preprocess.py
cd /workspace/text2program-for-ehr/

##aws CLI - to get S3 info over checkpoint (trained models)
apt update
apt install -y unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
mkdir -p ~/.aws


##bash
cp credentials ~/.aws

## Data
### Prepare Knowledge Graph
You should build knowledge graph for MIMICSPARQL* following instruction in [official MIMICSPARQL* github](https://github.com/junwoopark92/mimic-sparql).  
The KG(`mimic_sparqlstar_kg.xml`) file should be in `./data/db/mimicstar_kg` directory.

### Download checkpoints

mkdir -p /workspace/text2program-for-ehr/saved/models/pretrained_models/natural/train/t5-base && \
cd /workspace && \
for d in ne100_lr0.0001_s2s_{1,2,3,4,5,6,7,8,12,42,123,1234}; do \
  aws s3 cp s3://ehr-mimic-checkpoints/${d}.zip . --region us-east-2 && \
  unzip -o ${d}.zip -d /workspace/text2program-for-ehr/saved/models/pretrained_models/natural/train/t5-base/; \
done
### Pre-process
Generate dictionary files for the recovery technique.
```shell script
$ cd data
$ python preprocess.py
```
##bash
uv run main.py --seed 42 --gpu_id 0
uv run main.py --seed 1 --gpu_id 1
uv run main.py --seed 12 --gpu_id 2
uv run main.py --seed 123 --gpu_id 3

uv run main.py --seed 1234 --gpu_id 0
uv run main.py --seed 2 --gpu_id 1
uv run main.py --seed 3 --gpu_id 2
uv run main.py --seed 4 --gpu_id 3



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




## Train
```shell script
$ python main.py
```

```bash
uv run main.py --seed 42 --gpu_id 0
uv run main.py --seed 1 --gpu_id 1
uv run main.py --seed 12 --gpu_id 2
uv run main.py --seed 123 --gpu_id 3

uv run main.py --seed 1234 --gpu_id 0
uv run main.py --seed 2 --gpu_id 1
uv run main.py --seed 3 --gpu_id 2
uv run main.py --seed 4 --gpu_id 3
```

## Test
```shell script
$ python main.py --test
```

# Reproduce Table 2 T5 Column:
```bash
pip install uv

uv run main.py --test
```


# Reproduce Figure 3:
```bash
uv run main.py --ensemble_test --gpu_id 0,1 --num_samples 5 --ensemble_seed 42,1,12,123,1234
```

Extension - Uncertainty Aware Test Time Scaling 
```bash
uv run main.py --ensemble_test --gpu_id 0,1 --num_samples 5 --ensemble_seed 42,1,12,123,1234
uv run main.py --ensemble_test --gpu_id 0,1 --num_samples 5 --ensemble_seed 42,1,12,123,1234
uv run main.py --ensemble_test --gpu_id 0,1 --num_samples 5 --ensemble_seed 42,1,12,123,1234
uv run main.py --ensemble_test --gpu_id 0,1 --num_samples 5 --ensemble_seed 42,1,12,123,1234
uv run main.py --ensemble_test --gpu_id 0,1 --num_samples 5 --ensemble_seed 42,1,12,123,1234

## Citation
```
@article{kim2022uncertainty,
  title={Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records},
  author={Kim, Daeyoung and Bae, Seongsu and Kim, Seungho and Choi, Edward},
  journal={arXiv preprint arXiv:2203.06918},
  year={2022}
}
```
