# Reproducing “Uncertainty‑Aware Text‑to‑Program for Question Answering on Structured Electronic Health Records”

This repository accompanies the **CS 598: Deep Learning for Healthcare** reproduction study and extension of  
Kim *et al.* (AAAI 2022). It contains all code, data‑processing scripts, trained checkpoints, and instructions  
necessary to recreate our results and run the proposed **dynamic‑ensemble** extension.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1  Quick start

```bash
# 1 Clone the repo 
git clone https://github.com/Eilbera/text2program-replication.git
or original repo:
git clone https://github.com/cyc1am3n/text2program-for-ehr.git

cd text2program

# 2 Install UV and pinned Python 3.8 environment
pip install uv
uv python pin 3.8
uv init
uv add transformers==4.5.1 numpy==1.19.5 pytorch-lightning==1.3.2 rdflib==5.0.0 pandas "packaging<21.0" sumeval torchmetrics==0.2.0 torch==2.2 scikit-learn

# 2.5 ⚡ Apply compatibility patch to upstream code (PyTorch ≥2.x)

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
/^[[:space:]]*for model_id in range$begin:math:text$len\\(model_list$end:math:text$\):/,/^[[:space:]]*model_inputs$begin:math:display$key$end:math:display$ = None/ c\
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
You will need mimic.db and to build mimic_sparqlstar_kg.xml
*This step can take >2 h and >200 GB RAM; see Section 3.*
make build-kg

# 5 Train and evaluate single‑seed model
uv run main.py train  \
  --seed 42 --batch_size 18 --epochs 100 --lr 1e-4
uv run main.py test

# 6 Run 5‑seed ensemble (paper default)
uv run main.py ensemble --seeds 1 12 123 1234 42 --num-samples 1

# 7 Run dynamic‑ensemble extension (adaptive beam)
uv run scripts/dyn_ensemble.py --top-frac 0.33 --max-beam 7
```

> **Hardware**   All experiments were performed on an **NVIDIA A100 SXM. I also used AWS for pre process step: (ml.g5.8xlarge) with  
> 64 vCPU and 500 GB RAM.  Training ➜ ≈ 5 h; KG build ➜ ≈ 2.3 h.

## 2  Data access & KG construction

1. **Request MIMIC‑III Clinical Database**  
2. Download `mimic.db` (SQLite) and place it in `data/`.  
3. Run **TREQS** pipeline to obtain the *schema‑aligned* version:

   ```bash
   git clone https://github.com/wangpinggl/TREQS
   python TREQS/scripts/export_mimic_sqlite.py --src data/mimic.db --out data/mimic_sql/
   ```

3. Generate *SPARQL*‑ready KG:

   ```bash
   git clone https://github.com/junwoopark92/mimic-sparql
   cd mimic-sparql && pip install -r requirements.txt
   python build_mimic_kg.py --db data/mimic_sql/ --out ../data/mimic-sparqlstar.xml
   ```

4. Pre‑process KG into compact lookup tables used by NLQ2Program:

   ```bash
   uv run scripts/build_kg.py  # writes rel_obj_lookup.json & cond_lookup.json
   ```

> **Tip**   Building the KG locally requires ≈ 400 GB RAM. If your workstation cannot handle this,  
> copy `mimic.db` to AWS S3 and launch an on‑demand **ml.g5.8xlarge** SageMaker notebook, as I did.

---

## 3  Reproduction experiments

| Experiment                              | Command                                                | My results | Paper |
|-----------------------------------------|--------------------------------------------------------|-----------|-------|
| Single‑seed (seed 42)                   | `uv run main.py test`                                  | **0.948** | 0.947 |
| Dynamic ensemble († top 33 % → beam 7)  | `uv run scripts/dyn_ensemble.py --top-frac 0.33`       | **0.986** |   —   |


---

## 4  Extension — dynamic ensemble

The original work runs a fixed 5‑model ensemble or 1 ensemble during inference. We adaptively scale the ensemble size  
using **total uncertainty**:

1. Run ensemble‑1 to obtain token‑level entropy.  
2. Rank questions by mean entropy.  
3. Re‑evaluate the top *K %* most uncertain with a larger beam (up to 7).

This strategy improved AccEX from **0.9842 → 0.9863** under the same compute budget.  
Implementation in `scripts/dyn_ensemble.py`.

---

## 5  Citation

```text
@misc{mansour2025repro,
  author    = {Eilbera Mansour},
  title     = {Reproducing “Uncertainty‑Aware Text‑to‑Program for Question Answering on Structured Electronic Health Records”},
  year      = {2025},
  url       = {https://github.com/<your‑handle>/text2program‑repro}
}
```

---

## 6  License

MIMIC‑III data is subject to the PhysioNet Credentialed Health Data License and **must not** be redistributed.






