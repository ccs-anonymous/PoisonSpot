# PoisonSpot


---

## Arguments
Below is a list of arguments you can use with PoisonSpot:

| Argument               | Description                                                    | Default Value                      |
|------------------------|----------------------------------------------------------------|------------------------------------|
| `--batch_level`        | Enable batch-level weight updates                             |                                    |
| `--clean_training`     | Perform clean training                                        |                                    |
| `--poisoned_training`  | Perform poisoned training                                     |                                    |
| `--sample_level`       | Enable sample-level weight updates                            |                                    |
| `--score_samples`      | Score suspected samples                                       |                                    |
| `--retrain`            | Retrain the model                                             |                                    |
| `--pr_sus`             | Percentage of poisoned data in the suspected set             | `100`                              |
| `--ep_bl`              | Training epochs for batch-level weight capture               | `10`                               |
| `--ep_bl_base`         | Training epochs before batch-level capture                   | `200`                              |
| `--ep_sl`              | Training epochs for sample-level training                    | `10`                               |
| `--ep_sl_base`         | Training epochs before sample-level capture                  | `200`                              |
| `--pr_tgt`             | Ratio of poisoned data in the target set                     | `0.1`                              |
| `--bs_sl`              | Batch size for sample-level training                         | `128`                              |
| `--bs_bl`              | Batch size for batch-level training                          | `128`                              |
| `--bs`                 | Batch size for training                                      | `128`                              |
| `--eps`                | Epsilon for the attack                                       | `16`                               |
| `--vis`                | Visibility for label consistent attack                       | `255`                              |
| `--target_class`       | Target class for the attack                                  | `2`                                |
| `--source_class`       | Source class for the attack                                  | `0`                                |
| `--dataset`            | Dataset to use for the experiment                            | `"CIFAR10"`                       |
| `--attack`             | Attack to use for the experiment                             | `"lc"`                             |
| `--model`              | Model to use for the experiment                              | `"ResNet18"`                      |
| `--dataset_dir`        | Root directory for the datasets                              | `"./datasets/"`                    |
| `--clean_model_path`   | Path to the clean model                                      | `'./saved_models/resnet18_200_clean.pth'` |
| `--saved_models_path`  | Path to save the models                                      | `'./saved_models/'`                |
| `--global_seed`        | Global seed for the experiment                               | `545`                              |
| `--gpu_id`             | GPU ID to use for the experiment                             | `0`                                |
| `--lr`                 | Learning rate for the experiment                             | `0.1`                              |
| `--figure_path`        | Path to save the figures                                     | `"./results/"`                     |
| `--prov_path`          | Path to save the provenance data                             | `"./Training_Prov_Data/"`          |
| `--epochs`             | Number of epochs for either clean or poisoned training       | `200`                              |
| `--scenario`           | Scenario to use for the experiment                           | `"from_scratch"`                   |
| `--get_result`         | Get results from previous runs                               |                                    |
| `--force`              | Force the run                                                |                                    |
| `--threshold`          | Threshold for scoring suspected samples                     | `0.5`                              |
| `--sample_from_test`   | Sample from the test set                                     |                                    |
| `--cv_model`           | Model to use for cross-validation                            | `"RandomForest"`                   |
| `--groups`             | Number of groups to use for cross-validation                 | `5`                                |
| `--opt`                | Optimizer to use for the experiment                          | `"sgd"`                            |

---

### Clean and Unknown Dataset (PR_{D_cln ∪ D_unk})
Configure the poison ratio for **PR_{D_cln ∪ D_unk}** using the following values:

| Percentage (%)  | Parameter (`pr_tgt`) |
|------------------|-----------------------|
| 1%              | `0.1`                |
| 2%              | `0.2`                |
| 3%              | `0.3`                |
| 4%              | `0.4`                |
| 5%              | `0.5`                |
| 7.5%            | `0.75`               |
| 10%             | `1.0`                |

### Unknown Dataset (PR_{D_unk})
Configure the poison ratio for **PR_{D_unk}** using the following values:

| Percentage (%)  | Parameter (`pr_sus`) |
|------------------|-----------------------|
| 10%             | `10`                 |
| 25%             | `25`                 |
| 50%             | `50`                 |
| 75%             | `75`                 |
| 100%            | `100`                |


---



## Usage Examples

### Narcissus Attack
#### Poison Ratio: 1% Training Set (`pr_tgt 0.1`), 50% Suspected Set (`pr_sus 50`)
Replace pr_tgt and pr_sus with the desired values.

1. **Train the poisoned model:**
   ```bash
   python3 capture_prov.py --attack narcissus --target_class 2 --pr_tgt 0.1 --pr_sus 50 --poisoned_training --epochs 200
   ```

2. **Capture provenance data and retrain the model:**
   ```bash
   python3 capture_prov.py --attack narcissus --target_class 2 --pr_tgt 0.1 --pr_sus 50 --batch_level --sample_level --score_samples --retrain
   ```

---

### Label Consistent Attack
Replace pr_tgt and pr_sus with the desired values.

1. **Train the poisoned model:**
   ```bash
   python3 capture_prov.py --attack lc --target_class 2 --pr_tgt 0.1 --pr_sus 50 --poisoned_training --epochs 200
   ```

2. **Capture provenance data and retrain the model:**
   ```bash
   python3 capture_prov.py --attack lc --target_class 2 --pr_tgt 0.1 --pr_sus 50 --batch_level --sample_level --score_samples --retrain
   ```

---

### Sample-Level Training
#### Sleeper Agent Attack
Replace pr_tgt and pr_sus with the desired values.
1. **Train the poisoned model:**
   ```bash
   python3 capture_prov.py --attack sa --target_class 1 --source_class 0 --pr_tgt 0.1 --pr_sus 50 --poisoned_training --epochs 200
   ```

2. **Capture provenance data and retrain the model:**
   ```bash
   python3 capture_prov.py --attack sa --target_class 1 --source_class 0 --pr_tgt 0.1 --pr_sus 50 --batch_level --sample_level --score_samples --retrain
   ```

---

### Fine-Tuning: Sleeper Agent
Replace pr_tgt and pr_sus with the desired values.
1. **Fine-tune the model:**
   ```bash
   python3 capture_prov.py --attack sa --target_class 1 --source_class 0 --pr_sus 50 --pr_tgt 0.5 --scenario fine_tuning --clean_model_path ./saved_models/model_sa_resnet_200_128.pth --ep_bl_base 0 --sample_from_test --poisoned_training --epochs 10 --lr 0.01
   ```

2. **Capture provenance and retrain:**
   ```bash
   python3 capture_prov.py --attack sa --target_class 1 --source_class 0 --pr_sus 50 --pr_tgt 0.5 --scenario fine_tuning --clean_model_path ./saved_models/model_sa_resnet_200_128.pth --ep_bl_base 0 --sample_from_test --batch_level --ep_sl_base 0 --sample_level --score_samples --retrain --lr 0.01 --epochs 10
   ```

---

### Fine-Tuning: Hidden Trigger
1. **Fine-tune the model:**
   ```bash
   python3 capture_prov.py --attack ht --clean_model_path ./saved_models/htbd_art_model_200.pth --target_class 4 --source_class 3 --pr_tgt 0.5 --scenario fine_tuning --model CustomCNN --pr_sus 50 --sample_from_test --poisoned_training --ep_bl_base 0 --epochs 10 --lr 0.01
   ```

2. **Capture provenance and retrain:**
   ```bash
   python3 capture_prov.py --attack ht --clean_model_path ./saved_models/htbd_art_model_200.pth --target_class 4 --source_class 3 --pr_tgt 0.5 --scenario fine_tuning --model CustomCNN --pr_sus 50 --sample_from_test --ep_bl_base 0 --epochs 10 --batch_level --ep_sl_base 0 --sample_level --score_samples --retrain --lr 0.01
   ```

---

## Steps of PoisonSpot
1. **Batch Level (`--batch_level`)**
2. **Sample Level (`--sample_level`)**
3. **Poisoning Score Attribution (`--score_samples`)**
4. **Retraining (`--retrain`)**
