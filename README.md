# PoisonSpot

This are the arguments to run PoisonSpot:

--batch_level: Enable batch level weight updates
--clean_training: Do clean training
--poisoned_training: Do poisoned training
--sample_level: Enable sample level weight updates
--score_samples: Enable scoring of suspected samples
--retrain: Enable retraining of the model
--pr_sus: Percentage of poisoned data in the suspected set (default: 100)
--ep_bl: Number of training epochs for batch level weight capture (default: 10)
--ep_bl_base: Number of training epochs before batch level capture (default: 200)
--ep_sl: Number of training epochs for sample level training (default: 10)
--ep_sl_base: Number of training epochs before sample level capture (default: 200)
--pr_tgt: Ratio of poisoned data in the target set (default: 0.1)
--bs_sl: Batch size for sample level training (default: 128)
--bs_bl: Batch size for batch level training (default: 128)
--bs: Batch size for training (default: 128)
--eps: Epsilon for the attack (default: 16)
--vis: Visibility for label consistent attack (default: 255)
--target_class: Target class for the attack (default: 2)
--source_class: Source class for the attack (default: 0)
--dataset: Dataset to use for the experiment (default: "CIFAR10")
--attack: Attack to use for the experiment (default: "lc")
--model: Model to use for the experiment (default: "ResNet18")
--dataset_dir: Root directory for the datasets (default: "./datasets/")
--clean_model_path: Path to the clean model (default: './saved_models/resnet18_200_clean.pth')
--saved_models_path: Path to save the models (default: './saved_models/')
--global_seed: Global seed for the experiment (default: 545)
--gpu_id: GPU ID to use for the experiment (default: 0)
--lr: Learning rate for the experiment (default: 0.1)
--figure_path: Path to save the figures (default: "./results/")
--prov_path: Path to save the provenance data (default: "./Training_Prov_Data/")
--epochs: Number of epochs for either clean or poisoned training (default: 200)
--scenario: Scenario to use for the experiment (default: "from_scratch")
--get_result: Get results from previous runs
--force: Force the run
--threshold: Threshold for scoring suspected samples (default: 0.5)
--sample_from_test: Sample from the test set
--cv_model: Model to use for cross validation (default: "RandomForest")
--groups: Number of groups to use for cross validation (default: 5)
--opt: Optimizer to use for the experiment (default: "sgd")




Use the following template to run PoisonSpot:
Example: Narcissus Poison ratio (1% training set or 10 % of target set)  = 0.1  Poison ratio (50% suspected set) = 50

1% training set -- pr_tgt 0.1 
2% training set -- pr_tgt 0.2
3% training set -- pr_tgt 0.3
There are four steps of PoisonSpot: batch level (batch_level). sample level (sample_level), poisoning score attribution (score_smaples), and retraining (retrain). 

First train accurate poisoned model.
python3 capture_prov.py --attack Narcissus --target_class 2 --pr_tgt 0.1 --pr_sus 50 --poisoned_training --epochs 200 
Then capture the provenance data for the poisoned model, and retrain the model.
python3 capture_prov.py --attack Narcissus --target_class 2 --pr_tgt 0.1 --pr_sus 50 --batch_level --sample_level --score_samples --retrain


For Label Consistent 
python3 capture_prov.py --attack lc --target_class 2 --pr_tgt 0.1 --pr_sus 50 --poisoned_training --epochs 200
python3 capture_prov.py --attack Narcissus --target_class 2 --pr_tgt 0.1 --pr_sus 50 --batch_level --sample_level --score_samples --retrain

For sample-level 
python3 capture_prov.py --attack sa  --target_class 1 --source_class 0 --pr_tgt 0.1 --pr_sus 50 --poisoned_training --epochs 200
python3 capture_prov.py --attack sa  --target_class 1 --source_class 0 --pr_tgt 0.1 --pr_sus 50 --batch_level --sample_level --score_samples --retrain


For fine-tuning 

python3 capture_prov.py --attack sa --target_class 1 --source_class 0 --pr_sus 50 --pr_tgt 0.5 --scenario fine_tuning --clean_model_path ./saved_models/model_sa_resnet_200_128.pth --ep_bl_base 0  --sample_from_test --poisoned_training --epochs 10 --lr 0.01

python3 capture_prov.py --attack sa --target_class 1 --source_class 0 --pr_sus 50 --pr_tgt 0.5 --scenario fine_tuning --clean_model_path ./saved_models/model_sa_resnet_200_128.pth --ep_bl_base 0   --sample_from_test --batch_level --ep_sl_base 0 --sample_level --score_samples -–retrain   --lr 0.01  --epochs 10