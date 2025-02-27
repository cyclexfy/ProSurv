#!/bin/bash

DATA_ROOT_DIR='/path/TCGA-BRCA/uni_features/pt_files'

TYPE_OF_PATH="combine"
MODEL="ProSurv" 
FUSION="concat"
STUDY="brca"

lrs=(0.00001)
mil_model_type="TransMIL"
memory_size=32
complete_rate=1.0
sim_loss=0.2
align_loss=0.2

test_modalitys=("path" "geno" "path_and_geno")
max_epochs=5

for lr in ${lrs[@]};
do
    CUDA_VISIBLE_DEVICES=7 python main.py \
        --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
        --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
        --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir results_${STUDY}/${MODEL} \
        --batch_size 1 --lr ${lr} --opt adam --reg 0.0001 \
        --alpha_surv 0.5 --weighted_sample --max_epochs ${max_epochs} --encoding_dim 1024 \
        --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 --fusion $FUSION \
        --input_modality path --mil_model_type ${mil_model_type} --memory_size ${memory_size} --complete_rate ${complete_rate} \
        --sim_loss ${sim_loss} --align_loss ${align_loss}
    
    for test_modality in ${test_modalitys[@]};
    do
        CUDA_VISIBLE_DEVICES=7 python main.py \
            --test_only True  --test_modality ${test_modality} \
            --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
            --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
            --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir results_${STUDY}/${MODEL} \
            --batch_size 1 --lr ${lr} --opt adam --reg 0.0001 \
            --alpha_surv 0.5 --weighted_sample --max_epochs ${max_epochs} --encoding_dim 1024 \
            --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 --fusion $FUSION \
            --input_modality path --mil_model_type ${mil_model_type} --memory_size ${memory_size} --complete_rate ${complete_rate} \
            --sim_loss ${sim_loss} --align_loss ${align_loss}
    done
done

