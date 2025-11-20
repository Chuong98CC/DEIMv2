export CUDA_VISIBLE_DEVICES=0
CFG=configs/traffic/deimv2_hgnetv2_n_traffic.yml
PRETRAINED=official_ckpts/deimv2_hgnetv2_n_coco.pth
# torchrun --master_port=7777 --nproc_per_node=1 train.py -c $CFG --use-amp --seed=0 
python train.py -c $CFG --use-amp --seed=0 
# -t $PRETRAINED