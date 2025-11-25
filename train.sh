export CUDA_VISIBLE_DEVICES=0,2
CFG=configs/traffic/deimv2_hgnetv2_n_traffic.yml
PRETRAINED=coco_ckpts/deimv2_hgnetv2_n_coco.pth
# torchrun --master_port=7777 --nproc_per_node=2 train.py -c $CFG --use-amp --seed=0 -r $PRETRAINED
# torchrun --master_port=7777 --nproc_per_node=2 train.py -c $CFG --use-amp --seed=0 -t $PRETRAINED
python train.py -c $CFG --use-amp --seed=0  -t $PRETRAINED