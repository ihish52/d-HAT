checkpoints_path=$1
lat=${2:-1000}
configs=${3:-"configs/wmt14.en-de/subtransformer/wmt14ende_gpu_jetson_search0@1000ms.yml"}
metrics=${4:-"normal"}
gpu=${5:-0}
subset=${6:-"test"}

output_path=$(dirname -- "$checkpoints_path")
out_name=$(basename -- "$configs")

mkdir -p $output_path/exp

CUDA_VISIBLE_DEVICES=$gpu python3 -W ignore dHAT_generate.py \
        --data data/binary/wmt16_en_de  \
        --path "$checkpoints_path" \
        --gen-subset $subset \
		--lat-config $lat \
        --beam 4 \
        --batch-size 128 \
        --remove-bpe \
        --lenpen 0.6 \
        --configs=$configs \
        > $output_path/exp/${out_name}_${subset}_gen.out
