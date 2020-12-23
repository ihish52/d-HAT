checkpoints_path=$1
lat=${2:-1000}
configs=${3:-"configs/wmt14.en-de/subtransformer/wmt14ende_jetson@1000ms.yml"}
metrics=${4:-"normal"}
gpu=${5:-0}
subset=${6:-"test"}

output_path=$(dirname -- "$checkpoints_path")
out_name=$(basename -- "$configs")

mkdir -p $output_path/exp

CUDA_VISIBLE_DEVICES=$gpu python dHAT_generate.py \
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
		
GEN=$output_path/exp/${out_name}_${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

# get normal BLEU or SacreBLEU score
if [ $metrics = "normal" ]
then
  echo "Evaluate Normal BLEU score!"
  grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
  grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
  python score.py --sys $SYS --ref $REF
elif [ $metrics = "sacre" ]
then
  echo "Evaluate SacreBLEU score!"
  grep ^H $GEN | cut -f3- > $SYS.pre
  grep ^T $GEN | cut -f2- > $REF.pre
  sacremoses detokenize < $SYS.pre > $SYS
  sacremoses detokenize < $REF.pre > $REF
  python score.py --sys $SYS --ref $REF --sacrebleu
fi