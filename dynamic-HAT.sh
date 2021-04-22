#Testing dynamic HAT implementation.

#space 0 -> for original design space
#space 1 -> new constrained design space

space=${1:-0}

if [[ $1 == 1 ]]
then
    bash configs/wmt14.en-de/d-HAT_run_generate.sh ./checkpoints/wmt14.en-de/supertransformer/HAT_wmt14ende_const_super_space1.pt 1000
else
    bash configs/wmt14.en-de/d-HAT_run_generate.sh ./downloaded_models/HAT_wmt14ende_super_space0.pt 1000
fi
