while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -nw)
    NUM_WORKERS="$2"
    shift # past argument
    shift # past value
    ;;
    -s)
    SAMPLE="$2"
    shift # past argument
    shift # past value
    ;;
    --sf)
    SFRAC="$2"
    shift # past argument
    shift # past value
    ;;
    --tf)
    TFRAC="$2"
    shift # past argument
    shift # past value
    ;;
    --seed)
    SEED="$2"
    shift # past argument
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

if [ ! -z $NUM_WORKERS ]; then
    NUM_WORKERS_TAG="--num_workers $NUM_WORKERS"
fi

SFRAC_TAG=""
if [ ! -z $SFRAC ]; then
    SFRAC_TAG="--s_frac $SFRAC"
fi

TFRAC_TAG=""
if [ ! -z $TFRAC ]; then
    TFRAC_TAG="--tr_frac $TFRAC"
fi

SEED_TAG=""
if [ ! -z $SEED ]; then
    SEED_TAG="--seed $SEED"
fi

if [ $SAMPLE = "iid" ]; then
    python3 split_data.py  --iid $NUM_WORKERS_TAG $SFRAC_TAG $TFRAC_TAG $SEED_TAG
else
    python3 split_data.py $NUM_WORKERS_TAG $SFRAC_TAG $TFRAC_TAG $SEED_TAG
fi