# Parse arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -nw)
    NUM_WORKERS="$2"
    shift # past argument
    shift # past value
    ;;
    -nc)
    NUM_CLASSES="$2"
    shift # past argument
    shift # past value
    ;;
    -dim)
    DIMENSION="$2"
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

NUM_WORKERS_TAG=""
if [ ! -z $NUM_WORKERS ]; then
    NUM_WORKERS_TAG="--num_workers $NUM_WORKERS"
fi

NUM_CLASSE_TAG=""
if [ ! -z $NUM_CLASSES ]; then
    NUM_CLASSES_TAG="--num_classes $NUM_CLASSES"
fi

DIMENSION_TAG=""
if [ ! -z $DIMENSION ]; then
    DIMENSION_TAG="--dimension $DIMENSION"
fi

TFRACTAG=""
if [ ! -z $TFRAC ]; then
    TFRAC_TAG="--tr_frac $TFRAC"
fi

SEED_TAG=""
if [ ! -z $SEED ]; then
    SEED_TAG="--seed $SEED"
fi


if [ ! -d "all_data" ]; then
  mkdir all_data
fi


if [ ! -f all_data/all_data.json ]; then
    echo "------------------------------"
    echo "generating  data"

    python3 generate_data.py  $NUM_WORKERS_TAG $NUM_CLASSES_TAG $DIMENSION_TAG $SEED_TAG

    echo "finished generating  data"
fi

if [ ! -f test/test.json ]; then
    echo "------------------------------"
    echo "spliting data"
    mkdir train
    mkdir test

    python3 split_data.py $TFRACTAG $SEED_TAG

    echo "finished splitting data"
fi