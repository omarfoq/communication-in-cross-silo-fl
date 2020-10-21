while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --network)
    NETWORK_NAME="$2"
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

NETWORK_NAME_TAG=""
if [ ! -z $NETWORK_NAME ]; then
    NETWORK_NAME_TAG="--network $NETWORK_NAME"
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

if [ ! -f raw_data/train2018.json ]; then
    echo "------------------------------"
    echo "downloading annotations and locations"

    cd raw_data
    wget http://www.vision.caltech.edu/~gvanhorn/datasets/inaturalist/fgvc5_competition/val2018.json.tar.gz
    wget http://www.vision.caltech.edu/~gvanhorn/datasets/inaturalist/fgvc5_competition/inat2018_locations.zip
    wget http://www.vision.caltech.edu/~gvanhorn/datasets/inaturalist/fgvc5_competition/train2018.json.tar.gz
    unzip inat2018_locations.zip -d .
    tar -xf val2018.json.tar.gz -C .
    tar -xf train2018.json.tar.gz -C .

    rm inat2018_locations.zip
    rm val2018.json.tar.gz
    rm train2018.json.tar.gz
    mv inat2018_locations/* .
    rm -r inat2018_locations
    echo "finished downloading annotations and locations"
    cd ../
fi

if [ ! -f test/test.json ]; then
    echo "------------------------------"
    echo "spliting data"
    mkdir train
    mkdir test

    python3 split_data.py  $NETWORK_NAME_TAG $NUM_WORKERS_TAG $SFRAC_TAG $TFRAC_TAG $SEED_TAG

    echo "finished splitting data"
fi