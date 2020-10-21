if [ ! -d "raw_data" ]; then
  mkdir raw_data
fi

if [ ! -f raw_data/test.csv ]; then
    echo "------------------------------"
    echo "retrieving raw data"

    cd raw_data

    if [ ! -f trainingandtestdata.zip ]; then
        wget --no-check-certificate http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
    fi

    unzip trainingandtestdata.zip

    mv training.1600000.processed.noemoticon.csv training.csv
    mv testdata.manual.2009.06.14.csv test.csv

    rm trainingandtestdata.zip

    cd ../
    echo "finished retrieving raw data"

    echo "------------------------------"
    echo "combining raw_data .csv files"

    python3 combine_data.py

    echo "finished combining raw_data .csv files"

fi
if [ ! -f test/test.json ]; then
    echo "------------------------------"
    echo "spliting data"
    mkdir train
    mkdir test

    ./split_data.sh "$@"

    echo "finished splitting data"
fi