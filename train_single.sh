cross_validation(){
    fn=$(echo $2 | sed "s/.pt/0.pt/")
    python scripts/main.py train --model-name=$1 --file-name=$fn --run-name=$3
    python scripts/main.py evaluate --model-name=$1 --file-name=$fn --run-name=$3
    sed -i 's/"validation_fold": 0/"validation_fold": 1/g' config/args.json

    fn=$(echo $2 | sed "s/.pt/1.pt/")
    python scripts/main.py train --model-name=$1 --file-name=$fn --run-name=$3
    python scripts/main.py evaluate --model-name=$1 --file-name=$fn --run-name=$3
    sed -i 's/"validation_fold": 1/"validation_fold": 2/g' config/args.json

    fn=$(echo $2 | sed "s/.pt/2.pt/")
    python scripts/main.py train --model-name=$1 --file-name=$fn --run-name=$3
    python scripts/main.py evaluate --model-name=$1 --file-name=$fn --run-name=$3
    sed -i 's/"validation_fold": 2/"validation_fold": 3/g' config/args.json

    fn=$(echo $2 | sed "s/.pt/3.pt/")
    python scripts/main.py train --model-name=$1 --file-name=$fn --run-name=$3
    python scripts/main.py evaluate --model-name=$1 --file-name=$fn --run-name=$3
    sed -i 's/"validation_fold": 3/"validation_fold": 4/g' config/args.json

    fn=$(echo $2 | sed "s/.pt/4.pt/")
    python scripts/main.py train --model-name=$1 --file-name=$fn --run-name=$3
    python scripts/main.py evaluate --model-name=$1 --file-name=$fn --run-name=$3
    sed -i 's/"validation_fold": 4/"validation_fold": 0/g' config/args.json
}

cross_validation Conv_LSTM_single Conv_LSTM_single.pt Conv_LSTM_single
