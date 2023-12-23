cross_validation(){
    python scripts/main.py train --model-save-name=$1 --run-name=$2
    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 0/"validation_fold": 1/g' config/args.json

    python scripts/main.py train --model-save-name=$1 --run-name=$2
    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 1/"validation_fold": 2/g' config/args.json

    python scripts/main.py train --model-save-name=$1 --run-name=$2
    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 2/"validation_fold": 3/g' config/args.json

    python scripts/main.py train --model-save-name=$1 --run-name=$2
    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 3/"validation_fold": 4/g' config/args.json

    python scripts/main.py train --model-save-name=$1 --run-name=$2
    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 4/"validation_fold": 0/g' config/args.json
}

sed -i 's/get_model = ConvNext_base3/get_model = ConvNext_LSTM/g' scripts/main.py

sed -i 's/"num_frames": 5/"num_frames": 1/g' config/args.json
sed -i 's/"output_frames": -5/"output_frames": -1/g' config/args.json
cross_validation Conv_LSTM_1.pt Conv_LSTM_1

sed -i 's/"num_frames": 1/"num_frames": 2/g' config/args.json
sed -i 's/"output_frames": -1/"output_frames": -2/g' config/args.json
cross_validation Conv_LSTM_2.pt Conv_LSTM_2

sed -i 's/"num_frames": 2/"num_frames": 3/g' config/args.json
sed -i 's/"output_frames": -2/"output_frames": -3/g' config/args.json
cross_validation Conv_LSTM_3.pt Conv_LSTM_3

sed -i 's/"num_frames": 3/"num_frames": 4/g' config/args.json
sed -i 's/"output_frames": -3/"output_frames": -4/g' config/args.json
cross_validation Conv_LSTM_4.pt Conv_LSTM_4

sed -i 's/"num_frames": 4/"num_frames": 6/g' config/args.json
sed -i 's/"output_frames": -4/"output_frames": -6/g' config/args.json
cross_validation Conv_LSTM_6.pt Conv_LSTM_6

sed -i 's/"num_frames": 6/"num_frames": 7/g' config/args.json
sed -i 's/"output_frames": -6/"output_frames": -7/g' config/args.json
cross_validation Conv_LSTM_7.pt Conv_LSTM_7

sed -i 's/"num_frames": 7/"num_frames": 8/g' config/args.json
sed -i 's/"output_frames": -7/"output_frames": -8/g' config/args.json
cross_validation Conv_LSTM_8.pt Conv_LSTM_8

sed -i 's/"num_frames": 8/"num_frames": 9/g' config/args.json
sed -i 's/"output_frames": -8/"output_frames": -9/g' config/args.json
cross_validation Conv_LSTM_9.pt Conv_LSTM_9

sed -i 's/"num_frames": 9/"num_frames": 10/g' config/args.json
sed -i 's/"output_frames": -9/"output_frames": -10/g' config/args.json
cross_validation Conv_LSTM_10.pt Conv_LSTM_10

sed -i 's/"num_frames": 10/"num_frames": 5/g' config/args.json
sed -i 's/"output_frames": -10/"output_frames": -5/g' config/args.json
sed -i 's/get_model = ConvNext_LSTM/get_model = ConvNext_base3/g' scripts/main.py
