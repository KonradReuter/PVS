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

test_model(){
    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 0/"validation_fold": 1/g' config/args.json

    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 1/"validation_fold": 2/g' config/args.json

    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 2/"validation_fold": 3/g' config/args.json

    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 3/"validation_fold": 4/g' config/args.json

    python scripts/main.py evaluate --model-name=$1 --run-name=$2
    sed -i 's/"validation_fold": 4/"validation_fold": 0/g' config/args.json
}

# base models

sed -i 's/get_model = ConvNext_base3/get_model = PolypSwin_base/g' scripts/main.py
cross_validation Swin_base.pt Swin_base

sed -i 's/get_model = PolypSwin_base/get_model = PolypSwin_base3/g' scripts/main.py
cross_validation Swin_base3.pt Swin_base3

sed -i 's/get_model = PolypSwin_base3/get_model = ConvNext_base/g' scripts/main.py
cross_validation Conv_base.pt Conv_base

sed -i 's/get_model = ConvNext_base/get_model = ConvNext_base3/g' scripts/main.py
cross_validation Conv_base3.pt Conv_base3


#simple models

sed -i 's/get_model = ConvNext_base3/get_model = ConvNext_simple/g' scripts/main.py
cross_validation Conv_simple.pt Conv_simple

sed -i 's/get_model = ConvNext_simple/get_model = ConvNext_simple_skip/g' scripts/main.py
cross_validation Conv_simple_skip.pt Conv_simple_skip

sed -i 's/get_model = ConvNext_simple_skip/get_model = ConvNext_simple_enc/g' scripts/main.py
cross_validation Conv_simple_enc.pt Conv_simple_enc

# 3D models

sed -i 's/get_model = ConvNext_simple_enc/get_model = ConvNext_3D/g' scripts/main.py
cross_validation Conv_3D.pt Conv_3D

sed -i 's/get_model = ConvNext_3D/get_model = ConvNext_3D_skip/g' scripts/main.py
cross_validation Conv_3D_skip.pt Conv_3D_skip

sed -i 's/get_model = ConvNext_3D_skip/get_model = ConvNext_3D_enc/g' scripts/main.py
cross_validation Conv_3D_enc.pt Conv_3D_enc

# LSTM models

sed -i 's/get_model = ConvNext_3D_enc/get_model = ConvNext_LSTM/g' scripts/main.py
cross_validation Conv_LSTM.pt Conv_LSTM

sed -i 's/get_model = ConvNext_LSTM/get_model = ConvNext_LSTM_skip/g' scripts/main.py
cross_validation Conv_LSTM_skip.pt Conv_LSTM_skip

sed -i 's/get_model = ConvNext_LSTM_skip/get_model = ConvNext_LSTM_enc/g' scripts/main.py
cross_validation Conv_LSTM_enc.pt Conv_LSTM_enc

# Attention models

sed -i 's/get_model = ConvNext_LSTM_enc/get_model = ConvNext_Attention/g' scripts/main.py
cross_validation Conv_Attention.pt Conv_Attention

sed -i 's/get_model = ConvNext_Attention/get_model = ConvNext_Attention_skip/g' scripts/main.py
cross_validation Conv_Attention_skip.pt Conv_Attention_skip

sed -i 's/get_model = ConvNext_Attention_skip/get_model = ConvNext_Attention_enc/g' scripts/main.py
cross_validation Conv_Attention_enc.pt Conv_Attention_enc

# NSA models

sed -i 's/"amp": true/"amp": false/g' config/args.json
sed -i 's/get_model = ConvNext_Attention_enc/get_model = ConvNext_NSA/g' scripts/main.py
cross_validation Conv_NSA.pt Conv_NSA

sed -i 's/get_model = ConvNext_NSA/get_model = ConvNext_NSA_skip/g' scripts/main.py
cross_validation Conv_NSA_skip.pt Conv_NSA_skip

sed -i 's/get_model = ConvNext_NSA_skip/get_model = ConvNext_NSA_enc/g' scripts/main.py
cross_validation Conv_NSA_enc.pt Conv_NSA_enc
sed -i 's/"amp": false/"amp": true/g' config/args.json
sed -i 's/get_model = ConvNext_NSA_enc/get_model = ConvNext_base3/g' scripts/main.py
