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

# base models

cross_validation Swin_base Swin_base.pt Swin_base

cross_validation Swin_base3 Swin_base3.pt Swin_base3

cross_validation Conv_base Conv_base.pt Conv_base

cross_validation Conv_base3 Conv_base3.pt Conv_base3


#simple models

cross_validation Conv_simple Conv_simple.pt Conv_simple

cross_validation Conv_simple_skip Conv_simple_skip.pt Conv_simple_skip

cross_validation Conv_simple_enc Conv_simple_enc.pt Conv_simple_enc

# 3D models

cross_validation Conv_3D Conv_3D.pt Conv_3D

cross_validation Conv_3D_skip Conv_3D_skip.pt Conv_3D_skip

cross_validation Conv_3D_enc Conv_3D_enc.pt Conv_3D_enc

# LSTM models

cross_validation Conv_LSTM Conv_LSTM.pt Conv_LSTM

cross_validation Conv_LSTM_skip Conv_LSTM_skip.pt Conv_LSTM_skip

cross_validation Conv_LSTM_enc Conv_LSTM_enc.pt Conv_LSTM_enc

# Attention models

cross_validation Conv_Attention Conv_Attention.pt Conv_Attention

cross_validation Conv_Attention_skip Conv_Attention_skip.pt Conv_Attention_skip

cross_validation Conv_Attention_enc Conv_Attention_enc.pt Conv_Attention_enc

# NSA models

cross_validation Conv_NSA Conv_NSA.pt Conv_NSA

cross_validation Conv_NSA_skip Conv_NSA_skip.pt Conv_NSA_skip

cross_validation Conv_NSA_enc Conv_NSA_enc.pt Conv_NSA_enc
