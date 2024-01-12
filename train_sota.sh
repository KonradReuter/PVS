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

cross_validation DeepLab DeepLab.pt DeepLab

cross_validation SANet SANet.pt SANet

cross_validation TransFuse TransFuse.pt TransFuse

cross_validation PraNet PraNet.pt PraNet

cross_validation CASCADE CASCADE.pt CASCADE

cross_validation COSNet COSNet.pt COSNet

cross_validation HybridNet HybridNet.pt HybridNet

cross_validation PNSNet PNSNet.pt PNSNet

cross_validation PNSPlusNet PNSPlusNet.pt PNSPlusNet

cross_validation VACSNet VACSNet.pt VACSNet

