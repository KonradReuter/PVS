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

sed -i 's/get_model = ConvNext_base3/get_model = get_DeepLab/g' scripts/main.py
cross_validation DeepLab.pt DeepLab
sed -i 's/get_model = get_DeepLab/get_model = ConvNext_base3/g' scripts/main.py

sed -i 's/get_model = get_ResUNetPP/get_model = get_SANet/g' scripts/main.py
cross_validation SANet.pt SANet

sed -i 's/"loss_factors": \[1.0\]/"loss_factors": \[0.2, 0.3, 0.5\]/g' config/args.json
sed -i 's/get_model = get_SANet/get_model = get_TransFuse/g' scripts/main.py
cross_validation TransFuse.pt TransFuse

sed -i 's/"loss_factors": \[0.2, 0.3, 0.5\]/"loss_factors": \[1.0, 1.0, 1.0, 1.0\]/g' config/args.json
sed -i 's/get_model = get_TransFuse/get_model = get_PraNet/g' scripts/main.py
cross_validation PraNet.pt PraNet

sed -i 's/get_model = get_PraNet/get_model = get_CASCADE/g' scripts/main.py
cross_validation CASCADE.pt CASCADE
sed -i 's/"loss_factors": \[1.0, 1.0, 1.0, 1.0\]/"loss_factors": \[1.0\]/g' config/args.json


sed -i 's/"num_frames": 5/"num_frames": 2/g' config/args.json
sed -i 's/"output_frames": -5/"output_frames": -2/g' config/args.json
sed -i 's/"time_interval": 1/"time_interval": 4/g' config/args.json
sed -i 's/get_model = get_CASCADE/get_model = CoattentionNet/g' scripts/main.py
cross_validation COSNet.pt COSNet
sed -i 's/"num_frames": 2/"num_frames": 5/g' config/args.json
sed -i 's/"output_frames": -2/"output_frames": -5/g' config/args.json
sed -i 's/"time_interval": 4/"time_interval": 1/g' config/args.json

sed -i 's/"loss_factors": \[1.0\]/"loss_factors": \[1.0, 1.0\]/g' config/args.json
sed -i 's/"output_frames": -5/"output_frames": 2/g' config/args.json
sed -i 's/"unique": true/"unique": false/g' config/args.json
sed -i 's/get_model = CoattentionNet/get_model = get_HybridNet/g' scripts/main.py
cross_validation HybridNet.pt HybridNet
sed -i 's/"output_frames": 2/"output_frames": -5/g' config/args.json
sed -i 's/"unique": false/"unique": true/g' config/args.json
sed -i 's/"loss_factors": \[1.0, 1.0\]/"loss_factors": \[1.0\]/g' config/args.json

sed -i 's/"amp": true/"amp": false/g' config/args.json
sed -i 's/get_model = get_HybridNet/get_model = PNSNet/g' scripts/main.py
cross_validation PNSNet.pt PNSNet

sed -i 's/"num_frames": 5/"num_frames": 6/g' config/args.json
sed -i 's/"anchor_frame": false/"anchor_frame": true/g' config/args.json
sed -i 's/get_model = PNSNet/get_model = PNSPlusNet/g' scripts/main.py
cross_validation PNSPlusNet.pt PNSPlusNet
sed -i 's/"amp": false/"amp": true/g' config/args.json
sed -i 's/"num_frames": 6/"num_frames": 5/g' config/args.json
sed -i 's/"anchor_frame": true/"anchor_frame": false/g' config/args.json

sed -i 's/"loss_factors": \[1.0\]/"loss_factors": \[0.2, 0.2, 0.2, 0.2, 0.2\]/g' config/args.json
sed -i 's/get_model = PNSPlusNet/get_model = VACSNet/g' scripts/main.py
cross_validation VACSNet.pt VACSNet
sed -i 's/"loss_factors": \[0.2, 0.2, 0.2, 0.2, 0.2\]/"loss_factors": \[1.0\]/g' config/args.json
sed -i 's/get_model = VACSNet/get_model = ConvNext_base3/g' scripts/main.py
