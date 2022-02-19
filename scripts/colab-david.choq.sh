
#!/bin/bash
#Creating Folders
cd /content
echo "===> Creating folders..."
mkdir DATASETS/
mkdir DATASETS/ActionTubesV2
mkdir DATASETS/Pretrained_Models

#RWF dataset
echo "===> Preparing RWF-2000 dataset..."
cp  "/content/drive/My Drive/VIOLENCE DATA/RWF-2000.zip" "/content/DATASETS"
unzip -q /content/DATASETS/RWF-2000.zip -d /content/DATASETS
rm /content/DATASETS/RWF-2000.zip

#Action Tubes
echo "===> Downloading RWF-2000 ActionTubes..."
filename="RWF-2000.zip"
src="/content/drive/My Drive/VIOLENCE DATA/Tubes/ActionTubesV2/${filename}"
dst="/content/DATASETS/ActionTubesV2"
f_name="${dst}/${filename}"
cp -r "$src" $dst
unzip -q $f_name -d $dst
rm $f_name

#Update wget
pip install --upgrade --no-cache-dir gdown

#Pretrained models

#Restore training
id="1-RDg150Os9C4ZxynJOUBgUA6mmH8PQWs"
echo "===> Downloading ${id}"
gdown --id $id
src="/content/$(ls -t | head -1)"
dst="/content/DATASETS/Pretrained_Models"
echo "===> Moving ${src} to ${dst}"
mv $src ${dst}

#Installing yacs
echo "===> Installing yacs"
pip3 install yacs

#Installing CUDA
echo "===> Installing yacs"
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html