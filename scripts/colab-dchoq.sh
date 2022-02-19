
#!/bin/bash
#Creating Folders
cd /content
echo "===> Creating folders..."
mkdir DATASETS/
mkdir DATASETS/ActionTubesV2
mkdir DATASETS/Pretrained_Models

#Update wget
pip install --upgrade --no-cache-dir gdown

######################################### RWF-2000 #########################################
############################################################################################
#RWF dataset
# echo "===> Preparing RWF-2000 dataset..."
# gdown --id 1sJFv-A-mbUFCcNflgXeCYeDNgGQixUXC
# mv /content/RWF-2000.zip /content/DATASETS
# unzip -q /content/DATASETS/RWF-2000.zip -d /content/DATASETS
# rm /content/DATASETS/RWF-2000.zip

#Action Tubes
# echo "===> Downloading RWF-2000 ActionTubes..."
# filename="RWF-2000.zip"
# gdown --id 13aVSz9j7XV4mqhcNAn0I7JNNteUYEV2A
# src="/content/${filename}"
# dst="/content/DATASETS/ActionTubesV2"
# mv $src dst
# f_name="${dst}/${filename}"
# unzip -q $f_name}-d $dst
# rm $f_name


######################################### RWF-2000 #########################################
############################################################################################
#Hockey Dataset
echo "===> Preparing Hockey dataset..."
gdown --id 1200VOpfMFUys_IrWWumI6qUP3-4-1U1G
mv  "/content/drive/My Drive/VIOLENCE DATA/HockeyFightsDATASET.zip" "/content/DATASETS"

##splits
gdown --id 1Rkff63JA16GmN0nv4VZzGcCrkJlFm0kA
# cp -r "/content/drive/My Drive/VIOLENCE DATA/VioNetDB-splits.zip" "/content/DATASETS"
# !unzip -q /content/DATASETS/VioNetDB-splits.zip -d /content/DATASETS/
# !rm /content/DATASETS/VioNetDB-splits.zip

#Action Tubes
# echo "===> Downloading Hockey ActionTubes..."
# filename="RWF-2000.zip"
# gdown --id 13aVSz9j7XV4mqhcNAn0I7JNNteUYEV2A
# src="/content/${filename}"
# dst="/content/DATASETS/ActionTubesV2"
# mv $src dst
# f_name="${dst}/${filename}"
# unzip -q $f_name}-d $dst
# rm $f_name

#Pretrained models

#Restore training
# id="1-RDg150Os9C4ZxynJOUBgUA6mmH8PQWs"
# echo "===> Downloading ${id}"
# gdown --id $id
# src="/content/$(ls -t | head -1)"
# dst="/content/DATASETS/Pretrained_Models"
# echo "===> Moving ${src} to ${dst}"
# mv $src ${dst}

# #Installing yacs
# echo "===> Installing yacs"
# pip3 install yacs

# #Installing CUDA
# echo "===> Installing yacs"
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html