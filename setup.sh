# setup script
PATH=$PATH:$PWD/miniconda/bin
curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8 \
 && conda clean -ya

# CUDA 11.1-specific steps
conda install -y -c pytorch \
    cudatoolkit=11 \
    "pytorch" \
    "torchvision" \
 && conda clean -ya
 
pip install -r requirements.txt
pip install -U albumentations[imgaug]
