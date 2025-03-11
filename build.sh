# Setup Conda environment
curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p ~/conda &&\
    rm ~/miniconda.sh

~/conda/bin/conda init bash
source ~/.bashrc

# install dependencies
conda env create -f environment.yml
export TORCH_CUDA_ARCH="8.9"
python -m pip install -r requirements.txt

# install acronym tools 
git clone https://github.com/RPFey/acronym.git ~/acronym_tools
cd ~/acronym_tools
python -m pip install -e .
