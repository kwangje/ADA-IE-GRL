### pyenv for fish shell
eval "$(pyenv init -)" 
pyenv shell ada

### install appropriate torch_ver
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

### training
python3 train.py yaml/ADA_IE.yaml --data_parallel_backend

