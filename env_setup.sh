conda create -n nesie python=3.8 -y
conda activate nesie 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.19.0 
pip install mmsegmentation==0.20.0  
conda install llvmlite==0.31.0
conda install -c conda-forge gxx=9.4
conda install -c conda-forge gxx_linux-64
pip install -v -e .  # or "python setup.py develop"
# pip install ninja
# conda install -c conda-forge blas openblas
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \
#     -v --no-deps \
#     --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
#     --install-option="--force_cuda" \
#     --install-option="--blas=openblas"
git clone https://github.com/lilanxiao/Rotated_IoU.git
cd Rotated_IoU
git checkout 3bdca6b20d981dffd773507e97f1b53641e98d0a
cd cuda_op
python setup.py install
cd ../..
cp -r Rotated_IoU/cuda_op mmdet3d/ops/rotated_iou
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl --user 
