# INSTALLATION

* Create environment:
```bash
conda env create -n 3d_thuync python=3.7
conda activate 3d_thuync
```

* Install required packages:
```bash
conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.2 -c pytorch
pip install -r requirements.txt
pip install ipython jupyterlab opencv-python mmcv pylint
ipython kernel install --name "3d_thuync" --user
```

* Install spconv:
```bash
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv/
git reset --hard 8da6f967fb9a054d8870c3515b1b44eca2103634
python setup.py bdist_wheel
cd dist/
pip install spconv-1.0-cp37-cp37m-linux_x86_64.whl
cd ../../
```

* Install OpenPCDet:
```bash
python setup.py develop
```

* Setup data:
```bash
cd data/kitti/
ln -s /data/kitti/training/ ./
ln -s /data/kitti/testing/ ./
cd ../../
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
