set -ex

export XACC=1
export XACC_ENABLE=1
#export BKCL_PCIE_RING=1
export BKCL_CCIX_RING=1
export BKCL_FORCE_SYNC=1
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XMLIR_D_XPU_L3_SIZE=66060288
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export XPUSIM_MFENCE_DEBUG=1
export XMLIR_D_FORCE_FALLBACK_STR="aten::_index_put_impl_,aten::index.Tensor"
#export XPUAPI_DEBUG=0x1
#export XPURT_DISPATCH_MODE=PROFILING
#export XTCL_DUMP_OP_ARGS=1
#export XDNN_LOG_FILE="xdnn.log"
#export XLOG_LEVEL="capture=info"
export XACC_ARGS="-L auto_tune"

#pip uninstall -y numpy || true
pip install psutil==5.9.5
pip install accelerate==0.20.3
pip install Pillow
pip install torchvision==0.13.1
pip install pycocotools==2.0.5
pip install --upgrade numpy==1.23.5

pip uninstall -y xmlir || true
pip install http://10.1.2.158:8111/flagperf/202307/retinanet/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
pip uninstall -y xacc || true
pip install http://10.1.2.158:8111/flagperf/202307/retinanet/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
pip show xmlir
pip show xacc
pip show numpy
python -m xacc.install


#md5sum xmlir
#ls /root/miniconda/envs/python38_torch1121/lib/python3.8/site-packages/xacc/plugins | grep auto

