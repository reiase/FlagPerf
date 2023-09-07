export XACC_ENABLE=1
export BKCL_PCIE_RING=1
export BKCL_FORCE_SYNC=1
export XMLIR_D_XPU_L3_SIZE=66060288
export XMLIR_D_FORCE_FALLBACK_STR="aten::_index_put_impl_,aten::index.Tensor"
export XACC_ARGS="-L auto_tune"

pip uninstall -y xmlir || true
pip install http://10.1.2.158:8111/flagperf/202307/retinanet/xmlir-0.0.1-cp38-cp38-linux_x86_64.whl
pip uninstall -y xacc || true
pip install http://10.1.2.158:8111/flagperf/202307/retinanet/xacc-0.1.0-cp38-cp38-linux_x86_64.whl
python -m xacc.install


