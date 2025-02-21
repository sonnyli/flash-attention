#!/usr/bin/env bash

pip uninstall flash_attn_v2 -y
rm -rf build
FLASH_ATTN_CUDA_ARCHS="80" python setup.py install | cu++filt -p