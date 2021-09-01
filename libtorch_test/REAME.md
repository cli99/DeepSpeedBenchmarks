```sh
cmake -DCMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/python3.6/site-packages/torch .
```