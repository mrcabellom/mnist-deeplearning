# !/bin/bash
conda-env create --file ../anaconda_env.yml
if [ $? -ne 0 ]
then
    echo 'Hubo un error creando el entorno virtual';
else
    source activate cntk22dotnetmalaga
fi
