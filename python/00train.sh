#!/bin/sh

do_command()
{
    echo $1
    $1
}

mkdir -p log
mkdir -p result


##~~~~~~~

BASENAME=classification 

OUTFILE=result/$BASENAME.model

if test x"$1" = x"--clean"; then
    rm $OUTFILE
fi

if test ! -f $OUTFILE; then
    do_command "python train.py --write-model $OUTFILE"
else
    echo "$OUTFILE is new"
fi

##~~~~~~~

INFILE=result/$BASENAME.model
OUTFILE=result/$BASENAME.onnx

if test $INFILE -nt $OUTFILE; then
    do_command "python test.py --read-model $INFILE --onnx $OUTFILE"
else
    echo "$OUTFILE is new"
fi

##~~~~~~~

INFILE=result/$BASENAME.onnx
OUTFILE=result/$BASENAME.xml

if test $INFILE -nt $OUTFILE; then

    source activate py37
    
    OPENVINO=openvino_2021
    
    export PYTHONPATH=/opt/intel/$OPENIVNO/opencv/lib/python3.7/site-packages:/opt/intel/$OPENIVNO/python/python3.7:/opt/intel/$OPENIVNO/python/python3:/opt/intel/$OPENIVNO/deployment_tools/model_optimizer:/opt/intel/$OPENIVNO/opencv/lib/python3.7/site-packages:/opt/intel/$OPENIVNO/python/python3.7:/opt/intel/$OPENIVNO/python/python3:/opt/intel/$OPENIVNO/deployment_tools/model_optimizer


    MO=/opt/intel/$OPENVINO/deployment_tools/model_optimizer/mo_onnx.py

    if test -x $MO; then
	$MO --input_model $INFILE --data_type FP16 --output_dir result
    fi
else
    echo "$OUTFILE is new"
fi

# end

