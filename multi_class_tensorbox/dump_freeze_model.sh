#!/bin/bash
echo hype file: $1
echo model: $2

hypes_input=$1
model_input=$2
model_output_dir=`dirname $2`
model_filename=`basename $2`

output_base_dir=`dirname $model_output_dir`
model_name_prefix=`basename $model_output_dir`

model_version=`echo $model_filename | tr '\n' ' ' | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ''`
new_pb_file_path=${output_base_dir}/${model_name_prefix}_${model_version}.pb
new_pb_checkpoint_file_path=${new_pb_file_path}.checkpoint.data

frozen_pb_file_path=${output_base_dir}/frozen_${model_name_prefix}_${model_version}.pb


echo $new_pb_file_path
echo $new_pb_checkpoint_file_path
echo $frozen_pb_file_path

echo Dumping model...
python dump_model.py --hypes=${hypes_input} --model=${model_input} --out_graph=${new_pb_file_path} --out_checkpoint=${new_pb_checkpoint_file_path}

echo Model dumped at ${new_pb_file_path}, checkpoint file at ${new_pb_checkpoint_file_path}

echo Freezing model...
cd ../tensorflow
echo Building tensorflow free_graph tool...
bazel build tensorflow/python/tools:freeze_graph
echo Freezing model...
bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=${new_pb_file_path} --input_checkpoint=${new_pb_checkpoint_file_path} --output_graph=${frozen_pb_file_path} --input_binary=True --output_node_names=add,Reshape_2
echo Finished freezing model. Model located at ${frozen_pb_file_path}
