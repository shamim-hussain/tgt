#!/usr/bin/bash
config="$1"
port=12356

nodes=(`srun hostname | sort -u | xargs echo`)
nnodes="${#nodes[@]}"
master="${nodes[0]}"

echo '================================================================================'
echo "nodes = ${nodes[@]}"
echo "master = $master"
echo '================================================================================'

for i in ${!nodes[@]} ; do
    node="${nodes[$i]}"
    echo '================================================================================'
    echo $node
    echo '--------------------------------------------------------------------------------'
    echo 'srun -w' "$node" '-N1 torchrun --nproc_per_node=gpu --nnodes='"$nnodes" '--node_rank='"$i" '--master_addr='"$master" '--master_port='"$port" 'run_training.py' "$config" "${@:2}" "&"
    srun -w "$node" -N1 torchrun --nproc_per_node=gpu --nnodes="$nnodes" --node_rank="$i" --master_addr="$master" --master_port="$port" run_training.py "$config"  ${@:2} &
    echo '================================================================================'
done

wait
