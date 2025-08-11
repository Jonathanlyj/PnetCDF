#!/bin/bash

#SBATCH -A mxxx
#SBATCH -t 00:2:00
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH --qos=debug

#SBATCH -o qout_log/new/qout_read_4n_64.256.%j # std::out is saved here
#SBATCH -e qout_log/new/qout_read_4n_64.256.%j 

#SBATCH --mail-type=end,fail
#SBATCH --mail-user=xxx@xxxxxx

if test "x$SLURM_NTASKS_PER_NODE" = x ; then #number of cores per node
   SLURM_NTASKS_PER_NODE=64 # 256 max
fi

NUM_NODES=$SLURM_JOB_NUM_NODES

NP=$((NUM_NODES * SLURM_NTASKS_PER_NODE))

ulimit -c unlimited

EXE=/path/to/executable/new_format_read_test_all

SB_EXE=/tmp/${USER}_new_format_read_test_all #tmp all users shared
sbcast -v ${EXE} ${SB_EXE} #executable copy to this position

export PNETCDF_HINTS="nc_hash_size_dim=4096;nc_hash_size_var=4096"
srun -n $NP -c $((256/$SLURM_NTASKS_PER_NODE)) --cpu_bind=cores ${SB_EXE} /pscratch/xxx/xxx/dataset_568k_metadata_new_format_test_all.pnc
#srun python ...