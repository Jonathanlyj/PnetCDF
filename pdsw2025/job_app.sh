#!/bin/bash

#SBATCH -A mxxx
#SBATCH -t 00:5:00
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH --qos=debug

#SBATCH -o qout_log/app/qout_4n_64.256.%j # std::out is saved here
#SBATCH -e qout_log/app/qout_4n_64.256.%j 

#SBATCH --mail-type=end,fail
#SBATCH --mail-user=xxx@xxx.xx

if test "x$SLURM_NTASKS_PER_NODE" = x ; then 
   SLURM_NTASKS_PER_NODE=64 # 256 max
fi

NUM_NODES=$SLURM_JOB_NUM_NODES

NP=$((NUM_NODES * SLURM_NTASKS_PER_NODE))

ulimit -c unlimited


EXE=/path/to/executable/app_baseline_test_all

SB_EXE=/tmp/${USER}_app_baseline_test_all #tmp all users shared
sbcast -v ${EXE} ${SB_EXE} #executable copy to this position

rm -f /pscratch/sd/xxxxx/FS_2M_8/dataset_568k_metadata_app_baseline_test_all.nc
export PNETCDF_HINTS="nc_hash_size_dim=16384;nc_hash_size_var=16384"
srun -n $NP -c $((256/$SLURM_NTASKS_PER_NODE)) --cpu_bind=cores ${SB_EXE} data/dataset_568k_metadata /pscratch/sd/xxxxx/FS_2M_8/dataset_568k_metadata_app_baseline_test_all.nc

#srun python ...