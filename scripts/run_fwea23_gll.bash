#!/bin/bash
#SBATCH --job-name=FWEA23-GLL
#SBATCH --partition=tyhcnormal
#SBATCH --account=acwub211ea
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=36
#SBATCH --time=05:00:00
#SBATCH --output=OUTPUT_FILES/job_%j.out
#SBATCH --error=OUTPUT_FILES/job_%j.err
#
# FWEA23 使用 GLL 格式模型的正演流程（NC→GLL→meshfem→solver）
# 用法: sbatch run_fwea23_gll.bash
# 需在项目根目录执行，或修改 PROJECT_ROOT

set -e

# 项目根目录：脚本在 specfem3d_globe/EXAMPLES/FWEA23，上级三级为项目根
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FWEA23="$SCRIPT_DIR"

cd $FWEA23

echo "======================================"
echo "FWEA23 GLL Workflow"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "FWEA23 dir:   $FWEA23"
echo "======================================"

# 加载环境（按服务器实际路径调整）
module purge
module load compiler/intel/2018.5.274
module load mpi/intelmpi/2018.4.274
module load mathlib/netcdf/4.4.1-intel-2017

NPROC=$(($(grep ^NCHUNKS DATA/Par_file | cut -d= -f2 | tr -d ' ') * \
        $(grep ^NPROC_XI DATA/Par_file | cut -d= -f2 | tr -d ' ') * \
        $(grep ^NPROC_ETA DATA/Par_file | cut -d= -f2 | tr -d ' ')))

echo "=== Step 1: EMC model (if needed) ==="
if [ ! -f ../../DATA/IRIS_EMC/model.nc ]; then
    python $PROJECT_ROOT/3_Data_space_simulation/convert_to_emc_nc.py --model 2024_FWEA23 --anisotropic
fi

echo "=== Step 2: First meshfem (EMC) ==="
mkdir -p OUTPUT_FILES DATABASES_MPI
cp DATA/Par_file OUTPUT_FILES/
grep -q "1D_transversely_isotropic_GLL" DATA/Par_file && \
  sed -i.bak 's/1D_transversely_isotropic_GLL/EMC_model/' DATA/Par_file || true
srun --mpi=pmi2 -n $NPROC ./bin/xmeshfem3D

echo "=== Step 3: NC to GLL ==="
python $PROJECT_ROOT/3_Data_space_simulation/convert_nc_to_gll.py \
    --mesh-dir $FWEA23/DATABASES_MPI \
    --model-nc $PROJECT_ROOT/data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    --output-dir $FWEA23/DATA/GLL \
    --anisotropic

echo "=== Step 4: Switch to GLL model ==="
sed -i 's/MODEL.*=.*EMC_model/MODEL                           = 1D_transversely_isotropic_GLL/' DATA/Par_file

echo "=== Step 5: Second meshfem + solver (GLL) ==="
rm -rf DATABASES_MPI/*
mkdir -p DATABASES_MPI
srun --mpi=pmi2 -n $NPROC ./bin/xmeshfem3D
srun --mpi=pmi2 -n $NPROC ./bin/xspecfem3D

echo "======================================"
echo "FWEA23 GLL Workflow DONE: $(date)"
echo "======================================"
