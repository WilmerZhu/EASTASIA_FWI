#!/bin/bash

echo "======================================"
echo "运行 FWEA23 - Chujie 修改版 SPECFEM3D Globe 7.0"
echo "======================================"
echo "运行时间: $(date)"
echo "工作目录: $(pwd)"

# ========================================
# 【重要】请根据您的服务器路径修改以下变量
# ========================================
# Chujie版本代码的根目录（包含 src/, setup/, DATA/ 等）
CHUJIE_CODE_DIR="/work/home/acf11bgjob/specfem/specfem3d_globe_code_new"

# Chujie提供的配置文件目录（包含 DATA/ 和 setup/ 子目录）
CHUJIE_CONFIG_DIR="/work/home/acf11bgjob/specfem/sem_config_tibet_v4"

# 当前工作目录（通常是您的EXAMPLES目录）
currentdir=$(pwd)

# GLL模型文件的位置（包含 proc*_reg1_*.bin 文件）
# 如果有多个位置，脚本会自动搜索
GLL_MODEL_DIR="/work/home/acf11bgjob/specfem/specfem3d_globe_code_new/DATA/GLL"

# 全局数据文件目录（包含 s362ani, crust1.0, topo_bathy 等）
GLOBAL_DATA_DIR="/work/home/acf11bgjob/specfem/specfem3d_globe_code_new/DATA"

echo ""
echo "======================================"
echo "配置路径："
echo "  Chujie代码: $CHUJIE_CODE_DIR"
echo "  Chujie配置: $CHUJIE_CONFIG_DIR"
echo "  GLL模型: $GLL_MODEL_DIR"
echo "  全局数据: $GLOBAL_DATA_DIR"
echo "======================================"

# 验证路径存在
if [ ! -d "$CHUJIE_CODE_DIR" ]; then
    echo "❌ Chujie代码目录不存在: $CHUJIE_CODE_DIR"
    echo "请修改脚本中的 CHUJIE_CODE_DIR 变量"
    exit 1
fi

if [ ! -d "$CHUJIE_CONFIG_DIR" ]; then
    echo "❌ Chujie配置目录不存在: $CHUJIE_CONFIG_DIR"
    echo "请修改脚本中的 CHUJIE_CONFIG_DIR 变量"
    exit 1
fi

echo ""
echo "======================================"
echo "设置模拟环境"
echo "======================================"

# 创建工作目录
WORK_DIR="$currentdir/FWEA23_chujie_version"
mkdir -p "$WORK_DIR"/{DATABASES_MPI,OUTPUT_FILES,DATA,bin}

cd "$WORK_DIR"

# 清理旧数据
rm -rf DATABASES_MPI/*
rm -rf OUTPUT_FILES/*

echo "✅ 工作目录创建: $WORK_DIR"

# 复制Chujie的配置文件
echo ""
echo "======================================"
echo "复制配置文件"
echo "======================================"

# 复制 Par_file
if [ -f "$CHUJIE_CONFIG_DIR/DATA/Par_file" ]; then
    cp "$CHUJIE_CONFIG_DIR/DATA/Par_file" DATA/
    echo "✅ Par_file 复制成功"
    
    # 显示关键参数
    echo ""
    echo "Par_file 关键参数:"
    grep "^MODEL" DATA/Par_file | head -1
    grep "^ATTENUATION" DATA/Par_file | head -1
    grep "^NEX_XI" DATA/Par_file
    grep "^NEX_ETA" DATA/Par_file
    grep "^NPROC_XI" DATA/Par_file
    grep "^NPROC_ETA" DATA/Par_file
else
    echo "❌ Par_file 不存在: $CHUJIE_CONFIG_DIR/DATA/Par_file"
    exit 1
fi

# 复制 STATIONS
if [ -f "$CHUJIE_CONFIG_DIR/DATA/STATIONS" ]; then
    cp "$CHUJIE_CONFIG_DIR/DATA/STATIONS" DATA/
    station_count=$(grep -v "^#" DATA/STATIONS | wc -l)
    echo "✅ STATIONS 复制成功 (台站数: $station_count)"
else
    echo "❌ STATIONS 文件不存在"
    exit 1
fi

# 复制 CMTSOLUTION
if [ -f "$CHUJIE_CONFIG_DIR/DATA/CMTSOLUTION" ]; then
    cp "$CHUJIE_CONFIG_DIR/DATA/CMTSOLUTION" DATA/
    echo "✅ CMTSOLUTION 复制成功"
    echo ""
    echo "CMTSOLUTION 前7行:"
    head -7 DATA/CMTSOLUTION
else
    echo "❌ CMTSOLUTION 文件不存在"
    exit 1
fi

echo ""
echo "======================================"
echo "编译 Chujie 版本代码"
echo "======================================"

cd "$CHUJIE_CODE_DIR"

# 清理模块环境
echo "清理并加载编译环境..."
module purge
module load compiler/intel/2018.5.274
module load mpi/intelmpi/2018.4.274

echo ""
echo "编译环境信息："
module list
echo ""
which ifort
which mpiifort
ifort --version | head -1

# 【关键】使用Chujie的 constants.h.in
echo ""
echo "======================================"
echo "配置 constants.h.in"
echo "======================================"

if [ -f "$CHUJIE_CONFIG_DIR/setup/constants.h.in" ]; then
    echo "使用 Chujie 的 constants.h.in..."
    
    # 备份原始文件
    if [ ! -f "setup/constants.h.in.original_backup" ]; then
        cp setup/constants.h.in setup/constants.h.in.original_backup
        echo "✅ 原始文件已备份"
    fi
    
    # 复制Chujie的配置
    cp "$CHUJIE_CONFIG_DIR/setup/constants.h.in" setup/constants.h.in
    
    echo "✅ constants.h.in 替换成功"
    
    # 显示关键配置
    echo ""
    echo "关键配置参数:"
    grep "USE_ECEF_CMTSOLUTION" setup/constants.h.in | grep -v "^!" | head -1
    grep "ATTENUATION_1D_WITH_3D_STORAGE" setup/constants.h.in | grep -v "^!" | head -1
    grep "REGIONAL_MOHO_MESH" setup/constants.h.in | grep -v "^!" | head -3
    grep "GLL_REFERENCE_1D_MODEL" setup/constants.h.in | grep -v "^!" | head -1
    grep "RMOHO_STRETCH_ADJUSTMENT" setup/constants.h.in | grep -v "^!" | head -1
else
    echo "⚠️  未找到 Chujie 的 constants.h.in，使用默认配置"
fi


# 验证修改
echo "当前ECEF设置:"
grep "USE_ECEF_CMTSOLUTION" setup/constants.h.in | grep -v "^!" | head -1

# 复制Par_file到编译目录的DATA
cp "$WORK_DIR/DATA/Par_file" DATA/
cp "$WORK_DIR/DATA/CMTSOLUTION" DATA/
cp "$WORK_DIR/DATA/STATIONS" DATA/

echo ""
echo "======================================"
echo "运行 configure"
echo "======================================"

./configure FC=ifort CC=icc CXX=icpc MPIFC=mpiifort CUSTOM_REAL=SIZE_REAL

if [ $? -ne 0 ]; then
    echo "❌ configure 失败"
    exit 1
fi

echo "✅ configure 成功"

echo ""
echo "======================================"
echo "编译代码"
echo "======================================"

make clean

# 只编译核心程序，跳过辅助工具
echo "编译核心程序 (xmeshfem3D 和 xspecfem3D)..."
make -j8 xmeshfem3D
if [ $? -ne 0 ]; then
    echo "❌ xmeshfem3D 编译失败"
    exit 1
fi

make -j8 xspecfem3D
if [ $? -ne 0 ]; then
    echo "❌ xspecfem3D 编译失败"
    exit 1
fi

echo "✅ 核心程序编译成功"
# 验证可执行文件
echo ""
echo "验证可执行文件:"
for exe in bin/xmeshfem3D bin/xspecfem3D; do
    if [ ! -f "$exe" ]; then
        echo "❌ $exe 未生成"
        exit 1
    else
        size=$(ls -lh "$exe" | awk '{print $5}')
        echo "✅ $exe (大小: $size)"
    fi
done

# 复制可执行文件到工作目录
cp bin/xmeshfem3D "$WORK_DIR/bin/"
cp bin/xspecfem3D "$WORK_DIR/bin/"

# 备份编译配置
cp setup/* "$WORK_DIR/OUTPUT_FILES/" 2>/dev/null || true
cp DATA/Par_file "$WORK_DIR/OUTPUT_FILES/"

echo ""
echo "======================================"
echo "准备数据文件"
echo "======================================"

cd "$WORK_DIR"

# 链接全局数据目录
cd DATA/

echo "链接全局数据..."
# 优先使用Chujie代码目录中的数据，然后是全局数据目录
for data_dir in crust1.0 s362ani QRFSI12 topo_bathy; do
    if [ -d "$CHUJIE_CODE_DIR/DATA/$data_dir" ]; then
        ln -sf "$CHUJIE_CODE_DIR/DATA/$data_dir" .
        echo "✅ $data_dir -> $CHUJIE_CODE_DIR/DATA/$data_dir"
    elif [ -d "$GLOBAL_DATA_DIR/$data_dir" ]; then
        ln -sf "$GLOBAL_DATA_DIR/$data_dir" .
        echo "✅ $data_dir -> $GLOBAL_DATA_DIR/$data_dir"
    else
        echo "⚠️  $data_dir 未找到"
    fi
done

cd ..

# 【关键】链接或复制GLL模型文件
echo ""
echo "======================================"
echo "准备 GLL 模型文件"
echo "======================================"

# 检查多个可能的GLL模型位置
GLL_SOURCE=""

# 搜索顺序：1.指定的GLL_MODEL_DIR 2.当前目录 3.Chujie代码目录 4.Chujie配置目录
for search_dir in "$GLL_MODEL_DIR" "$currentdir/DATA/GLL" "$CHUJIE_CODE_DIR/DATA/GLL" "$CHUJIE_CONFIG_DIR/DATA/GLL"; do
    if [ -d "$search_dir" ] && [ $(ls "$search_dir"/proc*_reg1_*.bin 2>/dev/null | wc -l) -gt 0 ]; then
        GLL_SOURCE="$search_dir"
        break
    fi
done

if [ -n "$GLL_SOURCE" ]; then
    echo "✅ 找到GLL模型文件: $GLL_SOURCE"
    ln -sf "$GLL_SOURCE" DATA/GLL
    
    gll_count=$(ls DATA/GLL/proc*_reg1_*.bin 2>/dev/null | wc -l)
    echo "GLL模型文件总数: $gll_count"
    
    # 显示示例文件
    echo "示例文件 (proc000000):"
    ls DATA/GLL/proc000000_reg1_*.bin 2>/dev/null | while read f; do
        size=$(ls -lh "$f" | awk '{print $5}')
        echo "  $(basename $f) - $size"
    done
    
    # 检查进程10的文件（之前出过问题的进程）
    proc10_count=$(ls DATA/GLL/proc000010_reg1_*.bin 2>/dev/null | wc -l)
    if [ $proc10_count -gt 0 ]; then
        echo "✅ 进程10的GLL文件: $proc10_count 个"
    fi
else
    echo "❌ 未找到GLL模型文件"
    echo ""
    echo "请检查以下位置之一是否包含 proc*_reg1_*.bin 文件:"
    echo "  1. $GLL_MODEL_DIR"
    echo "  2. $currentdir/DATA/GLL/"
    echo "  3. $CHUJIE_CODE_DIR/DATA/GLL/"
    echo "  4. $CHUJIE_CONFIG_DIR/DATA/GLL/"
    echo ""
    echo "或修改脚本中的 GLL_MODEL_DIR 变量指向正确位置"
    exit 1
fi

# 验证关键数据文件
echo ""
echo "======================================"
echo "验证数据文件完整性"
echo "======================================"

# 验证S362ANI参考模型
if [ -f "DATA/s362ani/S362ANI" ]; then
    s362_size=$(ls -lh DATA/s362ani/S362ANI | awk '{print $5}')
    echo "✅ S362ANI 参考模型 ($s362_size)"
else
    echo "⚠️  S362ANI 参考模型未找到"
fi

# 验证地壳模型
if [ -d "DATA/crust1.0" ]; then
    crust_files=$(ls DATA/crust1.0/ 2>/dev/null | wc -l)
    echo "✅ crust1.0 地壳模型 ($crust_files 个文件)"
else
    echo "⚠️  crust1.0 地壳模型未找到"
fi

# 验证地形数据
if [ -f "DATA/topo_bathy/topo_bathy_etopo4_smoothed_window_7.bin" ]; then
    topo_size=$(ls -lh DATA/topo_bathy/topo_bathy_etopo4_smoothed_window_7.bin | awk '{print $5}')
    echo "✅ 地形/海深数据 ($topo_size)"
else
    echo "⚠️  地形数据未找到"
fi

echo ""
echo "======================================"
echo "创建 SLURM 作业脚本"
echo "======================================"

cat > go_mesher_solver_chujie.bash << 'SLURM_SCRIPT'
#!/bin/bash
#SBATCH --job-name=FWEA23_chujie
#SBATCH --partition=tyhcnormal
#SBATCH --account=acwub211ea 
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=48
#SBATCH --constraint=64core
#SBATCH --time=24:00:00
#SBATCH --output=OUTPUT_FILES/job_%j.out
#SBATCH --error=OUTPUT_FILES/job_%j.err

echo "======================================"
echo "FWEA23 - Chujie 修改版 SPECFEM3D Globe 7.0"
echo "======================================"
echo "作业开始: $(date)"
echo "工作目录: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "分配节点: $SLURM_JOB_NUM_NODES ($SLURM_NNODES)"
echo "节点列表: $SLURM_JOB_NODELIST"
echo "每节点任务数: $SLURM_NTASKS_PER_NODE"
echo "======================================"

# 设置栈空间为无限
echo ""
echo "设置系统资源..."
ulimit -s unlimited
ulimit -l unlimited
echo "✅ Stack size: $(ulimit -s)"
echo "✅ Max locked memory: $(ulimit -l)"

# 加载编译环境
echo ""
echo "加载编译环境..."
module purge
module load compiler/intel/2018.5.274
module load mpi/intelmpi/2018.4.274

# Intel Fortran运行时环境变量
export KMP_STACKSIZE=512m
export OMP_STACKSIZE=512m
export OMP_NUM_THREADS=1

# Intel MPI配置
export I_MPI_FABRICS=shm:dapl
export I_MPI_FALLBACK=0

# Intel Fortran设置
export FOR_IGNORE_EXCEPTIONS=true
export KMP_AFFINITY=compact

echo ""
echo "======================================"
echo "运行时环境配置"
echo "======================================"
module list
echo ""
echo "Intel编译器:"
which ifort
which mpiifort
echo ""
echo "环境变量:"
echo "  KMP_STACKSIZE: $KMP_STACKSIZE"
echo "  OMP_STACKSIZE: $OMP_STACKSIZE"
echo "  I_MPI_FABRICS: $I_MPI_FABRICS"
echo "======================================"

cd $SLURM_SUBMIT_DIR

# 验证必要文件
echo ""
echo "验证必要文件..."
for file in DATA/Par_file DATA/STATIONS DATA/CMTSOLUTION; do
    if [ ! -f "$file" ]; then
        echo "❌ $file 缺失"
        exit 1
    fi
    echo "✅ $file"
done

for exe in bin/xmeshfem3D bin/xspecfem3D; do
    if [ ! -f "$exe" ]; then
        echo "❌ $exe 缺失"
        exit 1
    fi
    size=$(ls -lh "$exe" | awk '{print $5}')
    echo "✅ $exe ($size)"
done

# 验证GLL模型文件
echo ""
echo "验证GLL模型文件..."
gll_count=$(ls DATA/GLL/proc*_reg1_*.bin 2>/dev/null | wc -l)
echo "GLL模型文件数: $gll_count"

if [ $gll_count -eq 0 ]; then
    echo "❌ GLL模型文件缺失"
    exit 1
fi

# 显示进程0和进程10的文件（关键进程）
for proc in 000000 000010; do
    proc_files=$(ls DATA/GLL/proc${proc}_reg1_*.bin 2>/dev/null | wc -l)
    if [ $proc_files -gt 0 ]; then
        echo "✅ 进程 $proc: $proc_files 个文件"
    else
        echo "⚠️  进程 $proc: 无文件"
    fi
done

# 读取Par_file参数
BASEMPIDIR=$(grep ^LOCAL_PATH DATA/Par_file | cut -d = -f 2 | tr -d ' ')
NPROC_XI=$(grep ^NPROC_XI DATA/Par_file | cut -d = -f 2 | tr -d ' ')
NPROC_ETA=$(grep ^NPROC_ETA DATA/Par_file | cut -d = -f 2 | tr -d ' ')
NCHUNKS=$(grep ^NCHUNKS DATA/Par_file | cut -d = -f 2 | tr -d ' ')

numnodes=$((NCHUNKS * NPROC_XI * NPROC_ETA))

echo ""
echo "======================================"
echo "计算配置"
echo "======================================"
echo "NPROC_XI: $NPROC_XI"
echo "NPROC_ETA: $NPROC_ETA"
echo "NCHUNKS: $NCHUNKS"
echo "总核心数: $numnodes"
echo "BASEMPIDIR: $BASEMPIDIR"
echo "======================================"

# 验证资源分配
total_tasks=$((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))
echo "SLURM分配任务数: $total_tasks"

if [ $total_tasks -ne $numnodes ]; then
    echo "⚠️  警告: SLURM任务数 ($total_tasks) 与 Par_file 要求 ($numnodes) 不匹配"
    if [ $total_tasks -lt $numnodes ]; then
        echo "❌ 计算资源不足"
        exit 1
    fi
fi

mkdir -p OUTPUT_FILES "$BASEMPIDIR"
cp DATA/Par_file OUTPUT_FILES/
cp DATA/STATIONS OUTPUT_FILES/
cp DATA/CMTSOLUTION OUTPUT_FILES/

##
## 阶段 1: 网格生成
##
echo ""
echo "======================================"
echo "阶段 1: 网格生成 (xmeshfem3D)"
echo "======================================"
echo "开始时间: $(date)"
echo "MPI命令: mpirun -np $numnodes ./bin/xmeshfem3D"
echo "======================================"

mesh_start=$(date +%s)

mpirun -np $numnodes ./bin/xmeshfem3D
MESH_STATUS=$?

mesh_end=$(date +%s)
mesh_time=$((mesh_end - mesh_start))

echo ""
echo "网格生成完成"
echo "退出码: $MESH_STATUS"
echo "耗时: $mesh_time 秒 ($((mesh_time / 60)) 分钟)"

# 检查网格生成结果
echo ""
echo "检查网格生成结果..."
proc_files=$(ls $BASEMPIDIR/proc*_reg1_solver_data.bin 2>/dev/null | wc -l)
echo "生成的进程数据文件: $proc_files"

total_db_files=$(ls $BASEMPIDIR/ 2>/dev/null | wc -l)
echo "数据库总文件数: $total_db_files"

db_size=$(du -sh $BASEMPIDIR 2>/dev/null | awk '{print $1}')
echo "数据库大小: $db_size"

if [ $MESH_STATUS -ne 0 ] || [ $proc_files -eq 0 ]; then
    echo ""
    echo "❌ 网格生成失败"
    echo "请检查: OUTPUT_FILES/output_mesher.txt"
    exit 1
fi

echo "✅ 网格生成成功: $(date)"

##
## 阶段 2: 正演模拟
##
echo ""
echo "======================================"
echo "阶段 2: 正演模拟 (xspecfem3D)"
echo "======================================"
echo "开始时间: $(date)"
echo "MPI命令: mpirun -np $numnodes ./bin/xspecfem3D"
echo "======================================"

solver_start=$(date +%s)

mpirun -np $numnodes ./bin/xspecfem3D
SOLVER_STATUS=$?

solver_end=$(date +%s)
solver_time=$((solver_end - solver_start))

echo ""
echo "正演模拟完成"
echo "退出码: $SOLVER_STATUS"
echo "耗时: $solver_time 秒 ($((solver_time / 60)) 分钟)"

if [ $SOLVER_STATUS -eq 0 ]; then
    echo ""
    echo "✅ 正演模拟成功: $(date)"
    echo "$(date)" > OUTPUT_FILES/simulation_end_time
    
    # 统计输出文件
    echo ""
    echo "输出文件统计:"
    sac_files=$(ls OUTPUT_FILES/*.sac 2>/dev/null | wc -l)
    ascii_files=$(ls OUTPUT_FILES/*.ascii 2>/dev/null | wc -l)
    
    echo "  SAC地震图: $sac_files 个"
    echo "  ASCII地震图: $ascii_files 个"
    
    if [ $sac_files -gt 0 ]; then
        echo ""
        echo "示例SAC文件 (前5个):"
        ls OUTPUT_FILES/*.sac 2>/dev/null | head -5 | while read f; do
            size=$(ls -lh "$f" | awk '{print $5}')
            echo "  $(basename $f) - $size"
        done
    fi
    
    if [ $sac_files -eq 0 ] && [ $ascii_files -eq 0 ]; then
        echo ""
        echo "⚠️  未找到地震图文件"
        echo "请检查 STATIONS 文件和 Par_file 中的输出设置"
    fi
    
    output_size=$(du -sh OUTPUT_FILES 2>/dev/null | awk '{print $1}')
    echo ""
    echo "输出目录大小: $output_size"
    
else
    echo ""
    echo "❌ 正演模拟失败 (退出码: $SOLVER_STATUS)"
    echo ""
    echo "调试建议:"
    echo "1. 检查 OUTPUT_FILES/output_solver.txt"
    echo "2. 检查 OUTPUT_FILES/job_${SLURM_JOB_ID}.err"
    echo "3. 验证 GLL 模型文件与 Par_file 中 NPROC 配置是否匹配"
    echo "4. 检查内存和栈空间设置"
    exit 1
fi

echo ""
echo "======================================"
echo "模拟完成 - Chujie 版本"
echo "======================================"
echo "总耗时: $(((mesh_time + solver_time) / 60)) 分钟"
echo "  网格生成: $((mesh_time / 60)) 分钟"
echo "  正演模拟: $((solver_time / 60)) 分钟"
echo ""
echo "结果位置:"
echo "  地震图: OUTPUT_FILES/"
echo "  数据库: $BASEMPIDIR"
echo ""
echo "完成时间: $(date)"
echo "======================================"
SLURM_SCRIPT

chmod +x go_mesher_solver_chujie.bash

echo "✅ SLURM作业脚本创建完成"

echo ""
echo "======================================"
echo "✅ 准备就绪"
echo "======================================"
echo ""
echo "工作目录: $WORK_DIR"
echo ""
echo "下一步操作:"
echo "  1. 进入工作目录:"
echo "     cd $WORK_DIR"
echo ""
echo "  2. 提交SLURM作业:"
echo "     sbatch go_mesher_solver_chujie.bash"
echo ""
echo "  3. 监控作业状态:"
echo "     squeue -u $(whoami)"
echo "     tail -f OUTPUT_FILES/job_XXXXX.out"
echo ""
echo "  4. 查看结果:"
echo "     ls -lh OUTPUT_FILES/*.sac"
echo ""
echo "======================================"
echo "脚本完成: $(date)"
echo "======================================"