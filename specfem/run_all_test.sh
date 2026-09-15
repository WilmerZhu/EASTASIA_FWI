#!/bin/bash
# ============================================================================
# EASTASIA-FWI  多震源多模型正演测试（全流程服务器端）
# ============================================================================
#
# 五个模型组 × 30 个震源（EastAsia_model 标准测试库），外加一个单事件参照组。
# 后四个模型组共用同一套 CRUST1.0
# 地壳，只在 Moho 以下的地幔不同，因此组间差异可以干净地归因到地幔模型：
#
#   BASE          原作者 model_updated 原样使用，零改动
#                 → 链路正确性基准，同时也是"目前可得的最好配置"参照
#   S362ANI_c1    CRUST1.0 地壳 + s362ani 地幔，不嵌任何区域模型
#                 → 共用地壳下的空对照，衡量区域地幔模型到底带来多少改进
#   FWEA23_c1     CRUST1.0 地壳 + FWEA23 地幔
#   EARA2024_c1   CRUST1.0 地壳 + EARA2024 地幔
#   SinoScope_c1  CRUST1.0 地壳 + SinoScope1.0 地幔
#   S362ANI_ref   原生 s362ani，复用 mesh0 网格，只跑 1 个事件
#                 → 与 S362ANI_c1 同事件对比，量化 GLL 往返的代价
#
# 为什么用 30 个震源:
#   单个震源只照亮一条方位的窄条带，模型间差异会被路径采样偏差掩盖。30 个事件
#   由 1_7_Build_test_database.py --mode eastasia_model_30 从 EastAsia_model 波形库
#   中按时段配额（2008–13×7, 2014–18×15, 2019–25×8）与空间约束选出，Mw≥5.5，
#   lat≥0°、lon≤143°，含浅/中/深源。排名要在多数事件上一致才算稳。
#
# 为什么共用地壳取 CRUST1.0 而不是 FWEA23 的地壳:
#   1. 网格本来就是按 CRUST1.0 的 Moho 拉伸的（moho_stretching_honor_crust），
#      只有用 CRUST1.0 时 Moho 间断面才正好压在单元边界上；
#   2. 地壳值直接从 mesher 的 solver_data.bin 反算，不经我们插值，因此不受深度
#      基准误差影响，也没有海域缺值和水层的问题；
#   3. DT 就是这套网格配 CRUST1.0+s362ani 算出来的，CFL 零风险；
#   4. 若共用 FWEA23 的地壳，等于给 FWEA23 主场优势。CRUST1.0 是公开标准地壳，
#      SinoScope 本来也没有地壳（H5 最浅 20 km），配标准地壳才是它的正常用法。
#   代价：CRUST1.0 精度不如 FWEA23 的 FWI 地壳，各组绝对拟合都会变差，9-20 s
#   尤其明显。但各组同等变差，模型间排名依然有效；BASE 保留下来报告最好拟合。
#
# 流程:
#   MESH0   mesher 一次 → mesh0/DATABASES_MPI
#           这套坐标是全流程唯一的插值基准，同时也是 CRUST1.0+s362ani 的底图来源
#   BASE    DATA/GLL 软链 model_updated，不构建
#   其余组  build_hybrid_gll.py 以 mesh0 坐标建模型 → mesher → 核对坐标 → solver
#
#   网格与模型都与震源无关，所以一组只网格化一次，solver 循环 30 个事件。每个
#   事件算完写 sac/<TAG>/<EVENT>/.done，作业被打断后重投会跳过已完成的事件。
#
# 为什么插值必须在服务器上做:
#   在本地建模型意味着要预先拿到一份与服务器 mesher 逐位相同的坐标，这个前提
#   无法保证——实测中本地参考网格与服务器网格在部分 proc 上不一致，而 NSPEC 相同
#   所以程序不会报错，速度值会安静地落在错误位置。改为读取 mesher 刚生成的坐标
#   之后，这类错误在构造上就不存在了。
#
# 用法:
#   ./run_all_test.sh prepare    校验路径与全部震源、编译、建目录、生成作业脚本
#   ./run_all_test.sh submit     提交 MESH0 + 各组（各组依赖 MESH0）
#   ./run_all_test.sh status     查看进度（含逐事件明细）
#   ./run_all_test.sh collect    核查各组各事件的 SAC 是否齐全
#   ./run_all_test.sh events     列出震源清单
#   ./run_all_test.sh resubmit <TAG>
#
# ============================================================================

set -o pipefail

# ============================================================================
# 配置
# ============================================================================

# SPECFEM3D Globe 源码根目录（含 src/ setup/ DATA/）
CODE_DIR="/work/home/acf11bgjob/specfem/specfem3d_globe_code_new"

# 数据目录：直接含 Par_file、STATIONS、CMTSOLUTION、constants.h.in
# 注意 sem_config_tibet_v4/DATA/ 里是另一个震源（C201102041353A，标准格式），
# 与原作者的参考波形对不上，不要用那一份
DATA_DIR="/work/home/acf11bgjob/specfem/DATA"

# 编译用的 constants.h.in（决定 Moho 拉伸、ECEF 震源格式等，必须与底图一致）
CONSTANTS_H_IN="$DATA_DIR/constants.h.in"

# 底图 GLL：原作者提供的 FWEA23 model_updated
BASE_GLL_DIR="/work/home/acf11bgjob/specfem/model_updated"

# MESH0 建基准网格用的 MODEL。
# mesh0 只作为坐标基准，网格几何理论上与速度模型无关，所以用哪个都行——但这个
# "理论上" 需要实测支撑，`$0 probe` 就是干这个的。
# 取 s362ani 是保守选择：它是本机唯一有干净运行记录的配置（run_forward.bash 的
# Phase 2）。MODEL=GLL 的 mesher 在本机尚未被验证过，且实测报过 ineighbour 错误。
# probe 若证明两者等价，改回 GLL 可省去 s362ani 的额外数据依赖。
MESH0_MODEL="s362ani"

# 原始模型文件目录
MODELS_DIR="/work/home/acf11bgjob/specfem/models"

# 震源集合。本地运行 package_eastasia_model_30.sh 打包后，整个
# events_eastasia_model_30/ 目录上传到服务器。events.list 每行一个事件名，
# CMTSOLUTION_ECEF/ 下是对应的 ECEF 格式震源（与 USE_ECEF_CMTSOLUTION 匹配）。
EVENTS_ROOT="/work/home/acf11bgjob/specfem/events_eastasia_model_30"
EVENTS_LIST="$EVENTS_ROOT/events.list"
EVENTS_DIR="$EVENTS_ROOT/CMTSOLUTION_ECEF"

# 原生 s362ani 对照组用的事件。须在 30 事件清单内，且台站覆盖较好。
# C201505120705A（尼泊尔 M7.23）在 2015 年观测峰值窗口内。
ROUNDTRIP_EVENT="C201505120705A"

# Python 解释器（build_hybrid_gll.py 与 check_mesh_diff.py 需要 numpy/h5py/netCDF4）。
# 这里写绝对路径而不是 module load + activate，是因为后者会安静地失败：
# `source activate eastasia_fwi` 在本集群上不报错却留在 base，而 base 没有 netCDF4，
# 作业要到跑起来才暴露。直接指解释器则没有任何中间环节可以出错。
PYTHON_BIN="/work/home/acf11bgjob/.conda/envs/eastasia_fwi/bin/python"

# 编译环境
MODULE_COMPILER="compiler/intel/2018.5.274"
MODULE_MPI="mpi/intelmpi/2018.4.274"

# Slurm 资源。4×36 是 run_forward.bash 里验证过能跑通 mesher+solver 的规格；
# 旧的 run_this_example_chujie_version.sh 用 3×48，实测 mesher 必挂。
SLURM_ACCOUNT="acwub211ea"
SLURM_PARTITION="tyhcnormal"
SLURM_NODES=4
SLURM_NTASKS_PER_NODE=36
SLURM_CONSTRAINT="64core"
SLURM_TIME_MESH="02:00:00"
SLURM_TIME_RUN="48:00:00"

# 同机架约束。集群 TopologyPlugin=topology/default，不支持 --switches，
# 只能提交时显式挑节点。同机架也便于把故障范围收敛到一组硬件上。
SAME_RACK=1
MAX_MESH_ATTEMPT=5

# 机架黑名单。probe 对照实验（2026-08-03）的结论：
#   c08r1 上 s362ani/GLL 各两轮全部崩溃，且四轮报四种互不相同的错
#         （Error Jacobian / ineighbour points differ / Error interfaces）；
#   c10r3 上同样四轮全部干净，且 144 个 proc 的坐标逐位可复现。
# 同二进制同输入下的这种差异只能归到硬件，故拉黑 c08r1。
# 若换机架后又遇到同类随机崩溃，把新机架追加进来。
BAD_RACKS="c08r1"

# 坐标核对判据：ibool 必须逐位相同；x/y/z 允许的相对误差。
# 1e-6 对应约 6 米，相对 25 km 量级的模型网格可忽略。
COORD_TOL="1e-6"

# 建模型的并行度。build_hybrid_gll.py 每个 worker 各自加载一份模型网格，
# 默认取满核数（64）会驻留 64 份，FWEA23 解压后单份即数 GB，必然 OOM。
BUILD_JOBS=16

# 替换区（度 / km）。边界处 w=0，taper 向内渐变到 1。
REGION_LATLON="10.0 55.0 80.0 150.0"
HORIZ_TAPER="5.0"

# 上界跟随 CRUST1.0 的 Moho：Moho 以上一律保留 CRUST1.0 地壳，以下换成各自的
# 地幔模型。DEPTH_RANGE 的上界在 moho 模式下被忽略，只有下界 1000 km 生效。
DEPTH_MIN_MODE="moho"
MOHO_OFFSET="0.0"
DEPTH_RANGE="0.0 1000.0"

# Moho 以下的过渡宽度取 15 km。它要盖住两处不确定性：我们用双线性读 crust1.bnds，
# 而 SPECFEM 查询时做 CAP 平滑，两者在 Moho 起伏剧烈处可差几 km；深度基准本身
# 在地壳内还有约 1 km 残差。下界 50 km 沿用原值。
DEPTH_TAPER="15.0 50.0"

# ============================================================================
# 以下一般无需修改
# ============================================================================

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$CURRENT_DIR/$(basename "${BASH_SOURCE[0]}")"
BUILD_PY="$CURRENT_DIR/build_hybrid_gll.py"
CHECK_PY="$CURRENT_DIR/check_mesh_diff.py"

TEST_ROOT="$CURRENT_DIR/eastasia_model_30_test"
JOBS_DIR="$TEST_ROOT/jobs"
RUNS_DIR="$TEST_ROOT/runs"
MESH0_DIR="$TEST_ROOT/mesh0"
PROBE_DIR="$TEST_ROOT/probe"
SAC_DIR="$TEST_ROOT/sac"

RUN_TAGS="BASE S362ANI_c1 FWEA23_c1 EARA2024_c1 SinoScope_c1 SinoScope_sem_c1"

# 需要在作业内构建 DATA/GLL 的组。S362ANI_c1 虽然不嵌任何模型，但它的底图是从
# solver_data.bin 反算出来的、磁盘上并不存在，同样得走一遍构建落盘。
HYBRID_TAGS="S362ANI_c1 FWEA23_c1 EARA2024_c1 SinoScope_c1 SinoScope_sem_c1"

# 原生 s362ani 对照组。它不建模型、不跑自己的 mesher，直接复用 mesh0 的网格
# （mesh0 本来就是 MODEL=s362ani 跑出来的），只在 ROUNDTRIP_EVENT 上跑一次 solver。
# 与 S362ANI_c1 的波形之差 = 把 s362ani 提取成 GLL 再读回来这一往返的全部代价。
# 两组的衰减模型都取 1DREF（s362ani 的 REFERENCE_1D_MODEL 与 constants.h.in 里的
# GLL_REFERENCE_1D_MODEL 恰好一致），所以差异里不掺非弹性成分。
REF_TAG="S362ANI_ref"

hr()   { echo "======================================================================"; }
info() { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "❌ $*" >&2; exit 1; }

# 事件名列表，由 load_events 填充
EVENTS=""
N_EVENTS=0

load_events() {
    [ -f "$EVENTS_LIST" ] || die "找不到事件清单: $EVENTS_LIST
   本地运行 specfem/package_eastasia_model_30.sh，再把 events_eastasia_model_30/ 上传到 $EVENTS_ROOT"
    EVENTS=$(grep -vE '^[[:space:]]*(#|$)' "$EVENTS_LIST" | tr -d '\r')
    N_EVENTS=$(echo "$EVENTS" | grep -c . )
    [ "$N_EVENTS" -gt 0 ] || die "事件清单为空: $EVENTS_LIST"
}

# 某组要跑的事件。参照组只跑一个，其余组跑全部。
events_of() {
    if [ "$1" = "$REF_TAG" ]; then echo "$ROUNDTRIP_EVENT"; else echo "$EVENTS"; fi
}

# 参照组不建模型、不跑 mesher，很多逻辑要分流
is_ref() { [ "$1" = "$REF_TAG" ]; }

# 各混合组对应的原始模型文件
# 各组要嵌入的地幔模型。S362ANI_c1 不嵌任何模型，只把底图落盘，故返回空串。
model_opt() {
    case "$1" in
        S362ANI_c1)   echo "" ;;
        FWEA23_c1)    echo "--model-nc $MODELS_DIR/FWEA23.r0.0-n4.nc" ;;
        EARA2024_c1)  echo "--model-nc $MODELS_DIR/EARA2024.r0.0-n4.nc" ;;
        SinoScope_c1)     echo "--model-h5 $MODELS_DIR/FWI_SinoScope_1.0_JMa_ET_AL_2022.h5" ;;
        SinoScope_sem_c1) echo "--model-salvus $MODELS_DIR/mesh.h5" ;;
        *)                return 1 ;;
    esac
}

read_par() {
    local key="$1" file="$2"
    grep "^[[:space:]]*${key}[[:space:]]*=" "$file" | head -1 | cut -d= -f2 | tr -d ' \t'
}

# 节点名形如 c08r1n20：去掉尾部 n<编号> 即机架标识 c08r1
rack_of() { echo "$1" | sed 's/n[0-9]*$//'; }

# 从一批候选节点里筛出真正可立刻开跑的。
# 关键：必须排除 IDLE+PLANNED。sinfo -t idle 会把「空闲但已有作业预约」的节点
# 也列出来；若再 --nodelist 钉死它们，作业会长期卡在 Reason=Priority，而那 4 个
# 节点其实马上会被别人占掉（昨夜 MESH0 钉死 c04r1n26-28,37 即此情形）。
allocatable_nodes() {
    scontrol show node "$1" 2>/dev/null | awk '
        {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^NodeName=/)   { split($i, a, "="); node = a[2] }
                else if ($i ~ /^State=/) {
                    split($i, b, "="); st = b[2]
                    if (node != "" && st ~ /IDLE/ &&
                        st !~ /PLANNED|DRAIN|DOWN|FAIL|MAINT|RESV|COMPLETING|POWER|NOT_RESPONDING/)
                        print node
                }
            }
        }'
}

# 挑 $1 个同机架且确实可分配的节点，输出 "机架 节点1,节点2,..."。
# $2 为要跳过的机架（空格分隔）。挑不到返回非零。
pick_rack_nodes() {
    local want="$1" skip=" ${2:-} "
    local list racks rack cands ok n picked

    list=$(sinfo -h -N -p "$SLURM_PARTITION" -t idle -o "%N|%f" 2>/dev/null \
           | awk -F'|' -v feat="$SLURM_CONSTRAINT" \
                 'feat == "" || index($2, feat) { print $1 }' | sort -u)
    [ -n "$list" ] || return 1

    racks=$(echo "$list" | sed 's/n[0-9]*$//' | sort -u)
    for rack in $racks; do
        case "$skip" in *" $rack "*) continue ;; esac
        case " $BAD_RACKS " in *" $rack "*) continue ;; esac
        cands=$(echo "$list" | grep "^${rack}n" | paste -sd, -)
        [ -n "$cands" ] || continue
        ok=$(allocatable_nodes "$cands")
        n=$(echo "$ok" | grep -c .)
        [ "$n" -ge "$want" ] || continue
        picked=$(echo "$ok" | head -n "$want" | paste -sd, -)
        echo "$rack $picked"
        return 0
    done
    return 1
}

# ============================================================================
# Python 核对
#
# 两种场景都要验，因为混合组作业在两个不同的环境状态下调用 Python：
#   阶段 1 建模型  — 未加载任何 module
#   阶段 3 坐标核对 — Intel 编译器与 MPI 已加载，LD_LIBRARY_PATH 被大幅改写
# 后者是真实踩过的风险点：conda 的 Python 链接到 Intel 的 libstdc++ 之类会直接崩。
#
# 这里不用 `module load anaconda3 && source activate`——本集群上该写法不报错却留在
# base，而 base 没有 netCDF4，作业要跑起来才暴露。直接指解释器绝对路径没有这个问题。
# ============================================================================
check_python() {
    hr; echo "Python 核对"; hr
    [ -x "$PYTHON_BIN" ] || die "Python 解释器不存在或不可执行: $PYTHON_BIN"
    echo "  解释器: $PYTHON_BIN"

    local probe='import sys, numpy, h5py, netCDF4
print("  前缀: " + sys.prefix)
print("  numpy %s  h5py %s  netCDF4 %s" % (numpy.__version__, h5py.__version__, netCDF4.__version__))'

    echo "  [场景 1] 裸环境（阶段 1 建模型）"
    "$PYTHON_BIN" -c "$probe" || die "裸环境下依赖不全，检查 $PYTHON_BIN 所在环境"

    echo "  [场景 2] Intel 编译器与 MPI 已加载（阶段 3 坐标核对）"
    ( module purge
      module load "$MODULE_COMPILER" 2>/dev/null
      module load "$MODULE_MPI" 2>/dev/null
      "$PYTHON_BIN" -c "$probe" ) \
        || die "Intel 模块加载后 Python 无法运行（多半是 LD_LIBRARY_PATH 冲突）。
   对策：在作业脚本阶段 3 之前插入 module purge，或改用 env -i 调用解释器"

    echo "✅ 两种场景下 Python 均可用"
}

# ============================================================================
# 震源核对：每个事件的 CMTSOLUTION 是否存在、事件号是否与文件名一致、格式是否
# 与 USE_ECEF_CMTSOLUTION 匹配。
#
# 这里拦的是一个真实踩过的坑：sem_config_tibet_v4/DATA/CMTSOLUTION 是另一个
# 事件（C201102041353A，缅甸）且为标准格式，而 constants.h.in 开了
# USE_ECEF_CMTSOLUTION，两者都与原作者的参考波形对不上。用错了不会报错，
# 只会安静地算出一个无法与参考对比的结果。事件多了以后这类错配的机会成倍增加，
# 所以逐个查，且在提交前查完——跑到第 7 个事件才发现文件名写错代价太大。
# ============================================================================
check_sources() {
    hr; echo "震源核对（$N_EVENTS 个事件）"; hr
    [ -d "$EVENTS_DIR" ] || die "找不到震源目录: $EVENTS_DIR"

    local want_ecef=""
    if [ -f "$CONSTANTS_H_IN" ]; then
        want_ecef=$(grep "USE_ECEF_CMTSOLUTION" "$CONSTANTS_H_IN" \
               | grep -v "^[[:space:]]*!" | head -1 | grep -o "\.true\.\|\.false\." | head -1)
    fi
    echo "  constants.h.in 的 USE_ECEF_CMTSOLUTION: ${want_ecef:-<未识别>}"

    local n_sta
    n_sta=$(grep -c . "$DATA_DIR/STATIONS" 2>/dev/null || echo 0)
    echo "  台站: $n_sta"
    hr
    printf "  %-16s %-6s %-9s %s\n" "事件" "格式" "半持续时间" "PDE 首行"

    local ev cmt fmt ev_in hdur
    for ev in $EVENTS; do
        cmt="$EVENTS_DIR/CMTSOLUTION_$ev"
        [ -f "$cmt" ] || die "缺少震源文件: $cmt"

        ev_in=$(awk -F: '/^event name:/ {gsub(/[[:space:]]/,"",$2); print $2; exit}' "$cmt")
        [ "$ev_in" = "$ev" ] \
            || die "$cmt 内的事件名是 ${ev_in:-<空>}，与文件名的 $ev 不符"

        if grep -q "^x(m):" "$cmt"; then fmt="ECEF"; else fmt="标准"; fi
        [ "$want_ecef" = ".true."  ] && [ "$fmt" != "ECEF" ] \
            && die "$ev: constants.h.in 要求 ECEF 格式，但该文件是标准格式，会被误解析"
        [ "$want_ecef" = ".false." ] && [ "$fmt" = "ECEF" ] \
            && die "$ev: constants.h.in 要求标准格式，但该文件是 ECEF 格式，会被误解析"

        # ECEF 版把三角形源时间函数折算成高斯的 tau，标准版仍是 half duration
        hdur=$(awk -F: '/^tau\(s\):|^half duration:/ {gsub(/[[:space:]]/,"",$2); print $2; exit}' "$cmt")
        printf "  %-16s %-6s %-9s %s\n" "$ev" "$fmt" "${hdur:-?}" "$(head -1 "$cmt" | cut -c1-46)"
    done

    # 参照组的事件必须在清单里，否则它跑的是一个别组没算过的震源，无从对比
    echo "$EVENTS" | grep -qx "$ROUNDTRIP_EVENT" \
        || die "ROUNDTRIP_EVENT=$ROUNDTRIP_EVENT 不在事件清单里，$REF_TAG 将无组可比"

    hr
    echo "✅ $N_EVENTS 个震源齐备，格式与编译开关一致"
    echo "   $REF_TAG 用 ${ROUNDTRIP_EVENT}（与 S362ANI_c1 同事件对比）"
}

# ============================================================================
# prepare
# ============================================================================
stage_prepare() {
    load_events
    hr; echo "阶段 PREPARE — 多震源多模型正演测试（服务器端建模型）"; hr
    info "震源:       $N_EVENTS 个（${EVENTS_LIST}）"
    info "模型组:     $RUN_TAGS"
    info "参照组:     ${REF_TAG}（${ROUNDTRIP_EVENT}）"
    info "测试根目录: $TEST_ROOT"

    # ---- 1. 路径校验 ----
    hr; echo "1. 路径校验"; hr
    [ -d "$CODE_DIR" ]     || die "源码目录不存在: $CODE_DIR"
    [ -d "$DATA_DIR" ]     || die "数据目录不存在: $DATA_DIR"
    [ -d "$BASE_GLL_DIR" ] || die "底图目录不存在: $BASE_GLL_DIR"
    [ -d "$MODELS_DIR" ]   || die "模型目录不存在: $MODELS_DIR"
    [ -d "$EVENTS_DIR" ]   || die "震源目录不存在: $EVENTS_DIR"
    [ -f "$BUILD_PY" ]     || die "找不到 build_hybrid_gll.py: $BUILD_PY"
    [ -f "$CHECK_PY" ]     || die "找不到 check_mesh_diff.py: $CHECK_PY"
    [ -f "$CURRENT_DIR/load_salvus_mesh.py" ] \
        || die "找不到 load_salvus_mesh.py（SinoScope_sem_c1 需要）"
    for f in Par_file STATIONS; do
        [ -f "$DATA_DIR/$f" ] || die "缺少 $DATA_DIR/$f"
    done
    [ -f "$CONSTANTS_H_IN" ] || die "找不到 constants.h.in: $CONSTANTS_H_IN"

    local tag opt
    for tag in $HYBRID_TAGS; do
        opt=$(model_opt "$tag") || die "$tag 未配置模型文件"
        if [ -z "$opt" ]; then
            echo "  $tag  <- 不嵌模型（CRUST1.0 + s362ani 底图本身）"
            continue
        fi
        local mf="${opt#* }"
        [ -f "$mf" ] || die "$tag 的模型文件不存在: $mf"
        echo "  $tag  <- $(basename "$mf")"
    done

    # moho 模式的替换上界要读 CRUST1.0 的层面文件，缺了作业跑起来才会暴露
    if [ "$DEPTH_MIN_MODE" = "moho" ]; then
        [ -f "$CODE_DIR/DATA/crust1.0/crust1.bnds" ] \
            || die "替换上界跟随 Moho，但找不到 $CODE_DIR/DATA/crust1.0/crust1.bnds"
        echo "  Moho 上界 <- DATA/crust1.0/crust1.bnds"
    fi
    echo "✅ 路径与模型文件齐备"

    check_python
    check_sources

    # ---- 2. Par_file ----
    hr; echo "2. Par_file 网格参数"; hr
    local NCHUNKS NEX_XI NPROC_XI NPROC_ETA MODEL
    NCHUNKS=$(read_par NCHUNKS "$DATA_DIR/Par_file")
    NEX_XI=$(read_par NEX_XI "$DATA_DIR/Par_file")
    NPROC_XI=$(read_par NPROC_XI "$DATA_DIR/Par_file")
    NPROC_ETA=$(read_par NPROC_ETA "$DATA_DIR/Par_file")
    MODEL=$(read_par MODEL "$DATA_DIR/Par_file")
    local NPROC_TOTAL=$((NCHUNKS * NPROC_XI * NPROC_ETA))
    local SLURM_TASKS=$((SLURM_NODES * SLURM_NTASKS_PER_NODE))

    echo "  NCHUNKS=$NCHUNKS  NEX_XI=$NEX_XI  NPROC_XI=$NPROC_XI  NPROC_ETA=$NPROC_ETA"
    echo "  MODEL=$MODEL"
    echo "  需要 MPI 进程: $NPROC_TOTAL    Slurm 提供: $SLURM_TASKS"
    echo "  （Globe 的 DT 由 mesher 自动计算，Par_file 中没有该项）"

    [ "$MODEL" = "GLL" ] || die "Par_file 的 MODEL 应为 GLL，当前是 $MODEL"
    [ "$SLURM_TASKS" -ge "$NPROC_TOTAL" ] \
        || die "Slurm 任务数 $SLURM_TASKS 少于所需 $NPROC_TOTAL"

    local n_base
    n_base=$(ls "$BASE_GLL_DIR"/proc*_reg1_vsv.bin 2>/dev/null | wc -l | tr -d ' ')
    [ "$n_base" -eq "$NPROC_TOTAL" ] \
        || die "底图 vsv 文件数 ${n_base} != ${NPROC_TOTAL}，与 Par_file 的 NPROC 不匹配"
    echo "✅ 底图含 $n_base 个 proc 的 GLL 文件"

    mkdir -p "$TEST_ROOT"/{bin,logs} "$JOBS_DIR"

    # ---- 3. 编译（只做一次，各组作业共用） ----
    hr; echo "3. 编译 xmeshfem3D / xspecfem3D"; hr
    if [ -x "$TEST_ROOT/bin/xspecfem3D" ] && [ "${FORCE_BUILD:-0}" != "1" ]; then
        echo "已存在可执行文件，跳过编译（FORCE_BUILD=1 可强制重编）"
    else
        ( set -e
          cd "$CODE_DIR"
          module purge
          module load "$MODULE_COMPILER"
          module load "$MODULE_MPI"
          echo "编译器: $(which ifort)"; ifort --version | head -1

          [ -f setup/constants.h.in.original_backup ] || \
              cp setup/constants.h.in setup/constants.h.in.original_backup
          cp "$CONSTANTS_H_IN" setup/constants.h.in
          echo "✅ 使用 $CONSTANTS_H_IN"
          grep -E "REGIONAL_MOHO_MESH|RMOHO_STRETCH_ADJUSTMENT|GLL_REFERENCE_1D_MODEL|USE_ECEF_CMTSOLUTION" \
              setup/constants.h.in | grep -v "^!" | head -6

          cp "$DATA_DIR/Par_file" "$DATA_DIR/CMTSOLUTION" "$DATA_DIR/STATIONS" DATA/

          ./configure FC=ifort CC=icc CXX=icpc MPIFC=mpiifort CUSTOM_REAL=SIZE_REAL
          make clean
          make -j8 xmeshfem3D
          make -j8 xspecfem3D
        ) 2>&1 | tee "$TEST_ROOT/logs/build.log"

        [ -x "$CODE_DIR/bin/xmeshfem3D" ] || die "xmeshfem3D 编译失败，见 logs/build.log"
        [ -x "$CODE_DIR/bin/xspecfem3D" ] || die "xspecfem3D 编译失败，见 logs/build.log"
        cp "$CODE_DIR/bin/xmeshfem3D" "$CODE_DIR/bin/xspecfem3D" "$TEST_ROOT/bin/"
        cp "$CODE_DIR/setup/constants.h" "$TEST_ROOT/logs/" 2>/dev/null || true
        echo "✅ 编译完成，可执行文件已放入 $TEST_ROOT/bin/"
    fi

    # ---- 4. 建目录 ----
    hr; echo "4. 建立 mesh0 与各 run 目录"; hr
    setup_workdir "$MESH0_DIR"
    ln -sfn "$BASE_GLL_DIR" "$MESH0_DIR/DATA/GLL"
    echo "✅ mesh0: MODEL=${MESH0_MODEL}，仅作为坐标基准"
    [ "$MESH0_MODEL" = "GLL" ] && echo "   DATA/GLL -> $BASE_GLL_DIR"

    for tag in $RUN_TAGS; do
        setup_workdir "$RUNS_DIR/$tag"
        if [ "$tag" = "BASE" ]; then
            ln -sfn "$BASE_GLL_DIR" "$RUNS_DIR/$tag/DATA/GLL"
            echo "✅ BASE: DATA/GLL -> ${BASE_GLL_DIR}（零改动对照）"
        else
            echo "✅ $tag: DATA/GLL 将在作业内由 build_hybrid_gll.py 生成"
        fi
    done
    setup_workdir "$RUNS_DIR/$REF_TAG" shared-mesh
    echo "✅ $REF_TAG: DATABASES_MPI -> mesh0（原生 ${MESH0_MODEL}，只跑 ${ROUNDTRIP_EVENT}）"

    mkdir -p "$SAC_DIR"

    # ---- 5. 作业脚本 ----
    hr; echo "5. 生成 Slurm 作业脚本"; hr
    write_mesh0_script "$NPROC_TOTAL"
    for tag in $RUN_TAGS; do
        write_run_script "$tag" "$NPROC_TOTAL"
    done
    write_ref_script "$REF_TAG" "$NPROC_TOTAL"

    # ---- 6. 规模与耗时预估 ----
    hr; echo "6. 规模预估"; hr
    local n_solver=$(( N_EVENTS * $(echo "$RUN_TAGS" | wc -w) + 1 ))
    echo "  模型组: $(echo "$RUN_TAGS" | wc -w) 组 × $N_EVENTS 事件 + 参照组 1 次 = $n_solver 次 solver"
    echo "  单次 solver 实测约 70 分钟（$SLURM_NODES 节点 × $SLURM_NTASKS_PER_NODE 核，30 分钟记录）"
    echo "  单组 $N_EVENTS 事件约 $(( N_EVENTS * 70 / 60 )) 小时；作业时限 $SLURM_TIME_RUN"
    echo "  单组作业若跑不完会自动重投接着跑（按事件 .done 断点续跑）"
    echo "  台站文件须已部署: cp events_eastasia_model_30/STATIONS DATA/STATIONS"

    hr
    echo "✅ PREPARE 完成"
    echo ""
    echo "下一步:  $0 submit"
    hr
}

# 建一个标准工作目录（DATA 里放 Par_file/STATIONS/CMTSOLUTION 与共享数据软链）。
# 第二个参数为 "shared-mesh" 时，DATABASES_MPI 做成指向 mesh0 的软链——只有不跑
# 自己 mesher 的参照组能这么用。
setup_workdir() {
    local wd="$1" mesh_mode="${2:-own}"
    # 早期版本把 BASE 的 DATABASES_MPI 做成了指向 mesh0 的软链以复用网格。现在
    # BASE 要跑自己的 mesher，沿用那个软链会直接写进坐标基准，而且坐标核对会变成
    # mesh0 跟自己比，必然通过却毫无意义。mkdir -p 对已存在的软链不报错也不替换，
    # 所以必须显式清掉。（DATA/GLL 的软链是有意为之，不动。）
    if [ -L "$wd/DATABASES_MPI" ]; then
        echo "   清理遗留软链: $wd/DATABASES_MPI -> $(readlink "$wd/DATABASES_MPI")"
        rm -f "$wd/DATABASES_MPI"
    fi
    mkdir -p "$wd"/{DATA,OUTPUT_FILES,bin}
    if [ "$mesh_mode" = "shared-mesh" ]; then
        # 正演 (SIMULATION_TYPE=1, SAVE_FORWARD=.false.) 只读 DATABASES_MPI，
        # 不会写回，因此多个作业共读 mesh0 这一份是安全的
        rm -rf "$wd/DATABASES_MPI"
        ln -sfn "$MESH0_DIR/DATABASES_MPI" "$wd/DATABASES_MPI"
    else
        mkdir -p "$wd/DATABASES_MPI"
    fi
    cp "$DATA_DIR/Par_file" "$wd/DATA/"
    cp "$DATA_DIR/STATIONS" "$wd/DATA/"
    # 占位震源。真正跑的时候阶段 4 会逐个事件覆盖它，这里放第一个只是为了让
    # mesher 阶段有一份格式正确的文件可读。
    cp "$EVENTS_DIR/CMTSOLUTION_$(echo "$EVENTS" | head -1)" "$wd/DATA/CMTSOLUTION"
    ln -sfn "$TEST_ROOT/bin/xmeshfem3D" "$wd/bin/xmeshfem3D"
    ln -sfn "$TEST_ROOT/bin/xspecfem3D" "$wd/bin/xspecfem3D"
    local d
    for d in crust1.0 crust2.0 s362ani QRFSI12 topo_bathy; do
        [ -d "$CODE_DIR/DATA/$d" ] && ln -sfn "$CODE_DIR/DATA/$d" "$wd/DATA/$d"
    done
}

# ============================================================================
# 作业脚本片段
# ============================================================================

# Slurm 头部
sbatch_header() {
    local name="$1" tlimit="$2"
    cat << EOF
#!/bin/bash
#SBATCH --job-name=hyb_$name
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --nodes=$SLURM_NODES
#SBATCH --ntasks-per-node=$SLURM_NTASKS_PER_NODE
#SBATCH --constraint=$SLURM_CONSTRAINT
#SBATCH --time=$tlimit
#SBATCH --output=$TEST_ROOT/logs/${name}_%j.out
#SBATCH --error=$TEST_ROOT/logs/${name}_%j.err
EOF
}

# 加载 Intel 编译器 + MPI 的环境块
env_intel() {
    cat << EOF
module purge
module load $MODULE_COMPILER
module load $MODULE_MPI
ulimit -s unlimited
ulimit -l unlimited
export KMP_STACKSIZE=512m
export OMP_STACKSIZE=512m
export OMP_NUM_THREADS=1
# MPI 运行时：与 run_forward.bash 保持一致。
# 不强制 fabric——旧脚本的 I_MPI_FABRICS=shm:dapl 配 I_MPI_FALLBACK=0 会把节点间
# 通信钉死在 legacy DAPL 路径上，meshfem 建 MPI 界面时的整数扫描累加对该路径上的
# 偶发不一致极其敏感，实测必然报 face flag / ineighbour points differ。
export I_MPI_DEBUG=5
export I_MPI_ADJUST_ALLREDUCE=5
export I_MPI_ADJUST_ALLTOALL=2
EOF
}

# ---------------------------------------------------------------------------
# mesher 执行块（MESH0 与三个混合组共用）
#
# 通过判据比「文件数够了」严格得多，原因是实测踩过的坑：mesher 在 MPI 界面校验
# 阶段中止时，144 个 solver_data.bin 其实已经全部写完了。按文件数判断会把这份
# 带界面不对称的网格当成好网格，重投时直接跳过并标记完成，后续全程不报错。
#
# 因此改为：只有 mesher 退出码为 0、文件数齐、且日志里零界面告警零错误，才写
# .mesh_ok 标记；跳过逻辑只认这个标记。任何一项不过就清空数据库再重投，杜绝
# 坏网格被复用。
#
# 界面告警在本项目里等同于失败。源码把若干一致性检查降级成了 warning
# （Models_GLL_workflow_server.md §19.4），放行的网格可能真的有拓扑问题。
# ---------------------------------------------------------------------------
mesher_block() {
    local tag="$1" NPROC="$2" label="$3"
    cat << EOF
if [ -f DATABASES_MPI/.mesh_ok ] && [ "\${FORCE_MESH:-0}" != "1" ]; then
    echo "--- mesher: ${label}已通过校验，跳过 ---"
else
    # mesher 不会自建输出目录，缺目录时第一个写 topo.bin 的进程就 MPI_Abort(30)，
    # 报的是「Error opening file ... please check if path exists」。
    mkdir -p DATABASES_MPI OUTPUT_FILES
    rm -f DATABASES_MPI/.mesh_ok
    echo "--- mesher (${label}) ---"
    t0=\$(date +%s)
    mpirun -np $NPROC ./bin/xmeshfem3D
    st=\$?
    echo "mesher 退出码=\$st 耗时=\$(( \$(date +%s) - t0 )) 秒"

    jo="$TEST_ROOT/logs/${tag}_\${SLURM_JOB_ID}.out"
    je="$TEST_ROOT/logs/${tag}_\${SLURM_JOB_ID}.err"
    n_sd=\$(ls DATABASES_MPI/proc*_reg1_solver_data.bin 2>/dev/null | wc -l | tr -d ' ')
    n_warn=\$(cat "\$jo" "\$je" 2>/dev/null \\
              | grep -cE "flag: missed|WARNING MPI interface" || true)
    n_err=\$(cat "\$jo" "\$je" 2>/dev/null \\
             | grep -cE "Error ineighbour|Error neighbour|MAX_NEIGHBOURS|Error.*valence|MPI_Abort" || true)
    echo "校验: 退出码=\$st  solver_data=\$n_sd/$NPROC  界面告警=\$n_warn  界面错误=\$n_err"

    if [ \$st -eq 0 ] && [ "\$n_sd" -eq "$NPROC" ] \\
       && [ "\$n_warn" -eq 0 ] && [ "\$n_err" -eq 0 ]; then
        echo "\$(date) job=\${SLURM_JOB_ID} nodes=\${SLURM_JOB_NODELIST}" > DATABASES_MPI/.mesh_ok
        echo "✅ 网格干净，已写 .mesh_ok"
    else
        echo "❌ mesher 未通过校验，第 \$ATTEMPT 次尝试"
        [ "\$n_warn" -gt 0 ] && echo "   界面告警视同失败——被降级的一致性检查放行的网格不可信"
        echo "   清空 DATABASES_MPI，避免这份网格被后续跳过逻辑当成好网格"
        rm -f DATABASES_MPI/proc*.bin DATABASES_MPI/.mesh_ok DATABASES_MPI/.coord_ok
$(retry_block "$tag")
        exit 1
    fi
fi
EOF
}

# mesher 失败后换一批同机架节点重投（同节点重试无效）
retry_block() {
    local tag="$1"
    cat << EOF
        # 环境类错误先拦下来。它们同样以 MPI_Abort 收场，但换节点解决不了，
        # 放任自动重投只会把同一个死因重复 MAX_MESH_ATTEMPT 遍（实测踩过：
        # DATABASES_MPI 目录被手工删掉后连挂 5 次）。
        if grep -qE "please check if path exists|Error opening file|No such file or directory|cannot open" \\
                 "\$jo" "\$je" 2>/dev/null; then
            echo "   识别为文件/路径错误，与节点无关，不自动重投"
            grep -m3 -E "Error opening file|please check if path exists|No such file" "\$jo" "\$je" 2>/dev/null
            echo "   多半是工作目录缺失或链接失效，跑一次 $SELF prepare 重建后再提交"
        elif grep -qE "face flag|MAX_NEIGHBOURS|valence|ineighbour|flag: missed|MPI points remain unrecognized|MPI_Abort" \\
                 "\$jo" "\$je" 2>/dev/null; then
            echo "   识别为 MPI 界面构造错误"
            if [ "\$ATTEMPT" -lt $MAX_MESH_ATTEMPT ]; then
                cur=\$("$SELF" rackof "\$(scontrol show hostnames "\${SLURM_JOB_NODELIST:-}" 2>/dev/null | head -1)")
                nodes=\$("$SELF" picknodes "\$cur")
                if [ -n "\$nodes" ]; then
                    echo "   换节点重投: \$nodes"
                    sbatch --nodelist="\$nodes" \\
                           --export=ALL,HYB_ATTEMPT=\$((ATTEMPT + 1)) \\
                           "$JOBS_DIR/run_${tag}.slurm"
                else
                    echo "   ⚠️  暂无可用同机架节点，稍后手动重投: $SELF resubmit $tag"
                fi
            else
                echo "   已连续 $MAX_MESH_ATTEMPT 次失败，停止自动重投"
                echo "   这类错误呈间歇性，也可能是网格几何本身的问题，需人工介入"
            fi
        else
            echo "   非 MPI 界面错误，不自动重投，请查看 output_mesher.txt"
        fi
EOF
}

# ---------------------------------------------------------------------------
# PROBE：mesher 对照实验
#
# 要回答两个已经卡住流程的问题，现有证据都不足以分辨：
#
#   Q1  MODEL=GLL 的 mesher 是否比 MODEL=s362ani 更容易触发 MPI 界面错误？
#       已知 s362ani 在 c12r4 上跑干净过（run_forward.bash 的 MESH_ONLY 运行），
#       GLL 在 c08r1 上报了 ineighbour points differ。两个变量同时不同，
#       无法归因。而该流程从未在本机验证过 MODEL=GLL 的 mesher。
#
#   Q2  同一批节点、同一个二进制、同一个 MODEL 跑两次，坐标是否逐位相同？
#       这是 §9.5 的核心问题，决定了「以 mesh0 坐标建模型」这条路是否成立。
#
# 四轮全部放在同一个作业里，节点分配和二进制完全相同，MODEL 是唯一变量；
# 同 MODEL 的两轮之间比坐标，即可分离出可复现性。mesher 约 1 分钟一轮。
# ---------------------------------------------------------------------------
write_probe_script() {
    local NPROC="$1"
    local wd="$PROBE_DIR/work"
    {
    sbatch_header PROBE "$SLURM_TIME_MESH"
    cat << EOF

set -u
echo "===== PROBE 开始: \$(date) ====="
echo "节点: \${SLURM_JOB_NODELIST:-<未知>}"
echo "四轮 mesher: s362ani, GLL, s362ani, GLL（同一批节点，MODEL 为唯一变量）"

$(env_intel)

cd "$wd" || exit 1
REPORT="$PROBE_DIR/report.txt"
: > "\$REPORT"

# rank 到节点的映射。mpirun 按块分配，前 PER_NODE 个 rank 在第一个节点，依此类推。
# 出错 rank 落在哪台机器上，是判断「坏节点」还是「普遍性问题」的直接依据。
NODES_ARR=(\$(scontrol show hostnames "\${SLURM_JOB_NODELIST:-}" 2>/dev/null))
PER_NODE=\$(( $NPROC / \${#NODES_ARR[@]:-1} ))
rank2node() {
    [ \${#NODES_ARR[@]} -gt 0 ] || { echo "?"; return; }
    idx=\$(( \$1 / PER_NODE ))
    [ \$idx -lt \${#NODES_ARR[@]} ] && echo "\${NODES_ARR[\$idx]}" || echo "?"
}

echo "节点: \${SLURM_JOB_NODELIST:-<未知>}  (每节点 \$PER_NODE rank)" | tee -a "\$REPORT"
echo "" | tee -a "\$REPORT"
printf "%-6s %-10s %8s %12s %6s %6s %s\n" \\
    轮次 MODEL 退出码 solver_data 告警 错误 结论 | tee -a "\$REPORT"

run_round() {
    r=\$1; mdl=\$2
    rlog="$PROBE_DIR/round\${r}_\${mdl}.log"
    sed -i "/^MODEL[[:space:]]*=/s|=.*|= \$mdl|" DATA/Par_file
    cp DATA/Par_file OUTPUT_FILES/ 2>/dev/null
    # 连轮之间清干净，排除上一轮残留状态造成的连锁反应
    rm -f DATABASES_MPI/proc*.bin
    rm -f OUTPUT_FILES/output_mesher.txt OUTPUT_FILES/values_from_mesher.h

    mpirun -np $NPROC ./bin/xmeshfem3D > "\$rlog" 2>&1
    st=\$?
    nsd=\$(ls DATABASES_MPI/proc*_reg1_solver_data.bin 2>/dev/null | wc -l | tr -d ' ')
    nw=\$(grep -cE "flag: missed|WARNING MPI interface" "\$rlog" || true)
    ne=\$(grep -cE "Error ineighbour|Error neighbour|MAX_NEIGHBOURS|Error.*valence|MPI_Abort" "\$rlog" || true)
    if [ \$st -eq 0 ] && [ "\$nsd" -eq "$NPROC" ] && [ "\$nw" -eq 0 ] && [ "\$ne" -eq 0 ]; then
        verdict="干净"
    else
        verdict="有问题"
    fi
    printf "%-6s %-10s %8s %12s %6s %6s %s\n" \\
        "\$r" "\$mdl" "\$st" "\$nsd/$NPROC" "\$nw" "\$ne" "\$verdict" | tee -a "\$REPORT"

    if [ "\$verdict" = "有问题" ]; then
        firsterr=\$(grep -m1 -iE "^ *Error|forrtl|MPI_Abort" "\$rlog" | sed 's/^ *//')
        badrank=\$(grep -m1 -oE "(rank:|proc) +[0-9]+" "\$rlog" | grep -oE "[0-9]+")
        if [ -n "\$badrank" ]; then
            echo "       首个错误: \$firsterr  [rank \$badrank @ \$(rank2node \$badrank)]" | tee -a "\$REPORT"
        else
            echo "       首个错误: \$firsterr" | tee -a "\$REPORT"
        fi
    fi

    # 只留坐标段（约 9 MB/proc），完整数据库太占空间。
    # check_mesh_diff.py 顺序读到 ibool 就停，截断文件不影响比对。
    cdir="$PROBE_DIR/coords_r\${r}"
    rm -rf "\$cdir"; mkdir -p "\$cdir"
    if [ "\$nsd" -gt 0 ]; then
        for f in DATABASES_MPI/proc*_reg1_solver_data.bin; do
            head -c 12000000 "\$f" > "\$cdir/\$(basename "\$f")"
        done
    fi
}

run_round 1 s362ani
run_round 2 GLL
run_round 3 s362ani
run_round 4 GLL

echo "" | tee -a "\$REPORT"
echo "坐标可复现性（同节点、同二进制、同 MODEL 的两次运行）" | tee -a "\$REPORT"
echo "======================================================" | tee -a "\$REPORT"

for pair in "1 3 s362ani" "2 4 GLL"; do
    set -- \$pair
    a=\$1; b=\$2; m=\$3
    na=\$(ls "$PROBE_DIR/coords_r\$a" 2>/dev/null | wc -l | tr -d ' ')
    nb=\$(ls "$PROBE_DIR/coords_r\$b" 2>/dev/null | wc -l | tr -d ' ')
    if [ "\$na" -eq "$NPROC" ] && [ "\$nb" -eq "$NPROC" ]; then
        echo "" | tee -a "\$REPORT"
        echo "--- \$m: 轮\$a vs 轮\$b ---" | tee -a "\$REPORT"
        "$PYTHON_BIN" "$CHECK_PY" \\
            "$PROBE_DIR/coords_r\$a" "$PROBE_DIR/coords_r\$b" \\
            --nproc $NPROC --tol 0 2>&1 | tail -14 | tee -a "\$REPORT"
    else
        echo "\$m: 轮\$a(\$na) 或 轮\$b(\$nb) 坐标不全，跳过比对" | tee -a "\$REPORT"
    fi
done

# 两个 MODEL 之间也比一次。几何理论上与速度模型无关，若这里出现差异，
# 说明 MODEL 确实会影响网格，mesh0 用哪个 MODEL 就不是无关紧要的选择了。
n2=\$(ls "$PROBE_DIR/coords_r1" 2>/dev/null | wc -l | tr -d ' ')
n4=\$(ls "$PROBE_DIR/coords_r2" 2>/dev/null | wc -l | tr -d ' ')
if [ "\$n2" -eq "$NPROC" ] && [ "\$n4" -eq "$NPROC" ]; then
    echo "" | tee -a "\$REPORT"
    echo "--- 跨 MODEL: s362ani(轮1) vs GLL(轮2) ---" | tee -a "\$REPORT"
    "$PYTHON_BIN" "$CHECK_PY" \\
        "$PROBE_DIR/coords_r1" "$PROBE_DIR/coords_r2" \\
        --nproc $NPROC --tol 0 2>&1 | tail -14 | tee -a "\$REPORT"
fi

echo "" | tee -a "\$REPORT"
echo "===== PROBE 完成: \$(date) =====" | tee -a "\$REPORT"
echo "报告: \$REPORT"
EOF
    } > "$JOBS_DIR/run_PROBE.slurm"
    chmod +x "$JOBS_DIR/run_PROBE.slurm"
    echo "✅ jobs/run_PROBE.slurm"
}

stage_probe() {
    local force_nodes="${1:-}"
    hr; echo "阶段 PROBE — mesher 对照实验"; hr
    [ -x "$TEST_ROOT/bin/xmeshfem3D" ] || die "缺少可执行文件，请先运行: $0 prepare"

    local wd="$PROBE_DIR/work"
    mkdir -p "$PROBE_DIR"
    setup_workdir "$wd"
    ln -sfn "$BASE_GLL_DIR" "$wd/DATA/GLL"
    echo "✅ 探测工作目录: $wd"

    local NPROC_TOTAL
    NPROC_TOTAL=$(( $(read_par NCHUNKS "$DATA_DIR/Par_file") \
                  * $(read_par NPROC_XI "$DATA_DIR/Par_file") \
                  * $(read_par NPROC_ETA "$DATA_DIR/Par_file") ))
    write_probe_script "$NPROC_TOTAL"

    submit_one PROBE "" "$force_nodes" || die "PROBE 提交失败"
    hr
    echo "四轮 mesher 约 5–10 分钟。完成后看:"
    echo "  cat $PROBE_DIR/report.txt"
    echo ""
    echo "换机架复测:  $0 probe <节点列表>"
    echo "  可用机架:  sinfo -h -N -p $SLURM_PARTITION -t idle -o '%N' | sed 's/n[0-9]*\$//' | sort | uniq -c"
    hr
}

# ---------------------------------------------------------------------------
# MESH0：唯一一次「用底图跑 mesher」，产出全流程的坐标基准
# ---------------------------------------------------------------------------
write_mesh0_script() {
    local NPROC="$1"
    {
    sbatch_header MESH0 "$SLURM_TIME_MESH"
    cat << EOF

set -u
ATTEMPT=\${HYB_ATTEMPT:-1}
echo "===== MESH0 开始: \$(date)  第 \$ATTEMPT 次尝试 ====="
echo "节点: \${SLURM_JOB_NODELIST:-<未知>}"

$(env_intel)

cd "$MESH0_DIR" || exit 1

# mesh0 只提供坐标，不需要正确的速度模型。用 MESH0_MODEL 而不是 Par_file 里的
# GLL，是因为本机只验证过 s362ani 这条路能把 mesher 跑干净（见配置项注释）。
sed -i "/^MODEL[[:space:]]*=/s|=.*|= $MESH0_MODEL|" DATA/Par_file
echo "基准网格 MODEL = $MESH0_MODEL"
cp DATA/Par_file DATA/STATIONS DATA/CMTSOLUTION OUTPUT_FILES/ 2>/dev/null

if [ "$MESH0_MODEL" = "GLL" ]; then
    n_gll=\$(ls DATA/GLL/proc*_reg1_vsv.bin 2>/dev/null | wc -l | tr -d ' ')
    echo "底图 GLL vsv 文件: \$n_gll / $NPROC"
    [ "\$n_gll" -eq "$NPROC" ] || { echo "❌ 底图不完整"; exit 1; }
fi

$(mesher_block MESH0 "$NPROC" "基准网格")

dt=\$(grep -i "the time step of the solver will be DT" OUTPUT_FILES/output_mesher.txt \\
      | tail -1 | sed 's/.*=//' | tr -d ' ')
echo "mesher 定的 DT = \${dt:-<未识别>}"
echo "\$dt" > OUTPUT_FILES/.dt
grep -A2 "crust mantle MPI" OUTPUT_FILES/output_mesher.txt | head -3

echo "\$(date)" > OUTPUT_FILES/.done
echo "===== MESH0 完成: \$(date) ====="

# 基准网格确认可用之后才提交各组。
# 不用 --dependency=afterok 是因为 MESH0 失败会连带把各组全部取消，而自动重投
# 产生的是新 JobID，各组的依赖仍指向已失败的旧作业，救不回来（实测各组全被
# CANCELLED）。改由 MESH0 自己在成功后提交，重投多少次都不影响。
echo "--- 提交各组正演 ---"
"$SELF" submit-runs
EOF
    } > "$JOBS_DIR/run_MESH0.slurm"
    chmod +x "$JOBS_DIR/run_MESH0.slurm"
    echo "✅ jobs/run_MESH0.slurm"
}

# ---------------------------------------------------------------------------
# 各组共用的作业脚本：建模型 → mesher → 坐标核对 → solver
#
# BASE 与三个混合组走完全相同的四个阶段，只在阶段 1 有别：BASE 的 DATA/GLL 就是
# model_updated 本身（prepare 时软链好），不需要构建。
#
# BASE 早期版本直接复用 mesh0 的 DATABASES_MPI 以省一次 mesher，现已取消。原因是
# mesh0 改用 MODEL=$MESH0_MODEL 建网格（那是本机唯一验证过能跑干净的配置），若 BASE
# 复用它，BASE 的网格就与其余组不同源，各组不可比。让 BASE 也跑一次自己的 GLL
# mesher 并接受同样的坐标核对，各组待遇才真正一致，代价只是约一分钟。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 参照组作业：原生 s362ani，复用 mesh0 的网格，只跑 ROUNDTRIP_EVENT
#
# mesh0 本来就是用 MODEL=s362ani 跑出来的，它的 DATABASES_MPI 就是原生 s362ani
# 网格，没有理由再跑一遍 mesher。本组与 S362ANI_c1 之差即「s362ani 提取成 GLL
# 再读回来」这一往返的全部代价——两者的弹性场理论上逐位相同，衰减也同为 1DREF，
# 所以波形上任何可见差异都来自往返本身。
# ---------------------------------------------------------------------------
write_ref_script() {
    local tag="$1" NPROC="$2"
    {
    sbatch_header "$tag" "$SLURM_TIME_RUN"
    cat << EOF

set -u
echo "===== $tag 开始: \$(date) ====="
echo "节点: \${SLURM_JOB_NODELIST:-<未知>}"

cd "$RUNS_DIR/$tag" || exit 1

[ -f "$MESH0_DIR/DATABASES_MPI/.mesh_ok" ] \\
    || { echo "❌ mesh0 没有 .mesh_ok 标记，基准网格未通过校验，请先跑通 MESH0"; exit 1; }

# ---- 阶段 1-3: 无（不建模型、不跑 mesher、无需坐标核对）----
# DATABASES_MPI 是指向 mesh0 的软链。正演 SIMULATION_TYPE=1 且 SAVE_FORWARD
# 为假时只读该目录，与其它组并发读是安全的。
sed -i "/^MODEL[[:space:]]*=/s|=.*|= $MESH0_MODEL|" DATA/Par_file
echo "本组 MODEL = ${MESH0_MODEL}（原生，不走 GLL）"
n_sd=\$(ls DATABASES_MPI/proc*_reg1_solver_data.bin 2>/dev/null | wc -l | tr -d ' ')
echo "复用 mesh0 网格: \$n_sd / $NPROC 个 proc"
[ "\$n_sd" -eq "$NPROC" ] || { echo "❌ mesh0 网格不完整"; exit 1; }

# solver 启动时会读 OUTPUT_FILES/addressing.txt（mesher 写的 rank 寻址表，
# 见 read_mesh_databases.F90）。本组不跑 mesher，必须从 mesh0 拷过来；
# 只拷 output_mesher.txt 会在 MPI 起来后立刻 Error opening addressing.txt。
mkdir -p OUTPUT_FILES
for f in addressing.txt output_mesher.txt values_from_mesher.h; do
    cp "$MESH0_DIR/OUTPUT_FILES/\$f" OUTPUT_FILES/ 2>/dev/null || true
done
[ -f OUTPUT_FILES/addressing.txt ] \\
    || { echo "❌ 缺少 OUTPUT_FILES/addressing.txt（应从 mesh0 拷贝）"; exit 1; }
echo "已从 mesh0 同步 mesher 侧车文件: addressing.txt + output_mesher.txt"
$(solver_loop_block "$tag" "$NPROC")
EOF
    } > "$JOBS_DIR/run_$tag.slurm"
    chmod +x "$JOBS_DIR/run_$tag.slurm"
    echo "✅ jobs/run_$tag.slurm  （参照组，1 个事件，复用 mesh0 网格）"
}

write_run_script() {
    local tag="$1" NPROC="$2"
    local mopt="" build_jobs="$BUILD_JOBS" salvus_extra=""
    [ "$tag" = "BASE" ] || mopt=$(model_opt "$tag")
    # Salvus 点云每个 worker 独立建 KDTree，并行度过高会 OOM
    if [ "$tag" = "SinoScope_sem_c1" ]; then
        build_jobs=2
        salvus_extra="--salvus-subsample 1"
    fi
    {
    sbatch_header "$tag" "$SLURM_TIME_RUN"
    cat << EOF

set -u
ATTEMPT=\${HYB_ATTEMPT:-1}
echo "===== $tag 开始: \$(date)  第 \$ATTEMPT 次尝试 ====="
echo "节点: \${SLURM_JOB_NODELIST:-<未知>}"

cd "$RUNS_DIR/$tag" || exit 1
cp DATA/Par_file DATA/STATIONS DATA/CMTSOLUTION OUTPUT_FILES/ 2>/dev/null

[ -f "$MESH0_DIR/DATABASES_MPI/.mesh_ok" ] \\
    || { echo "❌ mesh0 没有 .mesh_ok 标记，基准网格未通过校验，请先跑通 MESH0"; exit 1; }
EOF

    if [ "$tag" = "BASE" ]; then
    cat << EOF

# ---- 阶段 1: 无 ----
# BASE 的 DATA/GLL 直接软链到 model_updated，零改动对照，不构建任何东西
n_gll=\$(ls DATA/GLL/proc*_reg1_vsv.bin 2>/dev/null | wc -l | tr -d ' ')
echo "底图 GLL vsv 文件: \$n_gll / $NPROC （直接使用 model_updated，无改动）"
[ "\$n_gll" -eq "$NPROC" ] || { echo "❌ 底图不完整"; exit 1; }
EOF
    else
    cat << EOF

# ---- 阶段 1: 以 mesh0 的实测坐标构建混合模型 ----
# 插值基准取自本次流程刚生成的 solver_data.bin，而不是任何预先准备的网格，
# 「坐标对不上但 NSPEC 相同」这类静默错误在构造上就不会发生。
n_gll=\$(ls DATA/GLL/proc*_reg1_vsv.bin 2>/dev/null | wc -l | tr -d ' ')
if [ "\$n_gll" -eq "$NPROC" ] && [ "\${FORCE_BUILD_GLL:-0}" != "1" ]; then
    echo "--- 混合模型已存在 (\$n_gll)，跳过构建 ---"
else
    echo "--- 构建混合模型 ---"
    t0=\$(date +%s)
    rm -rf DATA/GLL && mkdir -p DATA/GLL
    "$PYTHON_BIN" "$BUILD_PY" \\
        --mesh-dir "$MESH0_DIR/DATABASES_MPI" \\
        --base-gll-dir "$BASE_GLL_DIR" \\
        --output-dir DATA/GLL \\
        --base-from-mesh \\
        $mopt $salvus_extra \\
        --region $REGION_LATLON \\
        --horiz-taper $HORIZ_TAPER \\
        --depth-range $DEPTH_RANGE \\
        --depth-taper $DEPTH_TAPER \\
        --depth-min-mode $DEPTH_MIN_MODE \\
        --moho-offset $MOHO_OFFSET \\
        --crust1-bnds "$CODE_DIR/DATA/crust1.0/crust1.bnds" \\
        --nproc $NPROC \\
        --jobs $build_jobs
    st=\$?
    echo "建模型退出码=\$st 耗时=\$(( (\$(date +%s) - t0) / 60 )) 分钟"
    [ \$st -eq 0 ] || { echo "❌ 混合模型构建失败"; exit 1; }
    n_gll=\$(ls DATA/GLL/proc*_reg1_vsv.bin 2>/dev/null | wc -l | tr -d ' ')
    [ "\$n_gll" -eq "$NPROC" ] || { echo "❌ DATA/GLL 不完整 (\$n_gll/$NPROC)"; exit 1; }
fi
EOF
    fi

    cat << EOF

# ---- 阶段 2: mesher (MODEL=GLL) ----
$(env_intel)

$(mesher_block "$tag" "$NPROC" "本组网格")

# ---- 阶段 3: 坐标核对（关键关卡）----
# 本组的模型是按 mesh0 的坐标插出来的，若本次 mesher 没有复现 mesh0 的坐标，
# 速度值就落在了错误的位置。ibool 要求逐位相同，x/y/z 允许 $COORD_TOL 相对误差。
# 通过后写 .coord_ok：多事件模式下作业会被重投多次，而网格一旦定下来就不再变，
# 每次都重扫 144 个 proc 纯属浪费。mesher 重跑时 .mesh_ok 连同它一起被清掉。
if [ -f DATABASES_MPI/.coord_ok ] && [ "\${FORCE_COORD:-0}" != "1" ]; then
    echo "--- 坐标核对: 已通过（见 DATABASES_MPI/.coord_ok），跳过 ---"
else
    echo "--- 坐标核对: 本组 mesher vs mesh0 ---"
    "$PYTHON_BIN" "$CHECK_PY" "$MESH0_DIR/DATABASES_MPI" DATABASES_MPI \\
        --nproc $NPROC --tol $COORD_TOL
    chk=\$?
    if [ \$chk -ne 0 ]; then
        echo "❌ 坐标核对未通过，不启动 solver（结果不可用）"
        exit 1
    fi
    echo "\$(date) tol=$COORD_TOL" > DATABASES_MPI/.coord_ok
fi

$(solver_loop_block "$tag" "$NPROC")
EOF
    } > "$JOBS_DIR/run_$tag.slurm"
    chmod +x "$JOBS_DIR/run_$tag.slurm"
    echo "✅ jobs/run_$tag.slurm  （$(events_of "$tag" | grep -c .) 个事件）"
}

# ---------------------------------------------------------------------------
# 阶段 4：逐个事件跑 solver
#
# 网格与模型都与震源无关，所以一组只网格化一次，solver 循环所有事件。相比
# 「每个事件从头跑一遍」省掉 9 次 mesher 和 9 次建模型，也确保同一组的 10 个
# 事件用的是逐位相同的模型——否则组内还多一个变量。
#
# 两处保护：
#   断点续跑  每个事件算完写 sac/<TAG>/<EVENT>/.done，重投时已完成的直接跳过。
#             30 个事件串起来约 35 小时，中途被抢占或节点故障是常态，不能从头再来。
#   时限保护  开跑前估算剩余墙钟够不够再跑一个事件，不够就主动收尾并重投，
#             而不是让 Slurm 在第 8 个事件跑到一半时把作业砍掉——那会留下一份
#             写了一半的 SAC，比没有更麻烦。
# ---------------------------------------------------------------------------
solver_loop_block() {
    local tag="$1" NPROC="$2"
    local evs; evs=$(events_of "$tag")
    cat << EOF

# ---- 阶段 4: solver × $(echo "$evs" | grep -c .) 个事件 ----
$(env_intel)

dt=\$(grep -i "the time step of the solver will be DT" OUTPUT_FILES/output_mesher.txt \\
      | tail -1 | sed 's/.*=//' | tr -d ' ')
echo "mesher 定的 DT = \${dt:-<未识别>}"
echo "\$dt" > OUTPUT_FILES/.dt

EVENTS_TODO="$(echo "$evs" | tr '\n' ' ')"
LAST_SEC=0
JOB_T0=\$(date +%s)

# 作业剩余秒数。squeue 的 %L 形如 [dd-]hh:mm:ss，解析失败时返回一个大数，
# 宁可让 Slurm 去砍也不要因为解析问题白白提前退出。
remaining_sec() {
    local L d h m s
    L=\$(squeue -h -j "\${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | tr -d ' ')
    [ -n "\$L" ] || { echo 999999; return; }
    d=0
    case "\$L" in *-*) d=\${L%%-*}; L=\${L#*-} ;; esac
    case "\$L" in
        *:*:*) h=\${L%%:*}; m=\$(echo "\$L" | cut -d: -f2); s=\${L##*:} ;;
        *:*)   h=0; m=\${L%%:*}; s=\${L##*:} ;;
        *)     h=0; m=0; s=\$L ;;
    esac
    echo \$(( 10#\$d * 86400 + 10#\$h * 3600 + 10#\$m * 60 + 10#\$s ))
}

n_ok=0; n_skip=0
for ev in \$EVENTS_TODO; do
    outdir="$SAC_DIR/$tag/\$ev"
    if [ -f "\$outdir/.done" ] && [ "\${FORCE_SOLVER:-0}" != "1" ]; then
        echo "--- \$ev 已完成，跳过 ---"
        n_skip=\$(( n_skip + 1 ))
        continue
    fi

    # 上一个事件的耗时是下一个的最好估计；还没跑过时按 2 小时保守估
    need=\$(( LAST_SEC > 0 ? LAST_SEC * 12 / 10 + 600 : 7200 ))
    left=\$(remaining_sec)
    if [ "\$left" -lt "\$need" ]; then
        echo "⏸  剩余墙钟 \${left}s 不足以再跑一个事件（需约 \${need}s），主动收尾"
        echo "   已完成 \$n_ok 个（另跳过 \$n_skip 个），提交后继作业接着跑"
        "$SELF" resubmit "$tag" || echo "   ⚠️  自动重投失败，请手工执行: $SELF resubmit $tag"
        exit 0
    fi

    echo "===== 事件 \$ev  开始 \$(date)  剩余墙钟 \${left}s ====="
    cp "$EVENTS_DIR/CMTSOLUTION_\$ev" DATA/CMTSOLUTION \\
        || { echo "❌ 缺少震源文件 CMTSOLUTION_\$ev"; exit 1; }
    # 上一个事件的 SAC 必须清干净，否则本次若中途失败会把旧波形当成新结果收走
    rm -f OUTPUT_FILES/*.sac
    cp DATA/Par_file DATA/STATIONS DATA/CMTSOLUTION OUTPUT_FILES/ 2>/dev/null

    t0=\$(date +%s)
    mpirun -np $NPROC ./bin/xspecfem3D
    st=\$?
    LAST_SEC=\$(( \$(date +%s) - t0 ))
    echo "solver 退出码=\$st 耗时=\$(( LAST_SEC / 60 )) 分钟"
    [ \$st -eq 0 ] || { echo "❌ $tag / \$ev solver 失败，见 OUTPUT_FILES/output_solver.txt"; exit 1; }

    n_sac=\$(ls OUTPUT_FILES/*.sac 2>/dev/null | wc -l | tr -d ' ')
    echo "SAC 地震图: \$n_sac 个"
    [ "\$n_sac" -eq 0 ] && { echo "❌ \$ev 未生成地震图"; exit 1; }

    mkdir -p "\$outdir"
    mv OUTPUT_FILES/*.sac "\$outdir/"
    cp OUTPUT_FILES/output_solver.txt DATA/CMTSOLUTION "\$outdir/" 2>/dev/null
    echo "\$(date) nsac=\$n_sac sec=\$LAST_SEC" > "\$outdir/.done"
    n_ok=\$(( n_ok + 1 ))
    echo "===== 事件 \$ev  完成 -> sac/$tag/\${ev}（\${n_sac} 个 SAC）====="
done

echo "\$(date) done=\$n_ok skipped=\$n_skip" > OUTPUT_FILES/.done
echo "===== $tag 全部完成: \$(date)  本次算了 \$n_ok 个事件，跳过 \$n_skip 个 ====="
EOF
}

# ============================================================================
# submit
# ============================================================================
LAST_RACK=""

# 提交一组。$2 为要避开的机架，$3 为显式指定的节点列表（优先于自动挑选）。
submit_one() {
    local tag="$1" skip="${2:-}" force_nodes="${3:-}"
    [ -f "$JOBS_DIR/run_$tag.slurm" ] || die "缺少 jobs/run_$tag.slurm"
    local jid pick nodes opts=""
    LAST_RACK=""

    if [ -n "$force_nodes" ]; then
        LAST_RACK=$(rack_of "${force_nodes%%,*}")
        # 手工指定时仍要拒绝「已占用 / 已预约」节点，否则一提交就 Priority 挂起
        local free
        free=$(allocatable_nodes "$force_nodes")
        if [ "$(echo "$free" | grep -c .)" -lt "$SLURM_NODES" ]; then
            echo "❌ $tag: 指定节点并非全部可立刻开跑（需纯 IDLE，不能是 allocated/PLANNED）"
            echo "   请求: $force_nodes"
            echo "   当前可开跑: $(echo "$free" | paste -sd, - || echo '(无)')"
            echo "   请换一批: $SELF picknodes   或   sinfo -N -t idle -p $SLURM_PARTITION"
            return 1
        fi
        opts="--nodelist=$force_nodes"
        info "$tag  指定节点 ${force_nodes}（机架 ${LAST_RACK}）"
    elif [ "$SAME_RACK" = "1" ]; then
        if pick=$(pick_rack_nodes "$SLURM_NODES" "$skip"); then
            LAST_RACK="${pick%% *}"; nodes="${pick#* }"
            opts="--nodelist=$nodes"
            info "$tag  机架 $LAST_RACK  节点 $nodes"
        elif pick=$(pick_rack_nodes "$SLURM_NODES" ""); then
            # 其它机架还有纯 IDLE 的 4 节点：钉过去立刻能跑，不要钉在已占用节点上排队
            LAST_RACK="${pick%% *}"; nodes="${pick#* }"
            opts="--nodelist=$nodes"
            info "$tag  机架 ${LAST_RACK}（另选空闲架）  节点 $nodes"
        else
            # 全分区凑不出 4 个同架纯 IDLE：绝不再 --nodelist 钉死忙节点（会 Priority 挂一夜）。
            # 默认改为不绑节点交给调度；REQUIRE_SAME_RACK=1 则拒绝提交等人工指定。
            if [ "${REQUIRE_SAME_RACK:-0}" = "1" ]; then
                echo "❌ $tag: 凑不出 $SLURM_NODES 个同机架纯 IDLE 节点（已排除 PLANNED）"
                echo "   指定重投: $SELF resubmit $tag <节点1,节点2,节点3,节点4>"
                return 1
            fi
            echo "⚠️  $tag: 暂无同机架纯 IDLE 四节点，不绑 --nodelist，交给调度器分配"
            echo "   （可能跨机架；若 mesher 报 face flag / ineighbour，再用同架空闲节点 resubmit）"
        fi
    fi

    # shellcheck disable=SC2086
    jid=$(sbatch --parsable $opts "$JOBS_DIR/run_$tag.slurm") || return 1
    info "run_$tag  -> JobID $jid"
    echo "$tag $jid" >> "$TEST_ROOT/logs/jobids.txt"
    LAST_JID="$jid"
}

# 提交各组。由 MESH0 作业在基准网格通过校验后调用，也可在网格已就绪时手工调用。
stage_submit_runs() {
    hr; echo "提交各组正演"; hr
    [ -f "$MESH0_DIR/DATABASES_MPI/.mesh_ok" ] \
        || die "mesh0 没有 .mesh_ok 标记，基准网格未通过校验，不能提交各组"

    local used_racks="" tag
    for tag in $RUN_TAGS $REF_TAG; do
        submit_one "$tag" "$used_racks" || die "$tag 提交失败"
        [ -n "$LAST_RACK" ] || continue
        # 机架数少于组数时会复用；复用到已用过的机架说明轮完一圈，清空重新计数
        case " $used_racks " in
            *" $LAST_RACK "*) used_racks="$LAST_RACK" ;;
            *)                used_racks="$used_racks $LAST_RACK" ;;
        esac
    done
    hr
}

stage_submit() {
    hr; echo "阶段 SUBMIT"; hr
    [ -d "$JOBS_DIR" ] || die "作业脚本不存在，请先运行: $0 prepare"

    : > "$TEST_ROOT/logs/jobids.txt"

    if [ -f "$MESH0_DIR/DATABASES_MPI/.mesh_ok" ] && [ "${FORCE_MESH:-0}" != "1" ]; then
        echo "基准网格已通过校验（见 mesh0/DATABASES_MPI/.mesh_ok），跳过 MESH0"
        cat "$MESH0_DIR/DATABASES_MPI/.mesh_ok" | sed 's/^/   /'
        stage_submit_runs
    else
        submit_one MESH0 "" || die "MESH0 提交失败"
        echo "   各组将由 MESH0 在基准网格通过校验后自行提交"
    fi

    hr
    echo "✅ 已提交，JobID 记录于 logs/jobids.txt"
    echo ""
    echo "监控:  $0 status"
    echo "       squeue -u $(whoami)"
    echo ""
    echo "单组重投:  $0 resubmit <TAG>"
    hr
}

# ============================================================================
# status
# ============================================================================
stage_status() {
    load_events
    hr; echo "阶段 STATUS — $TEST_ROOT"; hr
    squeue -u "$(whoami)" -o "%.10i %.14j %.10T %.10M %.6D %R" 2>/dev/null || echo "(squeue 不可用)"

    local n0
    n0=$(ls "$MESH0_DIR"/DATABASES_MPI/proc*_reg1_solver_data.bin 2>/dev/null | wc -l | tr -d ' ')
    hr
    if [ -f "$MESH0_DIR/DATABASES_MPI/.mesh_ok" ]; then
        echo "基准网格 mesh0: $n0 / 144 个 proc  ✅ 已通过校验"
        sed 's/^/   /' "$MESH0_DIR/DATABASES_MPI/.mesh_ok"
    else
        echo "基准网格 mesh0: $n0 / 144 个 proc  ⚠️  无 .mesh_ok 标记，未通过校验"
        [ "$n0" -gt 0 ] && echo "   有文件但没标记 = 上次 mesher 写完了却没通过界面校验，这份网格不可用"
    fi

    hr; printf "%-14s %6s %11s %8s %11s %s\n" \
        "TAG" "GLL" "solver_data" "事件" "DT" "状态"; hr
    local dts="" tag
    for tag in $RUN_TAGS $REF_TAG; do
        local rd="$RUNS_DIR/$tag"
        [ -d "$rd" ] || { printf "%-14s %s\n" "$tag" "(未建立)"; continue; }
        local n_gll n_sd n_done n_want state dt
        n_gll=$(ls "$rd"/DATA/GLL/proc*_reg1_vsv.bin 2>/dev/null | wc -l | tr -d ' ')
        n_sd=$(ls "$rd"/DATABASES_MPI/proc*_reg1_solver_data.bin 2>/dev/null | wc -l | tr -d ' ')
        n_want=$(events_of "$tag" | grep -c .)
        n_done=$(ls -d "$SAC_DIR/$tag"/*/.done 2>/dev/null | wc -l | tr -d ' ')
        dt=$(cat "$rd/OUTPUT_FILES/.dt" 2>/dev/null || echo "-")
        [ "$dt" != "-" ] && dts="$dts $dt"
        if   [ "$n_done" -ge "$n_want" ];  then state="✅ 全部完成"
        elif [ "$n_done" -gt 0 ];          then state="🔄 正在跑事件"
        elif [ "$n_sd" -eq 144 ];          then state="🔄 已网格化"
        elif [ "$n_gll" -eq 144 ];         then state="⏳ 模型就绪"
        else                                    state="⏳ 等待"
        fi
        is_ref "$tag" && n_gll="-"
        printf "%-14s %6s %11s %8s %11s %s\n" \
            "$tag" "$n_gll" "$n_sd" "$n_done/$n_want" "$dt" "$state"
    done
    hr
    # 各组的时间步必须相同，否则波形不在同一采样上，misfit 不可直接比较
    local n_uniq
    n_uniq=$(echo "$dts" | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l | tr -d ' ')
    if [ "$n_uniq" -gt 1 ]; then
        echo "⚠️  各组 DT 不一致（$n_uniq 种取值），对比前需重采样"
    elif [ "$n_uniq" -eq 1 ]; then
        echo "✅ 各组 DT 一致"
    fi

    # 逐事件明细：哪一组卡在哪个事件上，一眼可见
    hr; printf "%-16s" "事件"
    for tag in $RUN_TAGS $REF_TAG; do printf " %-13s" "$tag"; done; echo; hr
    local ev
    for ev in $EVENTS; do
        printf "%-16s" "$ev"
        for tag in $RUN_TAGS $REF_TAG; do
            if [ -f "$SAC_DIR/$tag/$ev/.done" ]; then
                printf " %-13s" "$(ls "$SAC_DIR/$tag/$ev"/*.sac 2>/dev/null | wc -l | tr -d ' ')"
            elif events_of "$tag" | grep -qx "$ev"; then
                printf " %-13s" "-"
            else
                printf " %-13s" "n/a"
            fi
        done
        echo
    done
    hr
}

# ============================================================================
# collect
# ============================================================================
# SAC 由作业直接写进 sac/<TAG>/<EVENT>/，这里只做完整性核查。
# 核查的核心是「同一事件下各组的台站集合完全相同」——只要有一组少了几个台站，
# 那几条记录在跨组比较时就变成了缺项，misfit 统计会被悄悄改变。
stage_collect() {
    load_events
    hr; echo "阶段 COLLECT — 完整性核查"; hr
    [ -d "$SAC_DIR" ] || die "还没有任何结果: $SAC_DIR"

    local ev tag n total=0 n_full=0 n_partial=0 n_none=0
    for ev in $EVENTS; do
        local counts="" first="" same=1 sum=0
        for tag in $RUN_TAGS; do
            n=$(ls "$SAC_DIR/$tag/$ev"/*.sac 2>/dev/null | wc -l | tr -d ' ')
            counts="$counts $tag:$n"
            total=$(( total + n )); sum=$(( sum + n ))
            [ -z "$first" ] && first="$n"
            [ "$n" = "$first" ] || same=0
        done
        if [ "$sum" -eq 0 ]; then
            echo "⏳ $ev  尚未开始"
            n_none=$(( n_none + 1 ))
        elif [ "$same" = "1" ]; then
            echo "✅ $ev  各组均 $first 个 SAC"
            n_full=$(( n_full + 1 ))
        else
            # 台站数不一致最要命：跨组比较时缺项会悄悄改变 misfit 统计
            echo "⚠️  $ev 各组台站数不一致 —$counts"
            n_partial=$(( n_partial + 1 ))
        fi
    done
    echo "   齐全 $n_full 个，不齐 $n_partial 个，未开始 $n_none 个"

    # 参照组只有一个事件，单列
    n=$(ls "$SAC_DIR/$REF_TAG/$ROUNDTRIP_EVENT"/*.sac 2>/dev/null | wc -l | tr -d ' ')
    echo "   $REF_TAG / $ROUNDTRIP_EVENT: $n 个 SAC"
    total=$(( total + n ))

    hr
    echo "SAC 总数: $total    目录: $SAC_DIR/<TAG>/<EVENT>/"
    echo ""
    echo "对比要点："
    echo "  1. $REF_TAG 与 S362ANI_c1（同为 ${ROUNDTRIP_EVENT}）之差 = GLL 往返的全部代价。"
    echo "     它理论上应接近零；若不接近，后面所有组间差异的下限就是它，先查这里"
    echo "  2. BASE 与原作者 sac/ 应几乎一致 —— 不一致说明链路本身有问题，其余组无意义"
    echo "  3. 四个 _c1 组共用 CRUST1.0 地壳，只在 Moho 以下不同，可直接互比"
    echo "  4. BASE 与 FWEA23_c1 之差 = 换掉地壳（FWEA23 FWI 地壳 → CRUST1.0）的代价"
    echo "  5. 排名要在 $N_EVENTS 个事件上一致才算稳。单个事件上的胜负受震源机制与路径影响很大，"
    echo "     深源（542/601/367 km）主要检验地幔过渡带，浅源主要检验地壳与岩石圈"
    hr
}

# ============================================================================
usage() {
    cat << EOF
用法: $0 {prepare|probe [节点列表]|submit|status|collect|events|resubmit <TAG>}

  prepare   校验路径、Python 与全部震源、编译一次、建目录、生成作业脚本
            FORCE_BUILD=1  强制重新编译
  probe     mesher 对照实验：同一批节点上跑 s362ani / GLL 各两轮，
            分离「MODEL 是否影响 MPI 界面错误」与「坐标是否可复现」两个问题
            可跟节点列表指定机架，用于排查节点硬件问题，例如:
            $0 probe c12r4n02,c12r4n03,c12r4n04,c12r4n05
  submit    提交 MESH0；基准网格通过校验后由 MESH0 自行提交各组
            基准网格已有 .mesh_ok 则直接提交各组
  status    查看队列、各组进度与逐事件明细
  collect   核查 sac/<TAG>/<EVENT>/ 的 SAC 是否齐全且各组台站数一致
  events    列出震源清单
  resubmit  单组重投，优先同机架；也可显式指定:
            $0 resubmit MESH0 c10r3n02,c10r3n03,c10r3n04,c10r3n05
            凑不齐同机架时默认跨机架提交（有失败风险）。
            REQUIRE_SAME_RACK=1 可禁止降级。已完成的事件会跳过

对照组:
  BASE          model_updated 原样，零改动
  S362ANI_c1    CRUST1.0 地壳 + s362ani 地幔（空对照）
  FWEA23_c1     CRUST1.0 地壳 + FWEA23 地幔
  EARA2024_c1   CRUST1.0 地壳 + EARA2024 地幔
  SinoScope_c1      CRUST1.0 地壳 + SinoScope1.0 地幔（公开 1° H5）
  SinoScope_sem_c1  CRUST1.0 地壳 + SinoScope Salvus mesh.h5（作者原生 SEM）
  S362ANI_ref       原生 s362ani，复用 mesh0 网格，只跑 $ROUNDTRIP_EVENT
  前五组共用同一套 CRUST1.0 地壳（从 mesher 的 solver_data.bin 反算，不经插值），
  Moho 以下才换成各自的地幔，因此组间差异可干净归因到地幔模型。
  S362ANI_ref 与 S362ANI_c1 之差是 GLL 往返的代价，它是一切组间差异的分辨率下限。

流程:
  MESH0 用 MODEL=$MESH0_MODEL 跑一次 mesher，产出全流程唯一的坐标基准
  mesh0/DATABASES_MPI。各模型组走完全相同的四个阶段：建模型（BASE 无）→
  MODEL=GLL 的 mesher → 核对坐标是否复现 mesh0（ibool 逐位相同、x/y/z 相对误差
  <= ${COORD_TOL}）→ 逐个事件跑 solver。BASE 不复用 mesh0 的网格，各组待遇一致才可比。
  S362ANI_ref 例外：它就是 mesh0 那套网格，直接跑 solver。

多事件:
  网格与模型与震源无关，故一组只网格化一次，solver 循环全部事件。每个事件
  单独写 sac/<TAG>/<EVENT>/，算完打 .done 标记。作业在剩余墙钟不够再跑一个
  事件时会主动收尾并自动重投，不会留下写了一半的结果。
  FORCE_SOLVER=1 可强制重算已完成的事件。

mesher 通过判据（比「文件数够了」严格）:
  退出码 0 + solver_data.bin 齐全 + 零界面告警 + 零界面错误 → 写 .mesh_ok
  跳过逻辑只认 .mesh_ok。任何一项不过就清空 DATABASES_MPI 再换节点重投，
  避免「写完了但校验没过」的网格被当成好网格复用。

关键路径:
  数据（Par_file/STATIONS/constants.h.in）: $DATA_DIR
  震源清单:  $EVENTS_LIST  （本地 package_eastasia_model_30.sh 生成）
  震源文件:  $EVENTS_DIR/CMTSOLUTION_<EVENT>
  台站文件:  $EVENTS_ROOT/STATIONS → 部署到 DATA/STATIONS
  BASE 底图: $BASE_GLL_DIR
  原始模型:  $MODELS_DIR
  替换区:    lat/lon $REGION_LATLON, 水平 taper ${HORIZ_TAPER}°
             上界 ${DEPTH_MIN_MODE}（偏移 ${MOHO_OFFSET} km），下界 ${DEPTH_RANGE##* } km
             深度 taper $DEPTH_TAPER km
  测试根目录: $TEST_ROOT
EOF
}

case "${1:-}" in
    prepare)     stage_prepare ;;
    probe)       stage_probe "${2:-}" ;;
    submit)      stage_submit ;;
    submit-runs) stage_submit_runs ;;
    status)   stage_status ;;
    collect)  stage_collect ;;
    resubmit) [ -n "${2:-}" ] || die "用法: $0 resubmit <TAG> [节点列表]"
              # 重投前按当前脚本重写该组 slurm，避免 jobs/ 里残留旧逻辑
              load_events
              _nproc=$(( $(read_par NCHUNKS "$DATA_DIR/Par_file") \
                       * $(read_par NPROC_XI "$DATA_DIR/Par_file") \
                       * $(read_par NPROC_ETA "$DATA_DIR/Par_file") ))
              case "$2" in
                  MESH0) write_mesh0_script "$_nproc" ;;
                  PROBE) write_probe_script "$_nproc" ;;
                  "$REF_TAG") write_ref_script "$2" "$_nproc" ;;
                  *)     write_run_script "$2" "$_nproc" ;;
              esac
              submit_one "$2" "" "${3:-}" ;;
    events)   load_events
              echo "$EVENTS" | nl -w3 -s'  '
              echo "共 $N_EVENTS 个事件，参照组用 $ROUNDTRIP_EVENT" ;;
    # 供作业脚本内部换节点重投时调用
    picknodes) _p=$(pick_rack_nodes "$SLURM_NODES" "${2:-}") \
                  || _p=$(pick_rack_nodes "$SLURM_NODES" "") || _p=""
               [ -n "$_p" ] && echo "${_p#* }" ;;
    rackof)    rack_of "${2:-}" ;;
    *)        usage; exit 1 ;;
esac
