#!/usr/bin/env bash
# 训练接力监控：等待 GPU0/GPU1 上本项目的训练结束后，自动启动下一轮训练。
#   GPU0 训练结束 -> bash scripts/train_P5_2.sh
#   GPU1 训练结束 -> bash scripts/train_P6_2.sh
#
# 启动方式（后台常驻，日志写入 logs/gpu_watchdog.log）:
#   cd /data1/workspace/DIF-Gaussian-up-new
#   nohup bash scripts/watch_and_chain.sh > /dev/null 2>&1 &
#
# 只做一次检测（不启动监控，用于验证识别逻辑）:
#   bash scripts/watch_and_chain.sh --check
#
# 停止监控:
#   kill $(cat /tmp/difg_gpu_watchdog.pid)

set -u

PROJECT_DIR=/data1/workspace/DIF-Gaussian-up-new
SCRIPT_GPU0=$PROJECT_DIR/scripts/train_P5_2.sh
SCRIPT_GPU1=$PROJECT_DIR/scripts/train_P6_2.sh
LOG_FILE=$PROJECT_DIR/logs/gpu_watchdog.log
PID_FILE=/tmp/difg_gpu_watchdog.pid

POLL_INTERVAL=60           # 轮询间隔（秒）
GRACE_AFTER_EXIT=120       # 检测到结束后再观察的缓冲秒数（等端口/显存释放，避免把瞬间空档误判为结束）
LAUNCH_VERIFY_TIMEOUT=900  # 执行后续脚本后，等待新训练进程出现在 GPU 上的最长时间（秒）
LAUNCH_VERIFY_STEP=30

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

# 防止重复启动监控器
if [[ -f $PID_FILE ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    log "watchdog 已在运行 (pid $(cat "$PID_FILE"))，本次退出"
    exit 0
fi
trap 'rm -f "$PID_FILE"' EXIT
echo $$ > "$PID_FILE"

# 判断指定 GPU 上是否仍有本项目的训练进程。
# 判据: 进程占用该 GPU 显存 + 工作目录是本项目 + 命令行含 code/train.py。
# 这样可排除 IDE/内核等其他占用显存的进程。
gpu_training_running() {
    local gpu_id=$1 pid cwd
    while IFS=, read -r pid _mem; do
        pid=${pid// /}
        [[ $pid =~ ^[0-9]+$ ]] || continue
        [[ -d /proc/$pid ]] || continue
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
        [[ $cwd == "$PROJECT_DIR" ]] || continue
        if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'code/train.py'; then
            return 0
        fi
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$gpu_id" 2>/dev/null)
    return 1
}

watch_gpu() {
    local gpu_id=$1 next_script=$2 tag=$3
    local seen_running=false idle_rounds=0
    log "[$tag] 开始监控 GPU$gpu_id（每 ${POLL_INTERVAL}s 轮询），结束后将执行: $next_script"
    while true; do
        if gpu_training_running "$gpu_id"; then
            seen_running=true
            idle_rounds=0
        elif $seen_running; then
            log "[$tag] GPU$gpu_id 训练进程已结束，等待 ${GRACE_AFTER_EXIT}s 后复核 ..."
            sleep "$GRACE_AFTER_EXIT"
            if gpu_training_running "$gpu_id"; then
                log "[$tag] 复核时 GPU$gpu_id 又出现训练进程（可能是新任务），继续监控 ..."
                seen_running=true
                continue
            fi
            log "[$tag] 确认 GPU$gpu_id 空闲，执行 $next_script"
            (cd "$PROJECT_DIR" && bash "$next_script") >> "$LOG_FILE" 2>&1
            local waited=0 appeared=false
            while (( waited < LAUNCH_VERIFY_TIMEOUT )); do
                if gpu_training_running "$gpu_id"; then appeared=true; break; fi
                sleep "$LAUNCH_VERIFY_STEP"
                waited=$((waited + LAUNCH_VERIFY_STEP))
            done
            if $appeared; then
                log "[$tag] 已确认 $next_script 在 GPU$gpu_id 上启动成功，本条监控链完成"
            else
                log "[$tag] 警告：$next_script 已执行，但 ${LAUNCH_VERIFY_TIMEOUT}s 内未在 GPU$gpu_id 检测到训练进程，请人工检查（可能端口冲突，报错见 logs/<新run>/train.log）"
            fi
            return
        else
            idle_rounds=$((idle_rounds + 1))
            if (( idle_rounds % 30 == 0 )); then
                log "[$tag] GPU$gpu_id 上尚未检测到本项目的训练进程，继续等待 ..."
            fi
        fi
        sleep "$POLL_INTERVAL"
    done
}

# 只检测一次并退出（验证模式）
if [[ ${1:-} == --check ]]; then
    for g in 0 1; do
        if gpu_training_running "$g"; then
            echo "GPU$g: 检测到本项目的训练进程在运行"
        else
            echo "GPU$g: 未检测到本项目的训练进程"
        fi
    done
    exit 0
fi

log "===== 训练接力监控启动 (pid $$) ====="
watch_gpu 0 "$SCRIPT_GPU0" "GPU0->P5" &
watch_gpu 1 "$SCRIPT_GPU1" "GPU1->P6" &
wait
log "===== 两条监控链均已完成，watchdog 退出 ====="
