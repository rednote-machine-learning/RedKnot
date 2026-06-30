#!/bin/bash
# watchdog_pro.sh —— DeepSeek-V4-Flash-Base server 看门狗
#
# 用途: 防止推理任务挂掉后 GPU 闲置。每隔 INTERVAL 秒检查一次 server 进程:
#   - 进程在跑           -> 跳过
#   - 进程不在(挂了/退出) -> 自动重启 DeepSeek-V4-Flash-Base 推理 server
#
# 说明: 本脚本只做"重启推理服务", 不做 GPU 空转占卡 (gpu_burn 之类滥用资源的行为)。
#
# 用法:
#   nohup setsid bash server/watchdog_pro.sh > /tmp/watchdog_fb.log 2>&1 &
# 停止:
#   touch /tmp/watchdog_fb.stop   # 看门狗下个周期会优雅退出

set -uo pipefail

INTERVAL="${WATCHDOG_INTERVAL:-3600}"          # 检查间隔, 默认 1 小时
SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"
START_SCRIPT="$SERVER_DIR/start_server_flashbase.sh"
SERVER_LOG=/tmp/fb_server.log
PROC_PATTERN="sglang.launch_server.*DeepSeek-V4-Flash-Base"
STOP_FLAG=/tmp/watchdog_fb.stop
HEALTH_URL="http://127.0.0.1:31999/health"

log() { echo "[$(date '+%F %T')] $*"; }

server_alive() {
    # 1) 进程存在 ; 2) (可选) health 端口可达
    if ! pgrep -f "$PROC_PATTERN" >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

restart_server() {
    log "server 进程不存在, 重启 DeepSeek-V4-Flash-Base 推理服务..."
    # 清理可能残留的半死进程
    pkill -9 -f "$PROC_PATTERN" 2>/dev/null || true
    sleep 5
    setsid bash -c "'$START_SCRIPT' > '$SERVER_LOG' 2>&1; echo SERVER_EXIT=\$? >> '$SERVER_LOG'" \
        < /dev/null > /dev/null 2>&1 &
    disown || true
    log "已发起重启 (日志: $SERVER_LOG)"
}

log "watchdog 启动, 检查间隔=${INTERVAL}s, 监控进程=[$PROC_PATTERN]"
while true; do
    if [[ -f "$STOP_FLAG" ]]; then
        log "检测到停止标志 $STOP_FLAG, 看门狗退出 (不影响 server)"
        rm -f "$STOP_FLAG"
        exit 0
    fi

    if server_alive; then
        log "server 进程在跑, 跳过本轮"
    else
        restart_server
    fi

    sleep "$INTERVAL"
done
