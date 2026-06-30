#!/bin/bash
# launch_pro_bg.sh —— 后台启动 Pro server 的封装, 立即返回不阻塞调用方
SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG=/tmp/pro_server.log
setsid bash -c "'$SERVER_DIR/start_server_pro.sh' > '$LOG' 2>&1; echo SERVER_EXIT=\$? >> '$LOG'" </dev/null >/dev/null 2>&1 &
disown
echo "Pro server 后台启动完成 (日志: $LOG)"
exit 0
