#!/bin/bash
# launch_flashbase_bg.sh —— 后台启动 Flash-Base server, 立即返回
SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG=/tmp/fb_server.log
setsid bash -c "'$SERVER_DIR/start_server_flashbase.sh' > '$LOG' 2>&1; echo SERVER_EXIT=\$? >> '$LOG'" </dev/null >/dev/null 2>&1 &
disown
echo "Flash-Base server 后台启动完成 (日志: $LOG)"
exit 0
