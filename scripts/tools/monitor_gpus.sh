#!/bin/bash
# GPU 监控脚本 - 用于诊断训练时的 GPU 负载问题

echo "========================================"
echo "GPU 监控脚本 - 每5秒更新一次"
echo "按 Ctrl+C 停止监控"
echo "========================================"

while true; do
    clear
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    
    # 显示 GPU 使用情况
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' '{
        printf "GPU %s [%s]\n", $1, $2
        printf "  利用率: GPU=%s%% MEM=%s%%\n", $3, $4
        printf "  显存: %s/%s MB\n", $5, $6
        printf "  温度: %s°C  功耗: %sW\n\n", $7, $8
    }'
    
    # 检查是否有进程卡住
    echo "========================================"
    echo "训练进程状态:"
    ps aux | grep -E "(python|torchrun)" | grep -v grep | grep -v monitor | \
    awk '{printf "PID: %s  CPU: %s%%  MEM: %s%%  CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    
    echo "========================================"
    sleep 5
done

