#!/usr/bin/env bash
# 运行 Ollama 服务的便捷脚本(支持多GPU)
# 用法：./ollamaserver.sh [GPU_LIST] [NUM_PARALLEL]
# 例如：bash ./ollamaserver.sh "1,2,3" 2
# 如果不传 GPU_LIST，默认使用 "5"
# 如果不传 NUM_PARALLEL，默认使用 8
# 
set -euo pipefail

# 读取 GPU 列表参数（默认 "5"）
gpu_list="${1:-5}"

# 读取 NUM_PARALLEL 参数（默认 8）
num_parallel="${2:-8}"

# 校验 NUM_PARALLEL 是否是数字
if [[ ! "$num_parallel" =~ ^[0-9]+$ ]]; then
  echo "错误: NUM_PARALLEL 必须是数字，例如 4、8、16"
  exit 1
fi

# 将 GPU 列表转换为数组
IFS=',' read -ra gpu_array <<< "$gpu_list"

# 校验每个 GPU_ID 是否是数字
for gpu in "${gpu_array[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "错误: GPU_ID 必须是数字，例如 0、1、5，您输入的是: $gpu"
    exit 1
  fi
done

echo "================================================"
echo "🚀 准备启动 Ollama 多GPU服务"
echo "GPU 列表: ${gpu_array[*]}"
echo "每个服务并行数: $num_parallel"
echo "================================================"

PORT_BASE=11435
pids=()
LOG_FILE="ollama_server.log"
is_first=true

# 为每张 GPU 启动一个 ollama serve
for gpu in "${gpu_array[@]}"; do
  port=$((PORT_BASE + gpu))
  
  # 检查端口是否被占用
  if command -v ss >/dev/null; then
    if ss -ltn | grep -q ":$port "; then
      echo "⚠️  警告: 端口 $port 已被占用"
      
      # 查找占用端口的进程并终止
      echo "   🔍 正在查找占用端口 $port 的进程..."
      pid_on_port=$(lsof -ti :$port 2>/dev/null || true)
      
      if [ -n "$pid_on_port" ]; then
        echo "   💀 终止进程: $pid_on_port"
        kill -9 $pid_on_port
        sleep 1
        echo "   ✅ 端口 $port 已释放"
      else
        echo "   ❌ 无法找到占用端口的进程"
        continue
      fi
    fi
  fi

  echo ""
  echo "📍 启动 GPU $gpu 上的 Ollama 服务..."
  echo "   端口: $port"
  echo "   并行数: $num_parallel"
  
  # 设置共有的环境变量
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OLLAMA_MODELS="~/.ollama/models"
  export OLLAMA_HOST="0.0.0.0:$port"
  export OLLAMA_MAX_LOADED_MODELS=1
  export OLLAMA_MAX_QUEUE=128
  export OLLAMA_NUM_PARALLEL="$num_parallel"
  export OLLAMA_FLASH_ATTENTION=1 # Flash Attention 加速
  export OLLAMA_NOPRUNE=1  # 禁用模型裁剪
  export OLLAMA_SCHED_SPREAD=0 # 禁用调度分散
  export OLLAMA_KEEP_ALIVE="5m" # 长时间保持模型
  export OLLAMA_GPU_LAYERS=999 # 使用所有 GPU 层
  export OLLAMA_LOG_LEVEL="info" # 日志级别
  export OLLAMA_GPU_MEMORY_FRACTION=1.0 # 使用全部 GPU 内存

  
  # 第一个服务保存日志,其他服务丢弃日志
  if [ "$is_first" = true ]; then
    nohup ollama serve > "$LOG_FILE" 2>&1 &
    echo "   📝 日志保存到: $LOG_FILE"
    is_first=false
  else
    nohup ollama serve > /dev/null 2>&1 &
  fi
  
  pid=$!
  pids+=($pid)
  echo "   ✅已启动,PID: $pid"
  
  # 短暂延迟，避免端口冲突
  sleep 1
done

echo ""
echo "================================================"
echo "✨ 所有服务启动完成！"
echo "进程 PIDs: ${pids[*]}"
echo ""
echo "📋 服务列表:"
for gpu in "${gpu_array[@]}"; do
  port=$((PORT_BASE + gpu))
  echo "   GPU $gpu -> http://0.0.0.0:$port"
done
echo ""
echo "📝 查看日志: tail -f $LOG_FILE (仅 GPU ${gpu_array[0]} 的日志)"
echo "🛑 停止所有服务: kill ${pids[*]}"
echo "================================================"

tail -f $LOG_FILE