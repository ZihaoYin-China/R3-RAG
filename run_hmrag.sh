
#!/bin/bash
cd /Yin_zi_hao/code/HMRAG-Self-Reflection

export OS_HOST="http://default-o4tm.platform-cn-shanghai.opensearch.aliyuncs.com"
export OS_WS="default"
export OS_SVC="ops-web-search-001"
export PYTHONPATH="$(pwd):$(pwd)/agents:$PYTHONPATH"

# 强制 Ollama 及时释放内存（防止显存碎片化）
export OLLAMA_KEEP_ALIVE="0"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
python -u main.py \
  --working_dir /Yin_zi_hao/code/HMRAG-Self-Reflection \
  --data_root /Yin_zi_hao/code/HMRAG-Self-Reflection/ScienceQA/data \
  --image_root /Yin_zi_hao/code/HMRAG-Self-Reflection/ScienceQA \
  --caption_file /Yin_zi_hao/code/HMRAG-Self-Reflection/ScienceQA/data/captions.json \
  --lightrag_workdir /Yin_zi_hao/code/HMRAG-Self-Reflection/ScienceQA/ScienceQA_lightrag_workdir_train_clean \
  --mode "hybrid" \
  --top_k 30 \
  --rerank_top_k 5 \
  --label hybrid_local_reflection_v66 \
  --save_every 2 \
  --ollama_host "http://localhost:11434" \
  --llm_model_name "qwen3-vl:32b" \
  --summary_use_vision \
  --summary_vlm_model_name "qwen3-vl:32b" \
  --lightrag_llm_max_async 1 \
  --lightrag_num_ctx 4096
