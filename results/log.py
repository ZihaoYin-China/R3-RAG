import json

# 读取文件
with open('hybrid_local_reflection_v66_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pid = "17988"
if pid in data:
    item = data[pid]
    print(f"=== PID: {pid} ===")
    print(f"Question: {item.get('question')}")
    print(f"Correct Answer: {item.get('answer')}")
    print(f"Prediction: {item.get('prediction')}")
    
    # 打印图片文件名（如果有）
    print(f"Image File: {item.get('image')}")
    
    # 打印反思过程 (Messages / Trace)
    print("\n--- Inference Trace ---")
    messages = item.get('messages', [])
    for msg in messages:
        # 这里通常包含 'Rewritten Query' 或 'Step 2' 等反思痕迹
        print(msg)
else:
    print(f"PID {pid} not found in this file.")