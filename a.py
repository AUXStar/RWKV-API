from rwkv_api import Client

client = Client()

# 方式1：create(stream=True) 返回生成器
for chunk in client.create("User:你好呀\n\nAssistant:", stream=True, max_tokens=50):
    print(chunk, end="", flush=True)
print()

# 方式2：create_stream 显式流式
for chunk in client.create_stream("User:你好呀\n\nAssistant:", max_tokens=50):
    print(chunk, end="", flush=True)
print()

# 方式3：非流式，返回 Task 对象
task = client.create("User:你好呀\n\nAssistant:", max_tokens=50)
result = task.wait()
print(result.result)
print(f"speed={result.speed} tok/s")
