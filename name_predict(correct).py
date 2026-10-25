# ============================================================
# makemore 第一讲：从统计二元语法 -> 单层神经网络(softmax)
# ============================================================
import torch
import torch.nn.functional as F

# ---------- 1. 读入数据 ----------
words = open("names.txt", "r").read().splitlines()   # 每行一个名字
print("总词数:", len(words))
print("样例:", words[:8])


# ---------- 2. 构建字符表与索引映射 ----------
# 所有出现过的字符（去重、排序）
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}   # 字符 -> 索引(1..26)
stoi["."] = 0                                    # '.' 代表句首/句尾，索引 0
itos = {i: s for s, i in stoi.items()}           # 索引 -> 字符（反向映射）

# 注意：'s' 与 '.' 的先后在字典里不影响最终结果，
# 但 stoi['.']=0 必须在建 itos 之前设置好

V = len(stoi)   # 词表大小 = 27
print("词表大小 V =", V)


# ---------- 3. 统计版：直接数共现次数 ----------
N = torch.zeros((V, V), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]           # 首尾都加 '.'
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        N[ix1, ix2] += 1


# ---------- 4. 加一平滑 + 行归一化 -> 概率矩阵 P ----------
P = (N + 1).float()                         # +1 平滑，避免 log(0)
P = P / P.sum(1, keepdim=True)              # 每行归一化，第 i 行=下一个字符分布


# ---------- 5. 用统计版 P 采样生成名字 ----------
g = torch.Generator().manual_seed(2147483647)
print("\n--- 统计版采样 ---")
for _ in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:                          # 抽到 '.' 结束
            break
        out.append(itos[ix])
    print("".join(out))


# ---------- 6. 构建训练集 (x -> y) ----------
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print("\n样本数:", num)


# ---------- 7. 初始化权重 W ----------
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((V, V), generator=g, requires_grad=True)


# ---------- 8. 训练：softmax 回归 ----------
for k in range(200):
    # ---- forward pass ----
    xenc = F.one_hot(xs, num_classes=V).float()      # [num, 27] 独热
    logits = xenc @ W                                # [num, 27] 取出 W 对应行
    counts = logits.exp()                            # 变正数
    probs = counts / counts.sum(1, keepdim=True)     # softmax 归一化

    # ---- 损失：交叉熵(负对数似然) + L2 正则 ----
    nll = -probs[torch.arange(num), ys].log().mean()
    loss = nll + 0.01 * (W ** 2).mean()              # 加 W² 正则

    if k % 20 == 0:
        print(f"step {k:3d}  loss = {loss.item():.4f}")

    # ---- backward pass ----
    W.grad = None                                    # 清零梯度
    loss.backward()                                  # 反向传播

    # ---- update ----
    W.data += -50 * W.grad                           # SGD 更新


# ---------- 9. 用训练好的 W 采样生成名字 ----------
print("\n--- 神经网络版采样 ---")
g = torch.Generator().manual_seed(2147483647)
for _ in range(10):
    ix = 0
    out = []
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=V).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p[0], num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("".join(out))
