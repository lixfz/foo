import time

import matplotlib.pyplot as plt
import numpy as np
data = [{"value":256},{"value":15},{"value":330}]
print(sorted(list([i["value"] for i in data])))
data=sorted(list([i["value"] for i in data]))
sample_indices = np.arange(len(data))  # 0, 1, 2, ..., 999

plt.figure(figsize=(10, 5))
plt.scatter(sample_indices, data, alpha=0.5, s=10, color="blue")  # s 控制点大小，alpha 控制透明度

# 添加标题和轴标签
plt.title("Scatter Plot of Samples")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.show()

print(f"%s_%d.png"%("abc",time.time()))
