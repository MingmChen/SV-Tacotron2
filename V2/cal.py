import numpy as np
import os

a = list()
cnt = 0

for file in os.listdir("dataset"):
    mel = np.load(os.path.join("dataset", file))
    # print(np.shape(mel))
    a.append(np.shape(mel)[1])
    if np.shape(mel)[1] >= 200:
        cnt = cnt + 1

print(np.array(a).mean())
print(cnt)
