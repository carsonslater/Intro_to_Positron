import matplotlib.pyplot as plt
import numpy as np
from chatlas import ChatOllama

# To activate venv,  `source myvenv/bin/activate`

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()
