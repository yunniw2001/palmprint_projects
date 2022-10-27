import make_label
import matplotlib.pyplot as plt
_,_,train_idx,train_loss= make_label.read_txt('data.txt')
plt.plot(train_idx,train_loss)
plt.show()
