import matplotlib.pyplot as plt

epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_loss = [2.276604, 1.940021, 1.601451, 1.156798, 0.822963, 0.638372, 0.528853, 0.444104, 0.380950, 0.341120]
val_loss = [2.143205,2.168455,2.247194,2.393478,2.457625,2.503088,2.521556,2.555432,2.476453,2.548854]

plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')# 'bo-' specifies blue color, round markers, and solid line
plt.title('Loss against Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("msi-loss.png")
plt.show()