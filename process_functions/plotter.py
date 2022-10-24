from re import X
import matplotlib.pyplot as plt

def create_graph(list, x_label = 'X', y_label = 'Y', name = 'Default Graph'):
    
    plt.plot(list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    