

import matplotlib.pyplot as plt
import numpy as np

def save_loss_graphs(t_c, t_r, t_tot, v_c, v_r, v_tot, start_epoch, no_epochs, exp):
  x = np.arange(start_epoch, start_epoch + no_epochs, 1)
  graphs = ['Classification', 'Regression', 'Total']
  train = [t_c, t_r, t_tot]
  valid = [ v_c, v_r, v_tot]
  for i in range(3):
    plt.plot(x, train[i], label = "Train")
    plt.plot(x, valid[i], label = "Valid")
    plt.xlabel('Epochs')
    plt.xticks(x)
    # Set the y axis label of the current axis.
    plt.ylabel('Loss')
    # plt.yticks(np.arange(0,3,0.1))
    # Set a title of the current axes.
    plt.title( graphs[i] +'Loss') 
    plt.text(start_epoch + no_epochs-10,0.75,"Final Train loss:  " + str(train[i][no_epochs-1]))
    plt.text(start_epoch + no_epochs-10,0.5,"Final Valid loss:  " + str(valid[i][no_epochs-1]))
  # show a legend on the plot
    plt.legend()
  # Display a figure.
    plt.savefig('saved_models/'+ str(exp)+ graphs[i] +'Loss.png')
    plt.show()
    
