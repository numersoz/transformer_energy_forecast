import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

settings = "informer_custom_ftMS_sl240_ll48_pl336_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_encoder_length_0"
metrics = np.load('./checkpoints/'+settings+'/learning_curves.npy')

plt.figure(figsize=(20,10))
plt.plot(metrics[0,:], label='Training Loss', color="blue", linestyle='--', marker='o')
plt.plot(metrics[1,:], label='Validation Loss', color="orange", linestyle='--', marker='o')
plt.xlabel('epochs', fontsize=20,fontweight='bold')
plt.ylabel('Loss', fontsize=20,fontweight='bold')
plt.legend(loc=0, prop={'weight': 'bold', "size": '16'})
plt.show()