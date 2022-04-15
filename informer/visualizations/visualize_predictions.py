import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

settings = "dto-drnn_custom_ftMS_sl240_ll72_pl72_dm300_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_dto_drnn_0"

preds = np.load('./results/'+settings+'/pred.npy')
trues = np.load('./results/'+settings+'/true.npy')

print(np.sqrt(np.mean((preds[-1,:,-1] -trues[-1,:,-1])**2)))

plt.figure(figsize=(20,10))
plt.plot(trues[-1,:,-1], label='Ground Truth', color="blue")
plt.plot(preds[-1,:,-1], label='Prediction', color="orange")
plt.xlabel('Hours', fontsize=20, fontweight='bold')
plt.ylabel('Kilowatts (normalized)', fontsize=20, fontweight='bold')
plt.legend(loc=0, prop={'weight': 'bold', "size": '16'})
plt.show()