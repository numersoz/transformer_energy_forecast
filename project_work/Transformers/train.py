import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader

import torch.nn as nn

from config import build_parser
from model import Transformer
from data_loader import ParseData,Dataset_Pred
from metrics import metric
from utils.tools import adjust_learning_rate


def get_model(args):
    return Transformer(args.embedding_size, args.hidden_size, args.input_len, args.dec_seq_len, args.pred_len,
                       output_len=args.output_len,
                       n_heads=args.n_heads, n_encoder_layers=args.n_encoder_layers,
                       n_decoder_layers=args.n_decoder_layers, dropout=args.dropout)


def get_params(mdl):
    return mdl.parameters()


def _get_data(args, flag):     
    Data = ParseData
    if flag == 'test':
        shuffle_flag = False;
        drop_last = True;
        batch_size = args.batch_size #32
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False;
        drop_last = False;
        batch_size = 1;
        freq = args.freq;
        Data = Dataset_Pred
        #flag = 'train'        
    else:
        shuffle_flag = True;
        drop_last = True;
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path='data',
        data_path=args.data+'.csv',
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        # timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size if len(data_set) > 1 else 1,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last if len(data_set) > 1 else False)

    return data_set, data_loader


def run_metrics(caption, preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    print('before shape:', preds.shape, trues.shape)    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('after shape:', preds.shape, trues.shape)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('{} ; MSE: {}, MAE: {}, rmse: {} '.format(caption, mse, mae, rmse))
    return mse, mae, rmse, preds, trues

'''
def vali(model, vali_data, vali_loader):
    model.eval()
    total_loss = []
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
        model.optim.zero_grad()

        batch = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        target = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32,
                              device=target_device)

        elem_num += len(batch)
        steps += 1

        result = model(batch)

        loss = nn.functional.mse_loss(result.squeeze(2), target.squeeze(2), reduction='mean')

        #pred = result.detach().cpu().unsqueeze(2).numpy()  # .squeeze()
        pred = result.detach().cpu().numpy()  # .squeeze()
        true = target.detach().cpu().numpy()  # .squeeze()
    total_loss = np.average(total_loss)
    self.model.train()
    return total_loss
'''

def run_iteration(model, loader, args, training=True, vali = False, message = ''):
    if not training:        
        model.eval()
    preds = []
    trues = []
    total_loss = []
    elem_num = 0
    steps = 0
    target_device = 'cuda:{}'.format(args.local_rank)        
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        
        if training:
            model.optim.zero_grad()

        batch = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        target = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32,
                              device=target_device)

        elem_num += len(batch)
        steps += 1

        result = model(batch)

        loss = nn.functional.mse_loss(result.squeeze(2), target.squeeze(2))#, reduction='mean')
        
        #pred = result.detach().cpu().unsqueeze(2).numpy()  # .squeeze()
        pred = result.detach().cpu().numpy()  # .squeeze()
        true = target.detach().cpu().numpy()  # .squeeze()

        preds.append(pred)
        trues.append(true)

        unscaled_loss = loss.item()
        #total_loss += unscaled_loss
        total_loss.append(unscaled_loss)
        #print("{} Loss at step {}: {}, mean for epoch: {}, mem_alloc: {}".format(message, steps, unscaled_loss, total_loss / steps,torch.cuda.max_memory_allocated()))

        if training:          
            loss.backward()
            model.optim.step()  
        if vali:
            model.train()

    return preds, trues, np.average(total_loss)


def preform_experiment(args):
    model = get_model(args)
    params = list(get_params(model))
    #print('Number of parameters: {}'.format(len(params)))
    lr=0.001
    if args.lr:
        lr = args.lr  
    for p in params:          
        model.to('cuda')
        model.optim = Adam(params, lr)

    train_data, train_loader = _get_data(args, flag='train')
    test_data, test_loader = _get_data(args, flag='test')
    vali_data, vali_loader = _get_data(args, flag = 'val')
    #pred_data, pred_loader = _get_data(args, flag='pred')


    train_loss_history = []    
    valid_loss_history = []    
    test_loss_history = []    
    train_losses = []    
    valid_losses = []    
    test_losses = []    
    tpreds = []
    ttrues = []

    start = time.time()
    for iter in range(1, args.iterations + 1):
        preds, trues, tloss = run_iteration(model , train_loader, args, training=True, vali= False, message=' Run {:>3}, iteration: {:>3}:  '.format(args.run_num, iter))
        tmse, tmae, _, tpreds, ttrues = run_metrics("Loss after iteration {}".format(iter), preds, trues)
        train_loss_history.append(tmse)
        train_losses.append(tloss)        

        v_preds, v_trues,v_loss = run_iteration(model, vali_loader, args, training=False, vali= True, message="Validation set")
        vmse, vmae,  _, _, _ = run_metrics("Loss for validation set iteration {}".format(iter), v_preds, v_trues)
        valid_loss_history.append(vmse)
        valid_losses.append(v_loss)

        t_preds, t_trues,t_loss = run_iteration(model, test_loader, args, training=False, vali= True, message="test set")
        t_mse, t_mae,  _, _, _ = run_metrics("Loss for test set iteration {}".format(iter), t_preds, t_trues)
        test_loss_history.append(t_mse)
        test_losses.append(t_loss)
        
        print("Epoch: {0}, - Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                iter, tloss, v_loss, t_loss))

        adjust_learning_rate(model.optim,iter,args)

        #print("Time per iteration {}, memory {}".format((time.time() - start)/iter, torch.cuda.memory_stats()))

    #print(torch.cuda.max_memory_allocated())

    if args.debug:
        model.record()
    
    #t_preds, t_trues = run_iteration(model, pred_loader, args, training=True, message="pred set")
    #mse, mae, rmse, t_preds_a, t_trues_a = run_metrics("Loss for pred set ", t_preds, t_trues)

    #model.eval()
    # Model evaluation on validation data   
    

    test_preds, test_trues, test_loss = run_iteration(model, test_loader, args, training=False,vali = False, message="For rmse test set")
    mse, mae, rmse, v_preds_a, v_trues_a = run_metrics("Loss for test set after exp ", test_preds, test_trues)
    print('shape:', v_preds_a.shape, v_trues_a.shape)

    #v_preds_l, v_trues_l = predtrue(v_preds_a,v_trues_a,ttrues,args)
    plot_curves(train_loss_history,valid_loss_history,v_preds_a, v_trues_a)

def predtrue(preds,trues,trainpts,args):    
    #print(preds.shape)
    #print(trues.shape)
    #print(trainpts.shape)
    #print(len(trainpts))
    predsa = []
    truesa = []
    traindt = []

    lookback = trainpts.shape[0] - args.pred_len * 5 # 4 days
    print(lookback)    
    for t in range(lookback,trainpts.shape[0]):
        traindt.append(trainpts[t, 0, -1])
        
    print(np.shape(traindt))
    for i in range(24*5):
        predsa.append(preds[i, 0, -1])
        truesa.append(trues[i, 0, -1])

    print(np.shape(predsa))
    print(np.shape(truesa))    
    combotrue = np.hstack((traindt, truesa))
    combopred = np.empty_like(combotrue)
    combopred[:] = np.nan
    combopred[len(traindt):] = predsa
    #print(combopred)
    print(np.shape(combotrue))
    print(np.shape(combopred))
    
    return predsa, truesa
    



def plot_curves(train_loss_history, valid_loss_history,preds,trues):    
    
    legend_properties = {'weight':'bold','size':'16'}
    
    fig = plt.figure(figsize=(20,10))

    plt.plot(trues[-1,:,-1])
    plt.plot(preds[-1,:,-1])
    plt.xlabel('Hours')
    plt.ylabel('Kilowatts')
    plt.title("Test vs. Prediction")    

    fig.savefig('elect_pred_truth_14.jpg')

    plt.close(fig)
    

    fig = plt.figure(figsize=(20,10))

    plt.plot(train_loss_history,label='training loss', color = "blue")

    plt.plot(valid_loss_history,label='validation loss',color = "orange")

    plt.legend(loc=0,prop=legend_properties)

    plt.xlabel('epochs', fontsize=20,fontweight='bold')

    #plt.xlim([0,epochs])

    plt.ylabel('Loss', fontsize=20,fontweight='bold')

    #plt.grid(True)

    plt.title("Training and Validation loss",fontsize=25,fontweight='bold')

    #plt.show()

    fig.savefig('elect_loss_hour14.jpg')

    plt.close(fig)


def main():   
    parser = build_parser()
    args = parser.parse_args(None)
    preform_experiment(args)


if __name__ == '__main__':
    main()
