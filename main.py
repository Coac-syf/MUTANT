import torch
import numpy as np
import torch.optim as optim
from utils import get_data, get_data_dim, get_loader
from eval_method import bf_search
from tqdm import tqdm

from Model import MUTANT

class ExpConfig():
    dataset = "SMAP"
    val = 0.35  # the ratio of validation set
    max_train_size = None  # `None` means full train set
    train_start = 0

    max_test_size = None  # `None` means full test set
    test_start = 0

    input_dim = get_data_dim(dataset)
    batch_size = 120

    out_dim = 5   # the dimension of embedding
    window_length = 20
    hidden_size = 80  # the dimension of hidden layer in LSTM-based attention
    latent_size = 80  # the dimension of hidden layer in VAE
    N = 256

def main():
    config = ExpConfig()

    (train_data, _), (test_data, test_label) = \
        get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
                 test_start=config.test_start)

    n = int(test_data.shape[0] * config.val)
    test_data = test_data[:-n]
    test_label = test_label[:-n]

    val_data = test_data[-n:]
    val_label = test_label[-n:]

    print("test_data:", test_data.shape)
    print("val_data:", val_data.shape)

    train_data = train_data[np.arange(config.window_length)[None, :] + np.arange(train_data.shape[0] - config.window_length)[:, None]]
    val_data = val_data[np.arange(config.window_length)[None, :] + np.arange(val_data.shape[0] - config.window_length)[:, None]]
    test_data = test_data[np.arange(config.window_length)[None, :] + np.arange(test_data.shape[0] - config.window_length)[:, None]]

    num_val = int(val_data.shape[0]/config.batch_size)
    con_val = val_data.shape[0] % config.batch_size
    num_t = int(test_data.shape[0]/config.batch_size)
    con_t = test_data.shape[0] % config.batch_size

    w_size = config.input_dim * config.out_dim

    train_loader = get_loader(train_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=True)
    val_loader = get_loader(val_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=True)
    test_loader = get_loader(test_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=False)

    model = MUTANT(config.input_dim, w_size, config.hidden_size, config.latent_size, config.batch_size, config.window_length, config.out_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    save_path = 'model.pt'
    flag = 0
    f1 = -1
    for epoch in range(10):
        l = 0
        i = 0
        for inputs in tqdm(train_loader):
            loss = model(inputs)
            loss.backward()
            if(i% config.N == 0):
                optimizer.step()
                optimizer.zero_grad()
            i += 1
            
        if(flag == 1):
            model.load_state_dict(torch.load(save_path))
        val_score = model.is_anomaly(val_loader, num_val, con_val)
        t, th = bf_search(val_score, val_label[-len(val_score):], step_num=700)
        if(t[0] > f1):
            f1 = t[0]
            torch.torch.save(model.state_dict(), save_path)
            flag = 1

    model.load_state_dict(torch.load(save_path))
    test_score = model.is_anomaly(test_loader, num_t, con_t)

    t, th = bf_search(test_score, test_label[-len(test_score):], step_num=700)
    print('*****************************************************')
    print('dataset:', config.dataset)
    print('th:', th)
    print("TP:", t[3])
    print("FP:", t[5])
    print("FN:", t[6])
    print("precision:", t[1], "recall:", t[2], "f_score", t[0])

    return t[1], t[2], t[0]

if __name__ == '__main__':
   main()
