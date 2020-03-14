import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.graph_objects as go


def plot(results_dict):
    sns.set()

    plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('train-val-ppl')
    plt.title('PPL-epoch for different GRUs')


    plt.figure(2)
    plt.xlabel('wall clock time')
    plt.ylabel('train-val-ppl')
    plt.title('PPL-time for different GRUs')


    colors = ['b', 'r', 'g', 'c', 'y']

    for i, filename in enumerate(results_dict):
        print(filename)
        chunks = filename.split('_')
        print(chunks)
        lr = float(chunks[5][3:])
        batch_size = int(chunks[7][5:])
        hidden = int(chunks[11][5:])
        layers = int(chunks[13][7:])
        dropout = 1.0 - float(chunks[16][5:])

        #if 'SGD' in filename: # for 3.1 and 3.2
        #    label = 'SGD, lr=' + str(lr) + ', b=' +str(batch_size)
        #else:
        #    label = 'ADAM, lr=' + str(lr) + ', b=' + str(batch_size)

        label = 'h=' + str(hidden) + ', l=' + str(layers) + ', d=' + str(dropout)

        result_list = results_dict[filename]
        train_ppl = result_list[0]
        print("train_ppl ", train_ppl)
        valid_ppl = result_list[1]
        print("valid_ppl ", valid_ppl)

        time = result_list[3]
        #print("time ", time)

        plt.figure(1)
        plt.plot(range(20), train_ppl, colors[i], label='train ' + label)
        plt.plot(range(20), valid_ppl, colors[i] + '--',  label='val ' + label)

        plt.figure(2)
        plt.plot(time, train_ppl, colors[i], label='train ' + label)
        plt.plot(time, valid_ppl, colors[i] + '--', label='val ' + label)

    plt.figure(1)
    plt.legend()
    #plt.ylim(30, 1500)
    plt.savefig('epoch3.3.jpg')

    plt.figure(2)
    plt.legend()
    plt.savefig('time3.3.jpg')


def parse_log(file_name):
    """
    log_str = 'epoch: ' + str(epoch) + '\t' \
            + 'train ppl: ' + str(train_ppl) + '\t' \
            + 'val ppl: ' + str(val_ppl)  + '\t' \
            + 'best val: ' + str(best_val_so_far) + '\t' \
            + 'time (s) spent in epoch: ' + str(times[-1])
    print(log_str)
    with open (os.path.join(args.save_dir, 'log.txt'), 'a') as f_:
        f_.write(log_str+ '\n')
    """
    f = open(file_name, 'r')
    train_ppl = []
    valid_ppl = []
    best_val = []
    time = []
    total_time = 0

    for line in f:
        chunks = line.split('\t')
        train_ppl += [float(chunks[1][11:])]
        valid_ppl += [float(chunks[2][9:])]
        best_val += [float(chunks[3][10:])]
        new_time = float(chunks[4][24:])
        total_time += new_time
        time += [total_time]

    return [train_ppl, valid_ppl, best_val, time]


def main():
    #    - For Problem 3.3 the hyperparameter settings you should run are as follows
    #            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.2  --num_epochs=20
    #            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20
    #            --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.5  --num_epochs=20
    dirs = os.walk('.')
    results_dict = {}
    for dir in dirs:
        # if dir[0].startswith('./Assignment2/RNN'): # for 3.1
        # if dir[0].startswith('./Assignment2/GRU') and 'hidden_size=512' in dir[0] and 'layers=2' in dir[0]: # for 3.2
        if dir[0].startswith('./Assignment2/GRU_ADAM') and 'batch_size=128' in dir[0] \
                and ('hidden_size=256' in dir[0] or 'hidden_size=2048' in dir[0] or \
                     ('hidden_size=512' in dir[0] and 'num_layers=4' in dir[0])): #for 3.3
            files = dir[2]
            filename = dir[0] + '/' + files[1]
            print(filename)
            results_dict[filename] = parse_log(filename)
    print("Plotting")
    plot(results_dict)




if __name__ == "__main__":
    main()