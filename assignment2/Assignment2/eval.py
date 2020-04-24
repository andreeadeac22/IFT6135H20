import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.graph_objects as go


def plot(results_dict):
    sns.set()

    plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('train-val-ppl')
    plt.title('PPL-epoch for different transformers')


    plt.figure(2)
    plt.xlabel('wall clock time')
    plt.ylabel('train-val-ppl')
    plt.title('PPL-time for different transformers')


    colors = ['b', 'r', 'g', 'c', 'y']

    for i, filename in enumerate(results_dict):
        print(filename)
        chunks = filename.split('_')
        print(chunks)
        lr = float(chunks[5][3:])
        batch_size = int(chunks[7][5:])
        hidden = int(chunks[11][5:])
        layers = int(chunks[13][7:])
        dropout = round(1.0 - float(chunks[16][5:]), 2)

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
    plt.savefig('epoch3.4.jpg')

    plt.figure(2)
    plt.legend()
    plt.savefig('time3.4.jpg')


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

def plot_gradients():
    sns.set()
    gru = [1.1034783124923706, 0.7521699666976929, 0.6085227727890015, 0.5766541957855225,
           0.5495656132698059, 0.4859800338745117, 0.45705774426460266, 0.4491726756095886,
           0.4264039099216461, 0.4018104672431946, 0.3858887851238251, 0.37800902128219604,
           0.37694263458251953, 0.3777613043785095, 0.3575204014778137, 0.36599305272102356,
           0.3653583526611328, 0.37312740087509155, 0.3892570436000824, 0.37735849618911743,
           0.36793264746665955, 0.3454682528972626, 0.3434099555015564, 0.33086416125297546,
           0.2790318429470062, 0.25834959745407104, 0.17171697318553925, 0.13950090110301971,
           0.13211959600448608, 0.11759805679321289, 0.11285962909460068, 0.09900135546922684,
           0.10353565216064453, 0.06380647420883179, 0.037302251905202866]
    min_gru = min(gru)
    max_gru = max(gru)
    max_min = max_gru - min_gru
    norm_gru = [(x-min_gru)/max_min for x in gru]

    rnn = [0.09558497369289398, 0.10094014555215836, 0.09700753539800644, 0.09581667184829712,
           0.09518793970346451, 0.09138790518045425, 0.10784641653299332, 0.10734105110168457,
           0.09636344760656357, 0.09609982371330261, 0.07991824299097061, 0.08668375760316849,
           0.08960548043251038, 0.09018288552761078, 0.08722814172506332, 0.08348348736763,
           0.08770351856946945, 0.09094373136758804, 0.08978399634361267, 0.10210102051496506,
           0.09554292261600494, 0.09386806190013885, 0.07771776616573334, 0.03571728616952896,
           0.07317306101322174, 0.07159148156642914, 0.083177849650383, 0.0921616330742836,
           0.08615701645612717, 0.06031247228384018, 0.0660326853394508, 0.04636253044009209,
           0.07645558565855026, 0.06771436333656311, 0.029756637290120125]

    min_rnn = min(rnn)
    #print(min_rnn)
    max_rnn = max(rnn)
    max_min = max_rnn - min_rnn
    norm_rnn = [(x - min_rnn) / max_min for x in rnn]

    print(norm_gru)
    print(norm_rnn)


    plt.figure()
    plt.xlabel('Timestep')
    plt.ylabel('Gradient norm')
    plt.title('Gradient norm over timesteps')

    plt.plot(range(35), norm_rnn, 'r', label='RNN grads')
    plt.plot(range(35), norm_gru, 'b', label='GRU grads')

    plt.legend()
    plt.show()



def main():
    #    - For Problem 3.4 the hyperparameter settings you should run are as follows
    #            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20
    #            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=2 --dp_keep_prob=0.9 --num_epochs=20
    #            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.6 --num_epochs=20
    #            --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=1024 --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20
    dirs = os.walk('.')
    results_dict = {}
    """
    1-4
    for dir in dirs:
        # if dir[0].startswith('./Assignment2/RNN'): # for 3.1
        # if dir[0].startswith('./Assignment2/GRU') and 'hidden_size=512' in dir[0] and 'layers=2' in dir[0]: # for 3.2
        #if dir[0].startswith('./Assignment2/GRU_ADAM') and 'batch_size=128' in dir[0] \
        #        and ('hidden_size=256' in dir[0] or 'hidden_size=2048' in dir[0] or \
        #             ('hidden_size=512' in dir[0] and 'num_layers=4' in dir[0])): #for 3.3
        if dir[0].startswith('./Assignment2/TRANSFORMER'): # for 3.4
            files = dir[2]
            filename = dir[0] + '/' + files[1]
            print(filename)
            results_dict[filename] = parse_log(filename)
    print("Plotting")
    plot(results_dict)
    
    for dir in dirs:
        if dir[0].startswith('./Assignment2/'):
            files = dir[2]
            filename = dir[0] + '/' + files[1]
            print(filename)
            f = open(filename, 'r')
            lines = f.readlines()
            print(lines[-1])
            results = parse_log(filename)
            print("time ", results[-1][-1])
    """
    plot_gradients()





if __name__ == "__main__":
    main()