import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot(results_dict):
    sns.set()

    plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('train-val-ppl')
    plt.title('PPL for different RNNs')


    plt.figure(2)
    plt.xlabel('wall clock time')
    plt.ylabel('train-val-ppl')
    plt.title('PPL-time for different RNNs')


    colors = ['b', 'r', 'g', 'c', 'y']

    for i, filename in enumerate(results_dict):
        print(filename)
        chunks = filename.split('_')
        print(chunks)
        print(chunks[5][3:])
        lr = float(chunks[5][3:])
        batch_size = int(chunks[7][5:])
        print(lr)
        print(batch_size)

        if 'SGD' in filename:
            label = 'SGD, batch=' + str(batch_size)
        else:
            label = 'ADAM, lr=' + str(lr)

        #batch_size = int(chunks[])

        result_list = results_dict[filename]
        train_ppl = result_list[0]
        print("train_ppl ", train_ppl)
        valid_ppl = result_list[1]
        print("valid_ppl ", valid_ppl)

        time = result_list[3]
        #print("time ", time)

        plt.figure(1)
        plt.plot(range(20), train_ppl, colors[i], label=label)
        plt.plot(range(20), valid_ppl, colors[i] + '--',  label=label)

        plt.figure(2)
        plt.plot(time, train_ppl, colors[i], label=label)
        plt.plot(time, valid_ppl, colors[i] + '--', label=label)

    plt.figure(1)
    plt.legend()
    #plt.ylim(30, 1500)
    plt.savefig('epoch3.1.jpg')

    plt.figure(2)
    plt.legend()
    plt.savefig('time3.1.jpg')


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
    # - For Problem 3.1 the hyperparameter settings you should run are as follows
    #            --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
    #            --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=20  --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    #            --model=RNN --optimizer=SGD --initial_lr=10.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    #            --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    #            --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    dirs = os.walk('.')
    results_dict = {}
    for dir in dirs:
        if dir[0].startswith('./Assignment2/RNN'):
            print(dir[0])
            files = dir[2]
            print(files[1])
            filename = dir[0] + '/' + files[1]
            print(filename)
            results_dict[filename] = parse_log(filename)

    plot(results_dict)




if __name__ == "__main__":
    main()