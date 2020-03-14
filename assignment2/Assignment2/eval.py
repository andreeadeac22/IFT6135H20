import matplotlib.pyplot as plt
import os



def plot(exp1, exp2, exp3, exp4, exp5):

    zero_init_train_loss = [2.302421418707069, 2.302274667119889, 2.3021431184461356, 2.3020252225102973,
                            2.3019195827656724, 2.3018249417288428, 2.30174016771185, 2.3016642427439344,
                            2.301596251583872, 2.3015353717321325]
    normal_init_train_loss = [2.1670133457355707, 1.6282529320101617, 1.412485651398122, 1.1769963806257004,
                              1.0669246246750133, 0.9999256249361914, 0.9400407181562437, 0.8923009483761754,
                              0.8431614317669878, 0.8175628259848486]
    glorot_init_train_loss = [1.8601016234556451, 1.4400731981324928, 1.1124919536438642, 0.897167160486312,
                              0.7593353841920966, 0.6672244247053242, 0.6021454189180506, 0.553894626041198,
                              0.5167791770125256, 0.4873786554075815]
    plt.figure()
    plt.plot(range(10), zero_init_train_loss, 'b', label='Zero initialization')
    plt.plot(range(10), normal_init_train_loss, 'r', label='Normal initialization')
    plt.plot(range(10), glorot_init_train_loss, 'g', label='Glorot initialization')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('Average training loss using different weight initializations')
    plt.legend()
    plt.savefig('initializations.jpg')


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

    for line in f:
        chunks = line.split('\t')
        print(chunks[0], chunks[1])



def main():
    # - For Problem 3.1 the hyperparameter settings you should run are as follows
    #            --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
    #            --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=20  --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    #            --model=RNN --optimizer=SGD --initial_lr=10.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    #            --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    #            --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
    dirs = os.walk('.')
    for dir in dirs:
        if dir[0].startswith('./Assignment2/RNN'):
            print(dir[0])
            files = dir[2]
            print(files[1])
            filename = dir[0] + '/' + files[1]
            #print(filename)
            parse_log(filename)




if __name__ == "__main__":
    main()