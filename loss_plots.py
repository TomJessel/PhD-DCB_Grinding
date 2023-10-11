import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tbparse
from pathlib import Path

if __name__ == "__main__":
    dir_path = r"/mnt/c/Users/tomje/OneDrive - Cardiff University/" \
               r"Documents/PHD/AE/Tensorboard/"
    # mlp_name = r"MLP/MLP-E-1000-B-10-L[128 128 128]-D-0.1-20221205-144254"
    # mlp_name = r"MLP/loss_plot/MLP-E-3000-B-10-L[128 128 128]-D-0.01-20231004-145159"
    mlp_name = r"MLP/loss_plot/MLP-E-3000-B-64-L[128 128 128]-D-0.01-20231010-114024"

    # mlp_win_name = r'MLP_WIN/loss_plot/MLP_Win-WLEN-5-E-1000-B-20-L-[128 128 128 128]-D-0.01-20230911-164133'
    mlp_win_name = r'MLP_WIN/loss_plot/MLP_Win-WLEN-5-E-3000-B-64-L-[128 128 128 128]-D-0.01-20231010-114709'

    # lstm_name = r"LSTM/loss_plot/LSTM-WLEN-15-E-2000-B-10-L-[128 128 128]-D-0.1-20230912-125148"
    # lstm_name = r'LSTM/loss_plot/LSTM-WLEN-15-E-1000-B-10-L-[64 64 64 64 64]-D-0.01-20230913-111650'
    # lstm_name = r'LSTM/loss_plot/LSTM-WLEN-15-E-3000-B-128-L-[64 64 64 64 64]-D-0.01-20231010-121406'
    # lstm_name = r'LSTM/loss_plot/LSTM-WLEN-15-E-3000-B-32-L-[64 64 64 64 64]-D-0.01-20231010-142841'
    lstm_name = r'LSTM/loss_plot/LSTM-WLEN-15-E-3000-B-32-L-[64 64 64 64 64]-D-0.01-20231011-093626'

    mod_paths = [Path(dir_path + name) for name in [mlp_name, mlp_win_name, lstm_name]]

    fig, ax = plt.subplots(1, len(mod_paths),
                           figsize=(12, 4),
                           constrained_layout=True,
                           sharey=True,
                           )
    ax = ax if len(mod_paths) > 1 else [ax]

    for i, mod_path in enumerate(mod_paths):
        reader = tbparse.SummaryReader(mod_path, extra_columns={'dir_name'})
        df = reader.tensors

        train_df = df[df['dir_name'] == 'train']
        val_df = df[df['dir_name'] == 'validation']

        METRICS = [
            'epoch_loss',
        ]
        metric_aliases = {'epoch_loss': 'Loss'}

        def smooth(scalars, weight: float = 0.6):  # Weight between 0 and 1
            last = scalars[0]
            smoothed = list()
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        for metric in METRICS:
            # Training Loss Plot
            # subdf = train_df[train_df['tag'] == metric]
            # ax[i].plot(subdf['step'],#.iloc[:1001],
            #            # subdf['value'].iloc[:1001],
            #            smooth(subdf['value'].values, 0.8),#[:1001],
            #            label='train'
            #            )
            
            # Validation Loss Plot
            subdf = val_df[val_df['tag'] == metric]
            ax[i].plot(subdf['step'].values[0:3000:5],  # .iloc[:1001],
                       # subdf['value'].iloc[:1001],
                       smooth(subdf['value'].values, 0.9)[0:3000:5],
                       label='validation'
                       )
            
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel('')
            ax[i].yaxis.major.formatter.set_powerlimits((0, 0))
            ax[i].set_ylim([0, 2e-4])
            ax[i].set_xlim([0, 3000])

        # fig.legend(['Train', 'Validation'],
        #            loc='outside lower center',
        #            ncols=2,
        #            )
        ax[0].set_title('MLP')
        ax[0].set_ylabel('Loss')
        ax[1].set_title('MLP_WIN')
        ax[2].set_title('LSTM')

    plt.show()
