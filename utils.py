def plot(data={}, loc="visualization.pdf", x_label="", y_label="", title="", kind='line',
         legend=True, index_col=None, clip=None, moving_average=False):
    if all([len(data[key]) > 1 for key in data]):
        if moving_average:
            smoothed_data = {}
            for key in data:
                smooth_scores = [np.mean(data[key][max(0, i - 10):i + 1]) for i in range(len(data[key]))]
                smoothed_data['smoothed_' + key] = smooth_scores
                smoothed_data[key] = data[key]
            data = smoothed_data
        df = pd.DataFrame(data=data)
        ax = df.plot(kind=kind, legend=legend, ylim=clip)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(loc)
        plt.close()


def write_to_csv(data={}, loc="data.csv"):
    if all([len(data[key]) > 1 for key in data]):
        df = pd.DataFrame(data=data)
        df.to_csv(loc)


def plot_and_write(plot_dict, loc, x_label="", y_label="", title="", kind='line', legend=True,
                   moving_average=False):
    for key in plot_dict:
        plot(data={key: plot_dict[key]}, loc=loc + ".pdf", x_label=x_label, y_label=y_label, title=title,
             kind=kind, legend=legend, index_col=None, moving_average=moving_average)
        write_to_csv(data={key: plot_dict[key]}, loc=loc + ".csv")


def create_folder(folder_location, folder_name):
    i = 0
    while os.path.exists(os.getcwd() + folder_location + folder_name + str(i)):
        i += 1
    folder_name = os.getcwd() + folder_location + folder_name + str(i)
    os.mkdir(folder_name)
    return folder_name