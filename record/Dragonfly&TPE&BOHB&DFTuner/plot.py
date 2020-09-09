import matplotlib.pyplot as plt
import json
import linecache
import os
import sys

def load_results(name):
    x, y1, y2, x_cu, y_cu1, y_cu2 = [], [], [], [0.0], [0.0], [0.0]
    best_acc1 = best_acc2 = best_time1 = best_time2 = t = 0.0
    count = 0
    with open(name) as f:
        for line in f:
            train_acc, train_top5_acc, val_acc, val_top5_acc, spent_time = [float(data) for data in line.split()[0:5]]
            trial_time = abs(spent_time)
            t += trial_time
            # if t > 600: 
            #     t=600
            x.append(trial_time)
            y1.append(val_top5_acc)
            y2.append(val_acc)
            x_cu.append(t)
            if val_top5_acc >= y_cu1[count]:
                y_cu1.append(val_top5_acc)
                best_acc1 = val_top5_acc
                best_time1 = trial_time
            else:
                y_cu1.append(y_cu1[count])
            if val_acc >= y_cu2[count]:
                y_cu2.append(val_acc)
                best_acc2 = val_acc
                best_time2 = trial_time
            else:
                y_cu2.append(y_cu2[count])
            count += 1
            if t > 800: 
                break
        
    location = '\\'.join(name.split('\\')[-2:])
    print(f"{location}\nBest acc1: {best_acc1:.3f}, best time1: {best_time1:.2f}min Best acc2: {best_acc2:.3f}, best time2: {best_time2:.2f}min")
    return x, y1, y2, x_cu, y_cu1, y_cu2


def plot_path(path):
    result = []
    for file in os.listdir(path):
        suffix = os.path.splitext(file)[1]
        if suffix != '.log': continue
        prefix = os.path.splitext(file)[0]
        label = prefix.split('-')[-1]
        x, y1, y2, x_cu, y_cu1, y_cu2 = load_results(os.path.join(path, file))
        result.append([x, y1, y2, x_cu, y_cu1, y_cu2,label])
    
    if not result:
        return None
    color = ["red", "cornflowerblue", "lightgreen", 'grey', 'coral', 'gold']

    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    folder_name = path.split('\\')[-1].split('-')
    model_name = folder_name[0]
    dataset_name = folder_name[1]
    comments = folder_name[2] if len(folder_name) > 2 else None
    plt.suptitle(f"Model:{model_name} Dataset:{dataset_name} Comments:{comments if comments else 'None'}", fontsize=16)
    # plt.subplot(221)
    # # Print cumulative_best_acc
    # for i in range(len(result)):
    #     # plt.plot(result[i][3], result[i][4], color[i%len(color)], label=result[i][-1], linestyle="--")
    #     plt.plot(result[i][3], result[i][4], label=result[i][-1], linestyle="--")
    #     # plt.plot(result[i][3], [0.0]+result[i][1], label=result[i][-1], linestyle="--")
    #     # for a, b in zip(result[i][3], [0.0]+result[i][1]):  
    #     #     plt.text(a, b, round(b,4),ha='center', va='bottom', fontsize=10)
    # plt.title("Cumulative Best Top-5 Accuracy")
    # # plt.title("Top-5 Accuracy")
    # plt.xlabel("Total Tuning Time(min)")
    # plt.ylabel("Accuracy(%)")
    # plt.legend(loc='lower right')
    # # plt.savefig(os.path.join(path, "cumulative_best_acc_top5.png"))
    # # plt.clf()

    plt.subplot(211)
    # Print cumulative_best_acc
    for i in range(len(result)):
        # plt.plot(result[i][3], result[i][5], color[i%len(color)], label=result[i][-1], linestyle="--")
        plt.plot(result[i][3], result[i][5], label=result[i][-1], linestyle="--")
        # plt.plot(result[i][3], [0.0]+result[i][2], label=result[i][-1], linestyle="--")
        # for a, b in zip(result[i][3], [0.0]+result[i][2]):  
        #     plt.text(a, b, round(b,4),ha='center', va='bottom', fontsize=10)
    plt.title("Cumulative Best Top-1 Accuracy")
    # plt.title("Top-1 Accuracy")
    plt.xlabel("Total Tuning Time(min)")
    plt.ylabel("Accuracy(%)")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='lower right')

    # plt.subplot(223)
    # # Print history_acc
    # for i in range(len(result)):
    #     # plt.scatter(result[i][0], result[i][1], alpha=0.6, c=color[i%len(color)], label=result[i][-1])
    #     plt.scatter(result[i][0], result[i][1], alpha=0.6, label=result[i][-1])
    # plt.title("Top-5 Accuracy-Runtime Distribution")
    # plt.xlabel("Single Trial Time(min)")
    # plt.ylabel("Accuracy(%)")
    # plt.legend(loc='lower right')

    plt.subplot(212)
    # Print history_acc
    for i in range(len(result)):
        # plt.scatter(result[i][0], result[i][2], alpha=0.6, c=color[i%len(color)], label=result[i][-1])
        plt.scatter(result[i][0], result[i][2], alpha=0.6, label=result[i][-1])
    plt.title("Top-1 Accuracy-Runtime Distribution")
    plt.xlabel("Single Trial Time(min)")
    plt.ylabel("Accuracy(%)")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='lower right')
    fig = plt.gcf()
    fig.set_size_inches(12, 9)
    # plt.savefig(os.path.join(path, "result.png"), dpi=300)
    
    plt.rcParams['savefig.dpi'] = 200
    fig.savefig(os.path.join(path, "result.png"))
    # plt.show()
    plt.clf()	


if __name__ == "__main__":
    dirname, filename = os.path.split(os.path.realpath(sys.argv[0]))
    for dir in os.listdir(dirname):
        real_dir = os.path.join(dirname,dir)
        if os.path.isdir(real_dir): # exclude files
            plot_path(real_dir)