import matplotlib.pyplot as plt
import json
import linecache
import os
import sys

def load_results(name):
    x, y, x_cu, y_cu = [], [], [0.0], [0.0]
    best_acc = best_time = t = 0.0
    count = 0
    suffix = os.path.splitext(name)[1]
    
    if name.split('\\')[-2] == "imagenet":
        CUT_OFF_TIME = 60 * 40 # extend or cut of the last trial to this time
    else:
        CUT_OFF_TIME = 60 * 10
    if suffix == '.log':
        # Deal with dragonfly result
        data = linecache.getline(name, 7)
        start_pos = data.find('query_true_vals') + len('query_true_vals') + 1
        end_pos = data.find('query_vals') - len(', ')
        data = eval(data[start_pos:end_pos])
        for trial in data:
            acc = trial[1]
            if name.split('\\')[-2] == "imagenet":
                trial_time = -trial[0] * 60.0
            else:
                trial_time = -trial[0] / 60.0
            t += trial_time
            x.append(trial_time)
            y.append(acc)
            x_cu.append(t)
            if acc >= y_cu[count]:
                y_cu.append(acc)
                best_acc = acc
                best_time = trial_time
            else:
                y_cu.append(y_cu[count])
            count += 1
            if t > CUT_OFF_TIME:
                x_cu[-1] = CUT_OFF_TIME
                break
    else:
        # Deal with NNI result
        with open(name) as json_file:
            data = json.load(json_file)
            for trial in data['trialMessage']:
                if trial['status'] == "SUCCEEDED":
                    acc = float(trial['finalMetricData'][0]['data'].strip('"'))
                    trial_time = (trial['endTime'] - trial['startTime']) / 60000.0
                elif trial['status'] == "RUNNING" and len(trial['intermediate']) > 0:
                    acc = float(trial['intermediate'][-1]['data'].strip('"'))
                    trial_time = (trial['intermediate'][-1]['timestamp'] - trial['startTime']) / 60000.0
                else:
                    continue
                t += trial_time
                x.append(trial_time)
                y.append(acc)
                x_cu.append(t)
                if acc >= y_cu[count]:
                    y_cu.append(acc)
                    best_acc = acc
                    best_time = trial_time
                else:
                    y_cu.append(y_cu[count])
                count += 1
            if t < CUT_OFF_TIME:
                x_cu.append(CUT_OFF_TIME)
                y_cu.append(y_cu[-1])
            else:
                x_cu[-1] = CUT_OFF_TIME

    location = '\\'.join(name.split('\\')[-2:])
    print(f"{location}\nBest acc: {best_acc:.3f}, best time: {best_time:.2f}min")
    return x, y, x_cu, y_cu


def plot_path(path):
    result = []
    for file in os.listdir(path):
        suffix = os.path.splitext(file)[1]
        if suffix in [".log", ".json"]:
            x,y,x_cu,y_cu = load_results(os.path.join(path, file))
            if suffix == ".log":
                label = 'Dragonfly'
            else:
                with open(os.path.join(path, file)) as json_file:
                    data = json.load(json_file)
                    if "advisor" in data["experimentParameters"]["params"].keys():
                        tuner = data["experimentParameters"]["params"]["advisor"]["builtinAdvisorName"]
                    else:
                        tuner = data["experimentParameters"]["params"]["tuner"]["builtinTunerName"]
                    label = 'NNI_' + tuner
            result.append([x,y,x_cu,y_cu,label])
    
    if not result:
        return None
    elif len(result) > 1 and result[1][4] == "Dragonfly":
        result[0], result[1] = result[1], result[0]
    color = ["red", "blue", "green"]

    # Print cumulative_best_acc
    for i in range(len(result)):
        plt.plot(result[i][2], result[i][3], color[i%len(color)], label=result[i][4], linestyle="--")
    plt.title(" & ".join([res[4] for res in result]))
    plt.xlabel("Total Tuning Time(min)")
    plt.ylabel("Accuracy(%)")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, "cumulative_best_acc.png"))
    plt.clf()

    # Print history_acc
    for i in range(len(result)):
        plt.scatter(result[i][0], result[i][1], alpha=0.6, c=color[i%len(color)], label=result[i][4])
    plt.title(" & ".join([res[4] for res in result]))
    plt.xlabel("Single Trial Time(min)")
    plt.ylabel("Accuracy(%)")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, "history_acc.png"))
    plt.clf()	


if __name__ == "__main__":
    dirname, filename = os.path.split(os.path.realpath(sys.argv[0]))
    for dir in os.listdir(dirname):
        real_dir = os.path.join(dirname,dir)
        if os.path.isdir(real_dir): # exclude files
            plot_path(real_dir)