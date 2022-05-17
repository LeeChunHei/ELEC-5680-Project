import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_path", type=str, help="support multi log, separate by comma")
    arg_parser.add_argument("--focus_tag", type=str, help="separate by comma")
    arg_parser.add_argument("--max_iter", type=int)
    arg_parser.add_argument("--iter_step", type=int)
    arg_parser.add_argument("--title", type=str, default="result")
    arg_parser.add_argument("--output_path", type=str, default="./test.png")
    args = arg_parser.parse_args()

    log_paths = args.log_path.split(",")
    plot_datas = []
    print(log_paths)
    for log_path in log_paths:
        log_result = []
        with open(log_path) as log_f:
            for line in log_f:
                line_result = []
                data = line.split(" ")
                for d in data:
                    if d != "" and d != "\n":
                        line_result.append(d)
                log_result.append(line_result)

        draw_tag = args.focus_tag.split(",")
        draw_idx = []
        for tag in draw_tag:
            if tag in log_result[0]:
                draw_idx.append(log_result[0].index(tag))

        plot_data = [[] for i in draw_idx]
        for i in range(1, args.max_iter+1, args.iter_step):
            for j in range(len(draw_idx)):
                plot_data[j].append(float(log_result[i][draw_idx[j]]))
        plot_datas.append(plot_data)

    plt.figure(args.title.replace("_", " "))
    plt.title(args.title.replace("_", " "))
    plt.xlabel("Iteration")
    plt.ylabel(args.focus_tag.replace("_", " "))
    for j in range(len(plot_datas)):
        plot_data = plot_datas[j]
        for i in range(len(draw_idx)):
            plt.plot(range(0, args.max_iter, args.iter_step), plot_data[i])
    # plt.plot(range(1, opts.train_iters + 1), d_real_losses)
    # plt.plot(range(1, opts.train_iters + 1), D_Y_losses)
    # plt.plot(range(1, opts.train_iters + 1), D_X_losses)
    # plt.plot(range(1, opts.train_iters + 1), d_fake_losses)
    # plt.plot(range(1, opts.train_iters + 1), g_losses)
    plt.legend([path.split('/')[-1].replace("_", " ") for path in log_paths])
    # plt.legend(["d_real_loss", "D_Y_loss", "D_X_loss", "d_fake_loss", "g_loss"])
    plt.savefig(args.output_path)