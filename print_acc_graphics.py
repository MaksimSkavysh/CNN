# # import matplotlib.pyplot as plt
# # import tkinter
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import math


def read_loss_and_acc_val(path):
    loss = list()
    acc = list()
    with open(path) as file:
        for line in file:
            if line.__contains__("val_acc") and line.__contains__("118409/118409"):
                loss_ = "val_loss: "
                acc_ = "- val_acc: "
                loss_start_index = line.find(loss_)
                acc_start_index = line.find(acc_)
                acc_before_value_index = acc_start_index + len(acc_)
                acc_end_index = acc_before_value_index + line[acc_before_value_index:].find(" ")
                loss.append(float(line[loss_start_index + len(loss_):acc_start_index].strip()))
                acc.append(float(line[acc_before_value_index:acc_end_index].strip()))
    return loss, acc


def read_loss_and_acc_train(path):
    loss = list()
    acc = list()
    with open(path) as file:
        for line in file:
            if line.__contains__(" loss: ") and line.__contains__(" acc: "):
                loss_ = " loss: "
                acc_ = "- acc: "
                loss_start_index = line.find(loss_)
                acc_start_index = line.find(acc_)
                acc_before_value_index = acc_start_index + len(acc_)
                acc_end_index = acc_before_value_index + line[acc_before_value_index:].find(" ")
                loss.append(float(line[loss_start_index + len(loss_):acc_start_index].strip()))
                acc.append(float(line[acc_before_value_index:acc_end_index].strip()))
    return loss, acc


def print_val(data_path, color):
    loss, acc = read_loss_and_acc_val(data_path)
    # loss2, acc2 = read_loss_and_acc_val('./out/vgg_im128_i2/vgg128_1.txt')
    # acc = acc2 + acc
    # loss = loss2 + loss

    # plt.subplot(211)
    # plt.title('accuracy')
    plt.plot(acc, color=color, label='alex_filter_3')

    # plt.subplot(212)
    # plt.title('loss')
    # plt.plot(loss, color=color, label='alex_filter_3')


def print_train(data_path, color):
    loss, acc = read_loss_and_acc_train(data_path)
    plt.plot(acc, color=color, label='alex_filter_3')


alex_filter_3 = "out/alex_f3_1/cnn_filter_3.txt"
alex_filter_5 = "out/alex_f5_1/cnn_filter_5.txt"
alex_filter_7 = "out/alex_f7_1/cnn_filter_7.txt"
alex_filter_9 = "out/alex_f9_1/cnn_filter_9.txt"
alex_filter_11 = "out/alex_f11_1/cnn_filter_11.txt"

vgg_im128_i3 = "./out/vgg_im128_i3/vgg_im128_i3.txt"
zfnet_1 = "./out/zfnet_1/zf.txt"

alex_32 = "out/alex_s32_2/cnn_32.txt"
alex_64 = "out/alex_s64_2/cnn_64.txt"
alex_128 = "out/alex_s128_2/cnn_128.txt"
alex_160 = "out/alex_s160_2/cnn_160.txt"
alex_256 = "out/alex_s256_2/cnn_256.txt"

# loss, acc = read_loss_and_acc_train(alex_filter_3)

# plt.figure(1)


# plt.subplot(411)
# plt.ylim(0.8, 0.95)
# plt.title('Image size: 32')
# print_val(alex_32, "red")
# plt.subplot(411)
# plt.ylim(0.8, 0.95)
# plt.title('Image size: 64')
# print_val(alex_64, "green")
# plt.subplot(412)
# plt.ylim(0.8, 0.95)
# plt.title('Image size: 128')
# print_val(alex_128, "blue")
# plt.subplot(413)
# plt.ylim(0.8, 0.95)
# plt.title('Image size: 160')
# print_val(alex_160, "black")
# plt.subplot(414)
# plt.title('Image size: 256')
# print_val(alex_256, "orange")

plt.subplot(511)
plt.title('Filter size: 3')
print_val(alex_filter_3, "red")
plt.subplot(512)
plt.title('Filter size: 5')
print_val(alex_filter_5, "green")
plt.subplot(513)
plt.title('Filter size: 7')
print_val(alex_filter_7, "blue")
plt.subplot(514)
plt.title('Filter size: 9')
print_val(alex_filter_9, "black")
plt.subplot(515)
plt.title('Filter size: 11')
print_val(alex_filter_11, "orange")


# plt.title('ZFNet accuracy')
# print_val(zfnet_1, "blue")


plt.show()
plt.savefig('myfig')
