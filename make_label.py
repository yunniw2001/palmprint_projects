import os


def make_text(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'a+')
    num = 0
    for item in files:
        if num>=4800:
            break
        num+=1
        tmp_item = list(item[0:5])
        if tmp_item[0] == '1':
            tmp_item[0] = '0'
        tmp_item = ''.join(tmp_item)
        tmp_item = int(tmp_item)
        belong_index = (tmp_item - 1) // 10
        write_file.write(item + ' ' + str(belong_index) + '\n')
    write_file.close()

def make_text_save(text_dir,epoch, pre_idx,tain_idx,train_loss):
    write_file = open(text_dir, 'w')
    write_file.write(str(epoch)+'\n')
    write_file.write(str(pre_idx )+ '\n')
    write_file.write(str(tain_idx)+'\n')
    write_file.write(str(train_loss)+'\n')
    write_file.close()

def read_txt(text_dir):
    file = open(text_dir,'r')
    lines = file.readlines()
    epoch = int(lines[0])
    per_idx = int(lines[1])
    cur_idx =lines[2][1:-2].split(',')
    for i in range(len(cur_idx)):
        cur_idx[i] = int(cur_idx[i])
    train_loss = lines[3][1:-2].split(',')
    for i in range(len(train_loss)):
        train_loss[i] = float(train_loss[i])
    return epoch, per_idx, cur_idx, train_loss


if __name__ == '__main__':
    # make_text('E:\digital_image_processing\datasets\\tongji\ROI\session2_rename\\',
    #           'E:\digital_image_processing\data_1\\tongji_cross_subject\\train_label.txt')
    # make_text_save('E:\digital_image_processing\data_1\\tongji_cross_subject\\data.txt',1,9,[1,2,3],[0.89,0.9])
    a,b,c,d = read_txt('E:\digital_image_processing\data_1\\tongji_cross_subject\\data.txt')
    print(a)
    print(b)
    print(c)
    print(d)

