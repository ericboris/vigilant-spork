pred = open('../../data/dev_predict', 'r')
ans = open('../../data/cleaned_data/dev_answer.txt', 'r')

pred_line = pred.readlines()
ans_line = ans.readlines()

total = 0
accurate = 0
for i in range(len(pred_line)):
    total += 1
    predicted = pred_line[i]
    predicted = [char for char in predicted]
    answer = ans_line[i]
    if answer in predicted:
        accurate += 1
print('total: ' + str(total))
print('accurate: ' + str(accurate))
print('percent: ' + str(float(accurate/total * 100)))