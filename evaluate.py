import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_top_10_acc(predictions_arr = None):
	num_predictions = predictions_arr.shape[0]

	if num_predictions > num_samples:
		print('WARNING: number of predictions are higher than number of samples!!!')
		num_predictions = num_samples

	# count correct predictions
	num_correct_pred = 0
	for i in range(num_predictions):
		pro_id = predictions_arr[i,0]
		lig_list = list(predictions_arr[i,1:])

		truth_lig_id = ground_truth_dict[pro_id]
		if truth_lig_id in lig_list:
			num_correct_pred += 1

	acc = num_correct_pred / num_samples

	# print('accuracy:{:.3f}'.format(acc))

	return acc


grade_file = 'project_grading.txt'
with open(grade_file,'a') as f_grade_file:
	f_grade_file.write('{}\t{}\t{}\t{}\n'.format('team_id', 'student1_id', 'student2_id', 'accuracy'))

ground_truth_filename = 'test_ground_truth__2018_10_18__16_33_54.txt'
ground_truth_arr = np.loadtxt(ground_truth_filename, dtype=np.int, delimiter='\t', skiprows=1)

num_samples = ground_truth_arr.shape[0]

# prepare ground truth dictionary
ground_truth_dict = dict()
for i in range(num_samples):
	dict_key = ground_truth_arr[i,0]
	dict_value = ground_truth_arr[i,1]
	ground_truth_dict[dict_key] = dict_value


submission_path = '../student_submissions/'
for root, dirnames, filenames in os.walk(submission_path):
	student_folder_list = dirnames
	break


acc_list = list()
team_id_list = list()
student1_id_list = list()
student2_id_list = list()
for student_folder in student_folder_list:
	# print(student_folder)

	if len(student_folder) > 10:
		student1_id = student_folder.split('_')[0]
		student2_id = student_folder.split('_')[1]
	else:
		student1_id = student_folder
		student2_id = 'XXXXXXXXXX'

	team_id_list.append(student_folder)
	student1_id_list.append(student1_id)
	student2_id_list.append(student2_id)

	# print('{}: {} - {}'.format(student_folder, student1_id, student2_id))

	predictions_filename = submission_path + student_folder + '/test_predictions.txt'

	if os.path.exists(predictions_filename):
		if os.path.getsize(predictions_filename) > 0:
			try:
				pred_arr = np.loadtxt(predictions_filename, dtype=np.int, delimiter='\t', skiprows=1)

				if pred_arr.shape[1] == 11:
					# print('TEAM ID:{}, MESSAGE:{}'.format(student_folder, 'Correct file structure'))

					temp_acc = calculate_top_10_acc(predictions_arr = pred_arr)

				else:
					print('TEAM ID:{}, ERROR:{}'.format(student_folder, 'Wrong file structure!!!'))
					temp_acc = 0
			except:
				print('TEAM ID:{}, ERROR:{}'.format(student_folder, 'File format is different!!!'))
				temp_acc = 0

		else:
			print('TEAM ID:{}, ERROR:{}'.format(student_folder, 'Prediction file is empty!!!'))
			temp_acc = 0
	else:
		print('TEAM ID:{}, ERROR:{}'.format(student_folder, 'There is no prediction file!!!'))
		temp_acc = 0
	
	acc_list.append(temp_acc)
	with open(grade_file,'a') as f_grade_file:
		f_grade_file.write('{}\t{}\t{}\t{:.3f}\n'.format(student_folder, student1_id, student2_id, temp_acc))


acc_arr = np.array(acc_list)
# print(acc_arr.shape)

fig = plt.figure()
n, bins, patches = plt.hist(acc_arr, 50, facecolor='g', alpha=0.75)
plt.xlabel('Top-10 Accuracy')
plt.ylabel('Number of submissions')
plt.title('Histogram of Top-10 Accuracy')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()




