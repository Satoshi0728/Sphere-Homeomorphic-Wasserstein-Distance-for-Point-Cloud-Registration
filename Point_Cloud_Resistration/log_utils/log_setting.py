import os


"""
ログの記録用
"""
class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()


#============================================================================
"""
ログの記録用
"""
def _init_(log_dir, point_clouds_save_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	if not os.path.exists(point_clouds_save_dir):
		os.makedirs(point_clouds_save_dir)
	if not os.path.exists(log_dir + '/' + 'models'):
		os.makedirs(log_dir + '/' + 'models')
	#point cloudの保存用
	if not os.path.exists(point_clouds_save_dir + '/' + 'initial_source_data'):
		os.makedirs(point_clouds_save_dir + '/' + 'initial_source_data')
	if not os.path.exists(point_clouds_save_dir + '/' +  'target_data'):
		os.makedirs(point_clouds_save_dir + '/' +  'target_data')
	if not os.path.exists(point_clouds_save_dir + '/' + 'transformed_source_data'):
		os.makedirs(point_clouds_save_dir + '/' + 'transformed_source_data')
