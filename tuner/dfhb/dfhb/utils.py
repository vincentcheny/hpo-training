import heapq
import os
import copy
import shutil
import glob
heap=[]

def preprocess(tid, config, path):
	'''
	Parameters
	----------
	model: tensorflow.python.keras.engine.training.Model
	config: dict
		trial configuration
	path: str
		a path storing checkpointing folders 

	Return
	-------
	is_load: bool
	load_path: str
	save_path: str
	remain_trial_budget: int
	'''
	item = copy.deepcopy(config)
	item.pop('TRIAL_BUDGET')
	hash_code = str(abs(hash(str(item))))
	load_pattern = '*' + hash_code
	load_paths = glob.glob(os.path.join(path, load_pattern))
	trial_budget = int(config['TRIAL_BUDGET'])
	if trial_budget == 1 or len(load_paths) < 1:
		is_load = False
		load_path = ""
	else:
		is_load = True
		load_path = load_paths[-1]
	save_path = os.path.join(path, "-".join(tid, trial_budget, hash_code))
	remain_trial_budget = trial_budget
	remain_trial_budget -= int(load_path.split('/')[-1].split('-')[1]) if len(load_path)>0 else 0
	return is_load, load_path, save_path, remain_trial_budget


def postprocess(model, result, config, path, heap_size):
	'''
	Parameters
	----------
	model: tensorflow.python.keras.engine.training.Model
	result: dict
		Invalid Value:
			0.2
			{'default':0.2,'attr0':0.3,'attr1':46,'maximize':'attr2'}
		Valid Value:
			{'default':0.2,'attr0':0.3}
			{'default':0.2,'attr0':0.3,'attr1':46}
			{'default':0.2,'attr0':0.3,'attr1':10,'maximize':'attr0'}
			{'default':0.2,'attr0':0.3,'attr1':23.45,'maximize':['attr0','attr1']}
	config: dict
		trial configuration
	path: str
		a path storing checkpointing folders 
	heap_size: int
		maximum number of ckpt to keep for next round's loading
	'''
	if not isinstance(heap_size, int) or heap_size <= 0:
		raise ValueError(f"ValueError: heap_size should be a positive integer but gets a value {heap_size}.")
	global heap
	if len(heap) == heap_size:
		if hasattr(result, 'maximize')\
		 and (isinstance(result['maximize'], str) and result['maximize'] == 'default'\
		 or isinstance(result['maximize'], list) and 'default' in result['maximize']):
			compared_item = heapq.nsmallest(1,heap,key=lambda x:x[0]['default'])[0]
			if compared_item['default'] >= result['default']:
				return
		else:
			compared_item = heapq.nlargest(1,heap,key=lambda x:x[0]['default'])[0]
			if compared_item[0] <= result['default']:
				return
		try:
			if os.path.exists(compared_item[1]):
				shutil.rmtree(compared_item[1])
		except OSError as e:
			print("Error: %s : %s" % (compared_item[1], e.strerror))
		heap.remove(compared_item)
	item = copy.deepcopy(config)
	item.pop('TRIAL_BUDGET')
	save_dir = "-".join(config['TRIAL_BUDGET'], hash(item))
	save_path = os.path.join(path, save_dir)
	model.save(save_path)
	heapq.append((result, save_path))

