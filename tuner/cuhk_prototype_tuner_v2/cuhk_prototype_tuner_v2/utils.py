import heapq
import os
import copy
import shutil
import glob
import hashlib
import json
from collections import defaultdict
import nni

def get_hash_code_from_config(config: dict):
	obj = copy.deepcopy(config)
	if 'TRIAL_BUDGET' in obj.keys():
		obj.pop('TRIAL_BUDGET')
	if "NUM_TRIAL_NEXT_ROUND" in obj.keys():
		obj.pop("NUM_TRIAL_NEXT_ROUND")
	hash_code = hashlib.md5(str(sorted(obj.items())).encode('utf-8')).hexdigest()[:16]
	return hash_code

def delete_last_line(path):
	with open(path, "rb+") as f:
		try:
			f.seek(-1, os.SEEK_END)
			while f.read() != b"\n":
				f.seek(-1, 1)
				f.truncate()
				f.seek(-1, 1)
			f.seek(-1, 1)
			f.truncate()
		except IOError as e:
			if e.errno == 22:
				print("[INFO] File pointer reaches the head of file.")
			else:
				raise

def preprocess(tid: str, config: dict, path: str):
	'''
	Parameters
	----------
	tid: str
		trial id
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

	Introduction
	------------
	Maintain the checkpoints in the path based on configuration.
	'''
	hash_code = get_hash_code_from_config(config)
	load_pattern = '*' + hash_code + '*'
	load_paths = glob.glob(os.path.join(path, load_pattern))
	trial_budget = int(config['TRIAL_BUDGET'])
	if trial_budget == 1 or len(load_paths) < 1:
		is_load = False
		load_path = ""
	else:
		is_load = True
		load_path = load_paths[-1]
	save_path = os.path.join(path, "-".join([tid, str(trial_budget), hash_code]))
	remain_trial_budget = trial_budget
	remain_trial_budget -= int(load_path.split('/')[-1].split('-')[1]) if len(load_path)>0 else 0
	nni.report_intermediate_result(os.path.abspath(path))
	return is_load, load_path, save_path, remain_trial_budget


def postprocess(eid, tid, result, config, path):
	'''
	Parameters
	----------
	eid: str
		experiment id
	tid: str
		trial id
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
		trial configuration, must contain keys 'NUM_TRIAL_NEXT_ROUND' and 'TRIAL_BUDGET'
	path: str
		a path storing checkpointing folders 
	
	Return
	------
	is_save: bool

	Introduction
	------------
	Maintain a heap of checkpoints with a given size.

	'''
	heap_size = int(config['NUM_TRIAL_NEXT_ROUND']) # maximum number of ckpt to keep
	i = str(config['TRIAL_BUDGET'])
	assert isinstance(heap_size, int) and heap_size >= 0, f"Heap size should be a non-negative integer but get a value {heap_size}."
	
	record_path = os.path.join(path, eid+".json")
	heap = defaultdict(list)
	if os.path.exists(record_path):
		with open(record_path, 'r') as f:
			records = f.readline()
			heap = defaultdict(list, json.loads(records))
			max_key = int(max(heap.keys(), key=int))
			if int(i) < max_key: # Assume that each round is run in series, need to use another metrics if it's not
				heap = defaultdict(list)
	
	if len(heap[i]) >= heap_size and heap_size != 0:
		if 'maximize' in result.keys():
			if (isinstance(result['maximize'], str) and result['maximize'] == 'default' \
			  or isinstance(result['maximize'], list) and 'default' in result['maximize']):
				compared_item = heapq.nsmallest(1,heap[i],key=lambda x:x[0]['default'])[0]
				if compared_item[0]['default'] >= result['default']:
					return False
			else:
				raise ValueError(f"Unexpected value of {result}")
		else:
			compared_item = heapq.nlargest(1,heap[i],key=lambda x:x[0]['default'])[0]
			if compared_item[0]['default'] <= result['default']:
				return False
		try:
			if os.path.exists(compared_item[1]):
				shutil.rmtree(compared_item[1])
		except OSError as e:
			print("Error: %s : %s" % (compared_item[1], e.strerror))
		heap[i].remove(compared_item)

	hash_code = get_hash_code_from_config(config)
	save_path = os.path.join(path, "-".join([tid, str(config['TRIAL_BUDGET']), hash_code]))
	heap[i].append([result, save_path])
	with open(record_path, 'w') as f:
		f.write(str(dict(heap)).replace("\'", '\"'))
	return True
