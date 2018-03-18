#!/usr/bin/python

import os
from os import listdir, getcwd
from os.path import join
import shutil 
import multiprocessing
import time
import sys

CURRENT = os.getcwd()
ROOT = "/home/ubuntu/DAC_Challenge/datasets/detectnet/"

TRAIN = ROOT + "training/"
VAL = ROOT + "testing/"

DEST_ROOT = "/home/ubuntu/DAC_Challenge/squeezeDet/data/KITTI/"
DEST_TRAIN_IMAGE = DEST_ROOT + "training/image_2/"  
DEST_VAL_IMAGE = DEST_ROOT + "testing/label_2/"
DEST_TRAIN_LABEL = DEST_ROOT + "training/label_2/"
DEST_VAL_LABEL = DEST_ROOT + "testing/image_2/"

def setup_train_images():

	p = multiprocessing.current_process()
	print 'Starting:', p.name, p.pid
	start_time = time.time()
	sys.stdout.flush()
	for cls in listdir(TRAIN + "images/"):
		src = "{0}/{1}/*".format(TRAIN + "images/", cls)
		cmd = "cp {0} {1}".format(src, DEST_TRAIN_IMAGE)
		print cmd
		os.system(cmd)
	elapsed_time = time.time() - start_time
	print "Exiting {%s}[%d]; Time taken %.2f seconds" %(p.name, p.pid, elapsed_time)
	sys.stdout.flush()		

def setup_val_images():

	p = multiprocessing.current_process()
	print 'Starting:', p.name, p.pid
	start_time = time.time()
	sys.stdout.flush()
	for cls in listdir(VAL + "images/"):
		src = "{0}/{1}/*".format(VAL + "images/", cls)
		cmd = "cp {0} {1}".format(src, DEST_VAL_IMAGE)
		print cmd
		os.system(cmd)
	elapsed_time = time.time() - start_time
	print "Exiting {%s}[%d]; Time taken %.2f seconds" %(p.name, p.pid, elapsed_time)
	sys.stdout.flush()		

def setup_val_labels():

	p = multiprocessing.current_process()
	print 'Starting:', p.name, p.pid
	start_time = time.time()
	sys.stdout.flush()
	for cls in listdir(VAL + "labels"):
		src = "{0}/{1}/*".format(VAL + "labels", cls)
		cmd = "cp {0} {1}".format(src, DEST_VAL_LABEL)
		print cmd
		os.system(cmd)
	elapsed_time = time.time() - start_time
	print "Exiting {%s}[%d]; Time taken %.2f seconds" %(p.name, p.pid, elapsed_time)
	sys.stdout.flush()		

def setup_train_labels():

	p = multiprocessing.current_process()
	print 'Starting:', p.name, p.pid
	start_time = time.time()
	sys.stdout.flush()
	for cls in listdir(TRAIN + "labels/"):
		src = "{0}/{1}/*".format(TRAIN + "labels", cls)
		cmd = "cp {0} {1}".format(src, DEST_TRAIN_LABEL)
		print cmd
		os.system(cmd)
	elapsed_time = time.time() - start_time
	print "Exiting {%s}[%d]; Time taken %.2f seconds" %(p.name, p.pid, elapsed_time)
	sys.stdout.flush()		

			
if __name__ == '__main__':

    ti = multiprocessing.Process(name='setup_train_images', target=setup_train_images)
    ti.daemon = True

    vi = multiprocessing.Process(name='setup_val_images', target=setup_val_images)
    vi.daemon = True

    tl = multiprocessing.Process(name='setup_train_labels', target=setup_train_labels)
    ti.daemon = True

    vl = multiprocessing.Process(name='setup_val_labels', target=setup_val_labels)
    vl.daemon = True

    ti.start()
    time.sleep(5)
    tl.start()
    time.sleep(30)
    vi.start()
    time.sleep(5)
    vl.start()

    vl.join()
    ti.join()
    vi.join()
    tl.join()


	
