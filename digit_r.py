# load pictures

#env
import os
import numpy as np
from scipy import ndimage
from IPython.display import display, Image
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline

#python current dir
#picture dir
pic_dir=os.getcwd()
def getfolders(filedir):
    data_folders=[os.path.join(filedir,x) for x in sorted(os.listdir(filedir))]
    return data_folders
train_folders=getfolders(pic_dir+'/notMNIST_large')
test_folders=getfolders(pic_dir+'/notMNIST_small')



#load letter
#image size
image_size=28
pixel_depth=225.0


def add_rows(dataset,label,i,l):
    dataset_one=np.ndarray(shape=(i+l,image_size,image_size),dtype=np.float32)
    dataset_one[0:i,:,:]=dataset[0:i]
    label_one=np.ndarray(shape=(i+l),dtype='|S1')
    label_one[0:i]=label[0:i]
    return dataset_one,label_one

def loadpics(data_folders):
    num_images=0
    dataset=np.ndarray(shape=(num_images,image_size,image_size),dtype=np.float32)
    label=np.ndarray(shape=(num_images),dtype='|S1')
    for folder in data_folders:
        image_files=os.listdir(folder)
        dataset,label=add_rows(dataset,label,num_images,len(image_files))
        for image in image_files:
            image_file=os.path.join(folder,image)
            try:
                image_data=(ndimage.imread(image_file).astype(float)-pixel_depth/2)/pixel_depth
                label_data=str(ord(os.path.basename(folder))-ord('A'))
                dataset[num_images,:,:]=image_data
                label[num_images]=label_data
                num_images+=1
            except IOError as e:
                pass
    return dataset,label
        
train_datasets,train_labels=loadpics(train_folders)
test_datasets,test_labels=loadpics(test_folders)


#save for sure
pickle_file='save'

try:
    f=open(pickle_file,'wb')
    save={
        'train_datasets':train_datasets,
        'train_labels':train_labels,
        'test_datasets':test_datasets,
        'test_labels':test_labels
    }
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('unable to save')
    raise


#test save
pickle_file='save'
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_datasets=save['train_datasets']
  train_labels=save['train_labels']
  test_datasets=save['test_datasets']
  test_labels=save['test_labels']


#shuffle
def randomize(dataset,label):
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset=dataset[permutation,:,:]
    shuffled_label=label[permutation]
    return shuffled_dataset,shuffled_label
train_datasets, train_labels = randomize(train_datasets, train_labels)
test_datasets, test_labels = randomize(test_datasets, test_labels)


sample_num=0
sample_image=train_datasets[sample_num,:,:]


#con
image_size=28
pixel_depth=225.0
train_size=100000
test_size=2000

def to5(dataset,label,size):
    new_dataset=np.ndarray(shape=(size,image_size,image_size*5),dtype=np.float32)
    new_label=np.ndarray(shape=(size,5),dtype=np.int32)
    for i in range(size):
        j=i*5
        new_dataset[i,:,0:image_size]=dataset[j,:,:]
        new_dataset[i,:,image_size:2*image_size]=dataset[j+1,:,:]
        new_dataset[i,:,2*image_size:3*image_size]=dataset[j+2,:,:]
        new_dataset[i,:,3*image_size:4*image_size]=dataset[j+3,:,:]
        new_dataset[i,:,4*image_size:5*image_size]=dataset[j+4,:,:]
        new_label[i,0]=int(label[j])
        new_label[i,1]=int(label[j+1])
        new_label[i,2]=int(label[j+2])
        new_label[i,3]=int(label[j+3])
        new_label[i,4]=int(label[j+4])
    return new_dataset,new_label

new_train_dataset,new_train_label=to5(train_datasets,train_labels,train_size)
new_test_dataset,new_test_label=to5(test_datasets,test_labels,test_size)



#start to model
import tensorflow as tf

#set everything
image_xsize,image_ysize=28,28*5
num_labels=11
batch_size=50
patch_size=5
depth1=32
depth2=66
num_hidden1=1024

sess = tf.InteractiveSession()
#x, y_ = new_train_dataset, new_train_label
x = tf.placeholder(tf.float32, shape=[None, image_xsize, image_ysize])
y_ = tf.placeholder(tf.float32, shape=[5,None, num_labels])


# functions
def weight_variable(shape):
    init=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def accuracy(predictions, labels):

    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])


# construct CNN
# c1 : onvolution size : 5 * 5 * 1 * 32
# m2 : batch_size * 14 * (14*5) * 32
# c3 : convolution size : 5 * 5 * 1 * 64
# m4 : batch_size * 7 * (7*5) * 64
# d5 : Dropout
# f6 : weight size 7 * (7*5) * 64 * 1024
# o7 : softmax weight size : 1024 * 11

#set 5 clfs
o7w1=weight_variable([num_hidden1,num_labels])
o7w2=weight_variable([num_hidden1,num_labels])
o7w3=weight_variable([num_hidden1,num_labels])
o7w4=weight_variable([num_hidden1,num_labels])
o7w5=weight_variable([num_hidden1,num_labels])

o7b1=weight_variable([num_labels])
o7b2=weight_variable([num_labels])
o7b3=weight_variable([num_labels])
o7b4=weight_variable([num_labels])
o7b5=weight_variable([num_labels])



#model
def model(data, keep_prob):
    #c1
    w1 = weight_variable([5,5,1,32])
    b1 = bias_variable([32])
    x = tf.reshape(data, [-1,28,28*5,1])
    c1 = tf.nn.relu(conv2d(x,w1) + b1)
    #m2
    m2 = max_pool_2x2(c1)
    #c3
    w3 = weight_variable([5,5,32,66])
    b3 = bias_variable([66])
    c3 = tf.nn.relu(conv2d(m2,w3) + b3)
    #m4
    m4 = max_pool_2x2(c3)
    #d5
    #keep_prob = tf.placeholder(tf.float32)
    d5 = tf.nn.dropout(m4, keep_prob)
    #f6
    w6 = weight_variable([7*7*5*66,1024])
    b6 = bias_variable([1024])
    f6 = tf.nn.relu(tf.matmul(tf.reshape(d5,[-1,7*7*5*66]),w6)+b6)
    #o7
    
    logits0 = tf.matmul(f6, o7w1) + o7b1
    logits1 = tf.matmul(f6, o7w2) + o7b2
    logits2 = tf.matmul(f6, o7w3) + o7b3
    logits3 = tf.matmul(f6, o7w4) + o7b4
    logits4 = tf.matmul(f6, o7w5) + o7b5
    return [logits0,logits1,logits2,logits3,logits4]
    


[logits0,logits1,logits2,logits3,logits4]=model(x,0.95)



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits0, y_[0])) +\
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits1, y_[1])) +\
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits2, y_[2])) +\
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits3, y_[3])) +\
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits4, y_[4]))
#for i in range(5):
#    loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[i],y_[:,i]))

sess.run(tf.global_variables_initializer())

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  
for i in range(train_size/batch_size):
    for j in range(5):
        labels = np.ndarray(shape=(5,batch_size,num_labels),dtype=np.int32)
        labels[j] = (np.arange(num_labels) == new_train_label[i*batch_size:(i*batch_size+batch_size),j:j+1]).astype(np.float32)
      
    train_step.run(feed_dict={x:new_train_dataset[i*batch_size:(i*batch_size+batch_size),:,:],y_:labels})   
    #train_step.run(feed_dict={x:new_train_dataset[i*100:(i*100+100),:,:],\
    #                          y_:[new_train_label[i*100:(i*100+100),0:1],\
    #                              new_train_label[i*100:(i*100+100),1:2],\
    #                              new_train_label[i*100:(i*100+100),2:3],\
    #                              new_train_label[i*100:(i*100+100),3:4],\
    #                              new_train_label[i*100:(i*100+100),4:5]]})
    
    
    #test
    if i%500==0:
        print('step',i)
        test_size=100
        
        accuracya=[]
    
        correct_prediction0 = tf.equal(tf.argmax(logits0,1), tf.argmax(y_[0],1))
        accuracya.append(tf.reduce_mean(tf.cast(correct_prediction0, tf.float32)))
    
        correct_prediction1 = tf.equal(tf.argmax(logits1,1), tf.argmax(y_[1],1))
        accuracya.append(tf.reduce_mean(tf.cast(correct_prediction1, tf.float32)))
    
        correct_prediction2 = tf.equal(tf.argmax(logits2,1), tf.argmax(y_[2],1))
        accuracya.append(tf.reduce_mean(tf.cast(correct_prediction2, tf.float32)))
    
        correct_prediction3 = tf.equal(tf.argmax(logits3,1), tf.argmax(y_[3],1))
        accuracya.append(tf.reduce_mean(tf.cast(correct_prediction3, tf.float32)))
    
        correct_prediction4 = tf.equal(tf.argmax(logits4,1), tf.argmax(y_[4],1))
        accuracya.append(tf.reduce_mean(tf.cast(correct_prediction4, tf.float32)))
    
        accuracya.append(accuracy0*accuracy1*accuracy2*accuracy3*accuracy4)
        
        
        for j in range(5):
            tslabels = np.ndarray(shape=(5,batch_size,num_labels),dtype=np.int32)
            tslabels[j] = (np.arange(num_labels) == new_test_label[0:batch_size,j:j+1]).astype(np.float32)
        for j in range(6):
            accuracy=accuracya[j]
            if j==5:
                print('total',j,accuracy.eval(feed_dict={x: new_test_dataset[0:batch_size], y_: tslabels}))
            else:
                print('ac',j,accuracy.eval(feed_dict={x: new_test_dataset[0:batch_size], y_: tslabels}))
            
            
#train_step.run(feed_dict={x:new_train_dataset,y_:new_train_label})    
#x, y_ = new_train_dataset, new_train_label
