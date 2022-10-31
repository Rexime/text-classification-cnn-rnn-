# 读取数据集
import pandas as pd
df=pd.read_csv("train.csv")
df_val=pd.read_csv("dev.csv")

# 数据预处理
df["content"]=df["Title"]+" "+df["Description"]
df_val["content"]=df_val["Title"]+" "+df_val["Description"]
df_val_num=df_val.shape[0]

df=pd.concat([df,df_val],axis=0)
df=df[["Class Index","content"]]
df["category"]=df["Class Index"]

df=df.dropna(axis=0) #丢弃有空值的列

num_classes = len(df["category"].unique())


# 分词
import jieba
sentence=[[j.lower() for j in i.split(" ")] for i in df["content"]]


# 训练word2vec词向量，size为词向量长度，迭代次数为10
import pandas as pd
import gensim
w2v_model = gensim.models.Word2Vec(sentence, size=128, iter=10, min_count=0)
word_vectors = w2v_model.wv
w2v_model.save("w2v")
w2v_model=gensim.models.Word2Vec.load("w2v")


# 导入实现神经网络必要的包
from keras.layers import *
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf


# 将文本转化成向量,将标签onehot,对文本向量以最大长度补零
x_train=sentence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)  #统计每个词对应的数字，以便于将文本转化成向量
train_sequence = tokenizer.texts_to_sequences(x_train)#将所有的文本转化成向量
MAX_SEQUENCE_LENGTH=128 #最大长度
EMBEDDING_DIM = 128 #向量维度
y_train =df["category"]
y_train = to_categorical(y_train)  #将标签 one-hot
y_train = y_train.astype(np.int32)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
train_pad = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH) #将每条文本按照最大长度补0


# 统计每个单词应该对应哪一条向量
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM), dtype=np.float32)
not_in_model = 0
in_model = 0
embedding_max_value = 0
embedding_min_value = 1
not_words = []

for word, i in word_index.items():
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.array(w2v_model[word])
        embedding_max_value = max(np.max(embedding_matrix[i]), embedding_max_value)
        embedding_min_value = min(np.min(embedding_matrix[i]), embedding_min_value)
    else:
        not_in_model += 1
        not_words.append(word)


# 用keras定义一个词嵌入层，并把刚才的矩阵加进去
embed = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable = True)
      # 定义一个词嵌入层,将句子转化成对应的向量


# 将验证集划分出来
# train_data, val_data, train_y, val_y = train_test_split(train_pad, y_train, test_size=0.2, random_state=43)
train_data=train_pad[:-df_val_num]
train_y=y_train[:-df_val_num]
val_data=train_pad[-df_val_num:]
val_y=y_train[-df_val_num:]








# 定义textcnn模型
def get_cnnmodel(embedding, class_num=5):
    inputs_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,))#设置输入向量维度
    sentence =(embedding(inputs_sentence))#定义词嵌入层
    con=Conv1D(256, 5, padding='same')(sentence)
    maxp=MaxPooling1D(3, 3, padding='same')(con)
    con=Conv1D(128, 5, padding='same')(maxp)
    maxp=MaxPooling1D(3, 3, padding='same')(con)
    con=Conv1D(64, 3, padding='same')(maxp)
    fla=Flatten()(con)
    drop=Dropout(0.1)(fla)
    bn=BatchNormalization()(drop)
    ds=Dense(256, activation='relu')(bn)
    dp=Dropout(0.1)(ds)
    output = Dense(class_num, activation='softmax')(dp)#softmax层
    model = Model(inputs=[inputs_sentence], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#定义损失函数，优化器，评分标准
    model.summary()
    return model
# model = get_cnnmodel(embed)


# 开始训练
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = get_cnnmodel(embed)
callbacks = [EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10),
			 ModelCheckpoint("textcnn.hdf5", monitor='val_acc',
							 mode='max', verbose=0, save_best_only=True)]
#设置模型提前停止,停止的条件是验证集val_acc两轮已经不增加,保存验证集val_acc最大的那个模型,名称为new_cnn.hdf5
history=model.fit(train_data,train_y, batch_size=16, epochs=5, callbacks=callbacks,validation_data=(val_data,val_y))


# 加载最优的模型并且测试验证集
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
test_data=val_data
test_y=val_y
lstm_attention= load_model("textcnn.hdf5")
testpre=lstm_attention.predict([test_data])
tpre=np.argmax(testpre,axis=1)
testy=np.argmax(test_y,axis=1)


from sklearn.metrics import classification_report
print (classification_report(testy,tpre,digits=4))

import matplotlib.pyplot as plt

val_loss = history.history['val_loss']
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.title('Loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()

plt.cla()
val_loss = history.history['val_acc']
loss = history.history['acc']
epochs = range(1, len(loss) + 1)

plt.title('acc')
plt.plot(epochs, loss, 'red', label='Training acc')
plt.plot(epochs, val_loss, 'blue', label='Validation acc')
plt.legend()
plt.show()


# 打印混淆矩阵和各个类别的准确率
def 给测试集输出指标(val_data):
    testpre=lstm_attention.predict([test_data])
    ypre=np.argmax(testpre,axis=1)
    ytrue=np.argmax(test_y,axis=1)
    def leibie_acc(test_pre,y_test):
        test_pre=np.array(test_pre)
        y_test=np.array(y_test)
#         label2汉字=dict(zip(d_category_to_number.values(), d_category_to_number.keys()))

        print ("准确率:",sum((test_pre==y_test).astype(int))/len(y_test))
        for i in set(list(y_test)):

            tmp=(test_pre[np.where(y_test==i)]==y_test[np.where(y_test==i)]).astype(int)
            acc=sum(tmp)/len(tmp)
            print (i,"类别准确率为:",acc,"数量为：",len(tmp))
        return


    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    def 混洗矩阵(valp,valy):
        sns.set()
        np.set_printoptions(suppress=True)
        f,ax=plt.subplots()
        valp=np.array(valp)
        valy=np.array(valy)
        C2= confusion_matrix(valy , valp, labels=list(range(1,len(set(df["category"]))+1)))
        print(C2) #打印出来看看
        sns.heatmap(C2,annot=True,ax=ax,fmt='.20g') #画热力图

        ax.set_title('confusion matrix') #标题
        ax.set_xlabel('predict') #x轴
        ax.set_ylabel('true') #y轴

    def 评价指标(val_data,val_y):
    #     valpre=model_load.predict(val_data)
        leibie_acc(val_data,val_y)
        混洗矩阵(val_data,val_y)

    评价指标(ypre,ytrue)
给测试集输出指标(test_data)









# LSTM+Attention模型
def attention_3d_block(inputs):

    input_dim = int(inputs.shape[2])

    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, train_data.shape[1]))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(train_data.shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def get_lstmmodel(embedding, class_num=len(set(df["category"]))):
    inputs_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,))  #设置输入向量维度
    sentence =(embedding(inputs_sentence))  #定义词嵌入层
    context1 = Bidirectional(LSTM(64, return_sequences=True))(sentence)  # 双向lstm层,lstm神经元维度为64
    atten = attention_3d_block(context1) #给lstm1层加上注意力机制
    atten = Flatten()(atten)
    x = Dense(100, activation='relu')(atten)# 全连接层,全连接层神经元维度为100
#     x = Dense(100, activation='relu')(x)# 全连接层,全连接层神经元维度为100
    output = Dense(5, activation='softmax')(atten)  #softmax层
    model = Model(inputs=[inputs_sentence], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #定义损失函数，优化器，评分标准
    model.summary()
    return model

model = get_lstmmodel(embed)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = get_lstmmodel(embed)
callbacks = [EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10),
			 ModelCheckpoint("LSTM.hdf5", monitor='val_acc',
							 mode='max', verbose=0, save_best_only=True)]
#设置模型提前停止,停止的条件是验证集val_acc两轮已经不增加,保存验证集val_acc最大的那个模型,名称为new_cnn.hdf5
history=model.fit(train_data,train_y, batch_size=16, epochs=5, callbacks=callbacks,validation_data=(val_data,val_y))



# 加载模型，测试验证集
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
lstm_attention= load_model("LSTM.hdf5")
testpre=lstm_attention.predict([test_data])

# pre=[]
tpre=np.argmax(testpre,axis=1)
testy=np.argmax(test_y,axis=1)

from sklearn.metrics import classification_report

print(classification_report(testy, tpre, digits=4))
import matplotlib.pyplot as plt

val_loss = history.history['val_loss']
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.title('Loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()

plt.cla()
val_loss = history.history['val_acc']
loss = history.history['acc']
epochs = range(1, len(loss) + 1)

plt.title('acc')
plt.plot(epochs, loss, 'red', label='Training acc')
plt.plot(epochs, val_loss, 'blue', label='Validation acc')
plt.legend()
plt.show()



# 打印混淆矩阵和各个类别的准确率
def 给测试集输出指标(val_data):
    testpre=lstm_attention.predict([test_data])
    ypre=np.argmax(testpre,axis=1)
    ytrue=np.argmax(test_y,axis=1)
    def leibie_acc(test_pre,y_test):
        test_pre=np.array(test_pre)
        y_test=np.array(y_test)
#         label2汉字=dict(zip(d_category_to_number.values(), d_category_to_number.keys()))

        print ("准确率:",sum((test_pre==y_test).astype(int))/len(y_test))
        for i in set(list(y_test)):

            tmp=(test_pre[np.where(y_test==i)]==y_test[np.where(y_test==i)]).astype(int)
            acc=sum(tmp)/len(tmp)
            print (i,"类别准确率为:",acc,"数量为：",len(tmp))
        return


    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    def 混洗矩阵(valp,valy):
        sns.set()
        np.set_printoptions(suppress=True)
        f,ax=plt.subplots()
        valp=np.array(valp)
        valy=np.array(valy)
        C2= confusion_matrix(valy , valp, labels=list(range(1,len(set(df["category"]))+1)))
        print(C2) #打印出来看看
        sns.heatmap(C2,annot=True,ax=ax,fmt='.20g') #画热力图

        ax.set_title('confusion matrix') #标题
        ax.set_xlabel('predict') #x轴
        ax.set_ylabel('true') #y轴

    def 评价指标(val_data,val_y):
    #     valpre=model_load.predict(val_data)
        leibie_acc(val_data,val_y)
        混洗矩阵(val_data,val_y)

    评价指标(ypre,ytrue)
给测试集输出指标(test_data)







# 输出测试集结果
# import jieba
cnn_model= load_model("textcnn.hdf5")
def getleibie(text):
#     d=dict(zip(d_category_to_number.values(),d_category_to_number.keys()))
#     print ([i for i in list(jieba.cut(text)) if i!=" "])
    x=tokenizer.texts_to_sequences([[i.lower() for i in text.split(" ")]])
    pre_X = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH) #将每条文本按照最大长度补0
    return np.argmax(cnn_model.predict(pre_X),axis=-1)[0]
#getleibie("i love you too")

df_test=pd.read_csv("test.csv")

from tqdm import tqdm
tmppre=[]
for i,row in tqdm(df_test.iterrows()):
    tmppre.append(getleibie(row["Title"]+" "+row["Description"]))

df_test["Class Index"]=tmppre
df_test.to_csv("submit.csv",header=True,index=False,encoding="utf-8-sig")
