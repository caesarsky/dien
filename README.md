# dien

# document link:  
https://docs.google.com/document/d/1F5bXb-Tb1LnPraOv09UI0Il_P_9cND-MywXpEDEwQJI/edit  
<br/>

# script from origin code that is not used:  
local_aggretor.py, process_data.py, split_by_user.py. 

# running instruction:  
enable virtual environment  

unzip data-collection.zip  
./run.sh  

or  
cd script/  
python data_prepare.py  
python generate_voc.py  
python train.py train DIEN  
or running in the background:  
python train.py train DIEN > dien_train.log 2>&1 &  


You can modify run.sh if you want to add extra config like seed, train diffferent model, just testing, check doc for extra running command introduction.  
<br/>


# feature config file:  
query_list: list of query features which do not have related history features  
item_feature_list: list of item features which have related history features  
history_list: list of history feature  
each item feature must have its related history in history_list, or move it to query_list  
the order of features in item_feature_list must be consistent with their related history features in history_list  

maxlen: the max length of history that will be extracted and used   
test_iter: how often to run test data  
Embedding_dim, batch_size, learning_rate_decay: common trainning parameter  <br/> 



# data preparation:  
header:  
label|...query features...|mid|...features that have history...|mid_history|...related history...  
query features:  
features without related history stream feature  
eg: uid, query, province, age  
mid: item id.  
features that have history: category, price etc  
related history: history of category, price, etc. <br/> 
the order of extra feature need to be relatively consistent with extra feature history  
Eg:  
label|uid|query|mid|category|price|mid_history|category_history|price_history  <br/> 

# feature_voc.pkl files:  
each feature including uid and mid obtains a voc file that contains the encoding of it for mathematical calculation in the model.  


# item-info:  
index file connect each item with its extra features  
data iterator will read this file and create a dict for item search  
header:  
mid|...extra features...  
Eg:  
mid|category|price  <br/> 
 
# reviews-info:  
index file contains action of user giving review score.  
data iterator will read this file and create negative item samples and history for each data sample.  
header:  
uid|mid|review score(review score is not used in the model)  





