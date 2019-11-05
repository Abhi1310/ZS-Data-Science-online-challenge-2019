import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='notebook')


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


test.loc[test['range_of_shot'] == -1,'range_of_shot'] = 4
test.loc[test['area_of_shot'] == -1,'area_of_shot'] = 0
test.loc[test['shot_basics'] == -1,'shot_basics'] = 4

train.loc[train['range_of_shot'] == -1,'range_of_shot'] = 4
train.loc[train['area_of_shot'] == -1,'area_of_shot'] = 0
train.loc[train['shot_basics'] == -1,'shot_basics'] = 4



train_ip = train.drop(columns = ['is_goal','knockout_match','home/away','game_season'], axis = 1)
test = test.drop(columns = ['knockout_match','home/away','game_season'], axis = 1)
train_op = train['is_goal']



####
sns.boxplot(x = train_ip['location_y'])
plt.show()

q75,q25 = np.percentile(train_ip.location_y,[75,25])
iqr = q75 - q25

min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)


train_ip.loc[train_ip['location_y'] < min,'location_y'] = train_ip['location_y'].mean()
train_ip.loc[train_ip['location_y'] > max,'location_y'] = train_ip['location_y'].mean()
###


sns.boxplot(x = train_ip['power_of_shot'])
plt.show()

q75,q25 = np.percentile(train_ip.power_of_shot,[75,25])
iqr = q75 - q25

min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)


train_ip.loc[train_ip['power_of_shot'] < min,'power_of_shot'] = train_ip['power_of_shot'].median()
train_ip.loc[train_ip['power_of_shot'] > max,'power_of_shot'] = train_ip['power_of_shot'].median()

###
sns.boxplot(x = train_ip['distance_of_shot'])
plt.show()

q75,q25 = np.percentile(train_ip.distance_of_shot,[75,25])
iqr = q75 - q25

min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)


train_ip.loc[train_ip['distance_of_shot'] < min,'distance_of_shot'] = train_ip['distance_of_shot'].mean()
train_ip.loc[train_ip['distance_of_shot'] > max,'distance_of_shot'] = train_ip['distance_of_shot'].mean()


###

lxm = train_ip.location_x.mean()
lxs = train_ip.location_x.std()

lym = train_ip.location_y.mean()
lys = train_ip.location_y.std()


train_ip['location_x'] = (train_ip['location_x'] - lxm)/lxs
train_ip['location_y'] = (train_ip['location_y'] - lym)/lys


ltm = train_ip.lat.mean()
lts = train_ip.lat.std()

lgm = train_ip.lng.mean()
lgs = train_ip.lng.std()

train_ip['lat'] =(train_ip['lat']-ltm)/lts
train_ip['lng'] =(train_ip['lng']-lgm)/lgs


tm = train_ip.type_of_shot.mean()
ts = train_ip.type_of_shot.std()

train_ip['type_of_shot'] =(train_ip['type_of_shot']-tm)/ts








l = sns.heatmap(train_ip.corr(),annot=False, fmt = ".2f", cmap = "coolwarm")
plt.show()




from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier



validation = 0.20
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(train_ip,train_op,test_size=validation
                                                                ,random_state=7)

grab = GradientBoostingClassifier()
grab.fit(X_train, Y_train)
mgb = grab.predict(X_validation)
mae = mean_absolute_error(Y_validation,mgb)
print(1/(1+mae))

plt.bar(train_ip.columns, grab.feature_importances_)
plt.title('Feature Importances')
plt.show()


train_ip.to_csv('final.csv',index = False)
###########

