import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import autokeras as ak
from keras_tuner import HyperModel
from keras_tuner import RandomSearch
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

file = 'cancer_reg.csv'

df = pd.read_csv(file)

dataInfo = df.info()
missingValues = df.isnull().sum()
print(df.describe())
print(dataInfo)
print(missingValues[missingValues > 0])

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

# Plotting key components
key_features = ['avganncount', 'avgdeathsperyear', 'incidencerate', 'medincome', 'povertypercent']
plt.figure(figsize=(15, 10))

for i, feature in enumerate(key_features, 1):
    plt.subplot(3, 2, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

numDf = df.select_dtypes(include=['float64', 'int64'])
numDf.hist(figsize=(15,12))
plt.tight_layout()
plt.show()


duplicates = df.duplicated().sum()
print(duplicates)   # there are none so we are good from here

# Trying out feature engineering
# df['cancer_impact_index'] = (df['avganncount'] + df['avgdeathsperyear'] + df['incidencerate']) / 3
#
# df = df.drop(columns=['avganncount', 'avgdeathsperyear', 'incidencerate'])


# Dropping the column with a significant number of missing values

data_cleaned = df.drop(columns=['pctsomecol18_24', 'binnedinc', 'geography'])

# Imputing
data_cleaned['pctemployed16_over'].fillna(data_cleaned['pctemployed16_over'].mean(), inplace=True)
data_cleaned['pctprivatecoveragealone'].fillna(data_cleaned['pctprivatecoveragealone'].mean(), inplace=True)

correlation_matrix = data_cleaned.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

correlation_with_target = correlation_matrix['target_deathrate'].sort_values(ascending=False)
# selected_features = correlation_with_target[abs(correlation_with_target) > 0.35].index.drop('target_deathrate')
selected_features = ['incidencerate', 'povertypercent', 'pcths25_over', 'pctbachdeg25_over', 'pctpubliccoveragealone',
                     'pcths18_24', 'pctunemployed16_over', 'pctpubliccoverage']

X = data_cleaned[selected_features]
# X = data_cleaned[selected_features]
y = data_cleaned['target_deathrate']


scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
# X_normalized = scaler.fit_transform(data_cleaned[selected_features])

Q1 = np.percentile(X_normalized, 25, axis=0)
Q3 = np.percentile(X_normalized, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 4.5 * IQR
upper_bound = Q3 + 4.5 * IQR
non_outlier_indices = ((X_normalized >= lower_bound) & (X_normalized <= upper_bound)).all(axis=1)
X_filtered = X_normalized[non_outlier_indices]
y_filtered = data_cleaned['target_deathrate'].values[non_outlier_indices]


X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

n_folds = 10
cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring='neg_root_mean_squared_error')
cv_rmse_scores = -cv_scores
cv_rmse_mean = np.mean(cv_rmse_scores)

meanR = df['target_deathrate'].mean()
varR = df['target_deathrate'].var()
mean1 = y_pred.mean()
var1 = y_pred.var()

print(meanR)
print(varR)
print(mean1)
print(var1)
print(cv_rmse_mean, r_squared)
print(rmse)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=400, batch_size=50, validation_split=0.2)
y_pred_nn = model.predict(X_test)

r_squared = r2_score(y_test, y_pred_nn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))

meanR = df['target_deathrate'].mean()
varR = df['target_deathrate'].var()
mean1 = y_pred_nn.mean()
var1 = y_pred_nn.var()

print(meanR)
print(varR)
print(mean1)
print(var1)
print(r_squared)
print(rmse)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_nn, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(y_pred_nn)), y_pred_nn, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

class NeuralNetworkHyperModel(HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        model = Sequential()

        for i in range(hp.Int('num_layers', 2, 6)):
            if i == 0:
                model.add(Dense(
                    units=hp.Int(f'units_{i}', min_value=200, max_value=2000, step=100),
                    activation=hp.Choice(f'activation_{i}', values=['relu']),
                    input_dim=self.input_dim
                ))
            else:
                model.add(Dense(
                    units=hp.Int(f'units_{i}', min_value=50, max_value=2000, step=10),
                    activation=hp.Choice(f'activation_{i}', values=['relu', 'sigmoid'])
                ))

        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model


hypermodel = NeuralNetworkHyperModel(input_dim=X_train.shape[1])

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=25,
    executions_per_trial=4,
    directory='my_dir',
    project_name='neural_network_tuning'
)
# tuner.search(X_train, y_train, epochs=200, validation_split=0.2, batch_size=30)
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
# loss, accuracy = best_model.evaluate(X_test, y_test)
best_model = hypermodel.build(best_hp)
best_model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=25)
loss = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
predictions = best_model.predict(X_test)

print("Best Hyperparameters:")
for hp in best_hp.values:
    print(f"{hp}: {best_hp.get(hp)}")

r_squared = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

meanR = df['target_deathrate'].mean()
varR = df['target_deathrate'].var()
mean1 = predictions.mean()
var1 = predictions.var()

print(meanR)
print(varR)
print(mean1)
print(var1)
print(r_squared)
print(rmse)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(predictions, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(predictions)), predictions, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()


correlations = correlation_matrix['target_deathrate'].sort_values(ascending=False)
top_correlated_columns = correlations.index[1:5]

for column in top_correlated_columns:
    df.plot(kind='scatter', x=column, y='target_deathrate', figsize=(10, 5))
    plt.title(f'Scatter Plot of target_deathrate vs {column}')
    plt.xlabel(column)
    plt.ylabel('Target Death Rate')
    plt.show()


X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_filtered, y_filtered, test_size=0.3, random_state=42
)

nn_model = MLPRegressor(
    hidden_layer_sizes=(50,),
    activation='relu',
    solver='adam',
    max_iter=5000,
    random_state=42
)

nn_model.fit(X_train_nn, y_train_nn)
y_pred_nn = nn_model.predict(X_test_nn)
r_squared = r2_score(y_test, y_pred_nn)
rmse = np.sqrt(mean_squared_error(y_test_nn, y_pred_nn))
meanR = df['target_deathrate'].mean()
varR = df['target_deathrate'].var()
mean1 = y_pred_nn.mean()
var1 = y_pred_nn.var()

print(meanR)
print(varR)
print(mean1)
print(var1)
print(r_squared)
print(rmse)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_nn, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(y_test_nn)), y_test_nn, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(y_pred_nn)), y_pred_nn, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

r_squared_nn = r2_score(y_test_nn, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test_nn, y_pred_nn))

cv_scores_nn = cross_val_score(nn_model, X, data_cleaned['target_deathrate'], cv=10, scoring='neg_root_mean_squared_error')

cv_rmse_scores_nn = -cv_scores_nn

cv_rmse_mean_nn = np.mean(cv_rmse_scores_nn)

mean = y_pred_nn.mean()
print(r_squared_nn, rmse_nn, cv_rmse_scores_nn, cv_rmse_mean_nn, mean)

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_nn, kde=True, color='red')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 8))

plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()


# GridSearch Tree
param_grid = {
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': (2, 3, 9, 10, 11)
}
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=23)
                           , param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_dt_regressor = grid_search.best_estimator_
y_pred_best_dt = best_dt_regressor.predict(X_test)


r_squared_best_dt = r2_score(y_test, y_pred_best_dt)
rmse_best_dt = np.sqrt(mean_squared_error(y_test, y_pred_best_dt))

r_squared = r2_score(y_test, y_pred_best_dt)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_best_dt))

meanR = df['target_deathrate'].mean()
varR = df['target_deathrate'].var()
mean1 = y_pred_best_dt.mean()
var1 = y_pred_best_dt.var()

print(meanR)
print(varR)
print(mean1)
print(var1)
print(r_squared)
print(rmse)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_best_dt, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(y_pred_best_dt)), y_pred_best_dt, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

print('Best Parameters:', best_params)
print('R-squared:', r_squared_best_dt)
print('Root Mean Squared Error:', rmse_best_dt)

from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_regressor.fit(X_train, y_train)

y_pred_gb = gb_regressor.predict(X_test)

r_squared_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print('R-squared:', r_squared_gb)
print('Root Mean Squared Error:', rmse_gb)



# parameters
param_dist = {
    'n_estimators': (349, 351),
    'max_depth': [None] + list(range(8, 9)),
    'min_samples_split': sp_randint(9, 10),
    'min_samples_leaf': sp_randint(1, 2),
    'bootstrap': [True]
}

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

n_iter_search = 3
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=n_iter_search, cv=2, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_rf_regressor = random_search.best_estimator_

y_pred_best_rf = best_rf_regressor.predict(X_test)


r_squared = r2_score(y_test, y_pred_best_rf)
rmse = np.sqrt(mean_squared_error(y_test_nn, y_pred_best_rf))

meanR = df['target_deathrate'].mean()
varR = df['target_deathrate'].var()
mean1 = y_pred_best_rf.mean()
var1 = y_pred_best_rf.var()

print(meanR)
print(varR)
print(mean1)
print(var1)
print(r_squared)
print(rmse)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_best_rf, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(y_pred_best_rf)), y_pred_best_rf, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()
r_squared_best_rf = r2_score(y_test, y_pred_best_rf)
rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))

print('Best Parameters:', best_params)
print('R-squared:', r_squared_best_rf)
print('Root Mean Squared Error:', rmse_best_rf)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Building a pipeline with PCA and RandomForestRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train_pca, y_train_pca)

y_pred_pca = pipeline.predict(X_test_pca)

r_squared_pca = r2_score(y_test_pca, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))

print('R-squared:', r_squared_pca)
print('Root Mean Squared Error:', rmse_pca)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_pca, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(y_test_pca)), y_test_pca, color='blue', alpha=0.5, label='Actual')

plt.scatter(range(len(y_pred_pca)), y_pred_pca, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()


# Auto find the best network
reg = ak.StructuredDataRegressor(max_trials=15, loss='mean_squared_error', tuner='greedy',  objective='val_loss', overwrite=True)  # Set the number of architectures to try
reg.fit(X_train_nn, y_train_nn, epochs=30)
loss = reg.evaluate(X_test_nn, y_test_nn)
y_pred_ak = reg.predict(X_test_nn)
r_squared_ak = r2_score(y_test_nn, y_pred_ak)
rmse_ak = np.sqrt(mean_squared_error(y_test_nn, y_pred_ak))
r_squared = r2_score(y_test, y_pred_ak)
rmse = np.sqrt(mean_squared_error(y_test_nn, y_pred_ak))
meanR = df['target_deathrate'].mean()
varR = df['target_deathrate'].var()
mean1 = y_pred_ak.mean()
var1 = y_pred_ak.var()
print(meanR)
print(varR)
print(mean1)
print(var1)
print(r_squared)
print(rmse)

plt.figure(figsize=(10, 6))
sns.histplot(df['target_deathrate'], kde=True)
plt.title('Distribution of Target Death Rate')
plt.xlabel('Target Death Rate')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_ak, kde=True, color='orange')
plt.title('Distribution of Predicted Target Death Rate')
plt.xlabel('Predicted Target Death Rate')
plt.ylabel('Frequency')
plt.show()
plt.scatter(range(len(y_test_nn)), y_test_nn, color='blue', alpha=0.5, label='Actual')
plt.scatter(range(len(y_pred_ak)), y_pred_ak, color='red', alpha=0.5, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()
