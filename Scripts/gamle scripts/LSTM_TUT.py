import pandas as pd
import tensorflow as tf

df = pd.read_csv('data/bike_data/hour.csv', index_col='instant')

# Extracting only a subset of features
def select_columns(df):
    cols_to_keep = [
        'cnt',
        'temp',
        'hum',
        'windspeed',
        'yr',
        'mnth',
        'hr',
        'holiday',
        'weekday',
        'workingday'
    ]
    df_subset = df[cols_to_keep]
    return df_subset


# Some of the integer features need to be onehot encoded;
# but not all of them
def onehot_encode_integers(df, excluded_cols):
    df = df.copy()

    int_cols = [col for col in df.select_dtypes(include=['int'])
                if col not in excluded_cols]

    df.loc[:, int_cols] = df.loc[:, int_cols].astype('str')

    df_encoded = pd.get_dummies(df)
    return df_encoded

# cnt will be target of regression, but also a feature:
# it needs to be normalized. This is not the correct way to do it,
# as it leads to information leakage from test to training set.
def normalize_cnt(df):
    df = df.copy()
    df['cnt'] = df['cnt'] / df['cnt'].max()
    return df

# I <3 pandas pipes
dataset = (df
           .pipe(select_columns)
           .pipe(onehot_encode_integers,
                 excluded_cols=['cnt'])
           .pipe(normalize_cnt)
           )
def create_dataset(df, n_deterministic_features,
                   window_size, forecast_size,
                   batch_size):
    # Feel free to play with shuffle buffer size
    shuffle_buffer_size = len(df)
    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)

    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    # Shuffling data (seed=Answer to the Ultimate Question of Life, the Universe, and Everything)
    data = data.shuffle(shuffle_buffer_size, seed=42)

    # Extracting past features + deterministic future + labels
    data = data.map(lambda k: ((k[:-forecast_size],
                                k[-forecast_size:, -n_deterministic_features:]),
                               k[-forecast_size:, 0]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#%%
# Times at which to split train/validation and validation/test
val_time = 10000
test_time = 14000

# How much data from the past should we need for a forecast?
window_len = 24 * 7 * 3  # Three weeks
# How far ahead do we want to generate forecasts?
forecast_len = 24 * 5  # Five days

# Auxiliary constants
n_total_features = len(dataset.columns)
n_aleatoric_features = len(['cnt', 'temp', 'hum', 'windspeed'])
n_deterministic_features = n_total_features - n_aleatoric_features

# Splitting dataset into train/val/test
training_data = dataset.iloc[:val_time]
validation_data = dataset.iloc[val_time:test_time]
test_data = dataset.iloc[test_time:]


# Now we get training, validation, and test as tf.data.Dataset objects

batch_size = 32

training_windowed = create_dataset(training_data,
                                   n_deterministic_features,
                                   window_len,
                                   forecast_len,
                                   batch_size)

validation_windowed = create_dataset(validation_data,
                                     n_deterministic_features,
                                     window_len,
                                     forecast_len,
                                     batch_size)

test_windowed = create_dataset(test_data,
                               n_deterministic_features,
                               window_len,
                               forecast_len,
                               batch_size=1)
