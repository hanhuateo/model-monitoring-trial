import pandas as pd
import numpy  as np

# create dataframe with the following types of columns: 1. continuous normal, 2.continuous non-normal, 3. categorical with 2 levels, 4. categorical with > 2 levels, 5. boolean

np.random.seed(14)

# continuous data
normal_data1 = np.random.normal(loc = 0, scale = 1, size = 10000)
normal_data2 = np.random.normal(loc = 0, scale = 1, size = 10000)
normal_data3 = np.random.normal(loc = 10, scale = 10, size = 10000)
non_normal_data = np.random.uniform(low = 0, high = 1, size = 10000)

# categorical data
categorical_2_levels1 = np.random.choice(a = ["Male", "Female"], size = 10000)
dependent_categorical_3_levels = np.where(categorical_2_levels1 == "Male", np.random.choice(["A", "B", "C"], size = 10000, p = [0.6, 0.2, 0.2]), np.random.choice(["A", "B", "C"], size = 10000, p = [0.3, 0.5, 0.2]))
categorical_2_levels2 = np.random.choice(a = ["Yes", "No"], size = 10000)
categorical_multi_levels = np.random.choice(a = ["Red", "Green", "Blue", "Orange"], size = 10000)

category_means = {"Red":10, "Green":20, "Blue":30, "Orange":40}
diff_mean_normal_samples = np.zeros_like(categorical_multi_levels, dtype = np.float64)

for category in category_means:
    mask = (categorical_multi_levels == category)
    size = np.sum(mask)
    mean = category_means[category]
    diff_mean_normal_samples[mask] = np.random.normal(loc=mean, scale=1, size=size)

categorical_fishers = np.random.choice(a = ["A", "B"], size = 10000, p = [0.0003, 0.9997])
categorical_nofishers = np.random.choice(a = ["A", "B", "C"], size = 10000, p = [0.0005, 0.9990, 0.0005])

# boolean data
boolean_data = np.random.choice(a = [True, False], size = 10000)

data = {
    "normal1": normal_data1,
    "normal2": normal_data2,
    "normal3": normal_data3,
    "non-normal": non_normal_data,
    "categorical_2_levels1": categorical_2_levels1,
    "dependent_categorical_3_levels": dependent_categorical_3_levels,
    "categorical_2_levels2": categorical_2_levels2,
    "categorical_multi_levels": categorical_multi_levels,
    "diff_mean_normal_samples": diff_mean_normal_samples, 
    "categorical_fishers": categorical_fishers,
    "categorical_nofishers": categorical_nofishers,
    "boolean": boolean_data
}

df = pd.DataFrame(data)

df.info()

df.to_csv("synthetic_data.csv", index = False)