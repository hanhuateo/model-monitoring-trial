import pandas as pd
import numpy as np
import math
from datetime import timedelta

df = pd.read_csv("statistical_testing\data\property.csv")
df2 = pd.read_csv("statistical_testing\data\project.csv")
df3 = pd.read_csv("statistical_testing\data\mrt.csv")

# change contractdate to datetime format 

df["contractdate"] = pd.to_datetime(df["contractdate"], format = "%b %Y")

df = df.sort_values(by = ["project_id", "contractdate"])    

# reset index

df.reset_index(drop = True, inplace = True)

df["index"] = df.index

# remove rows with only one instance of "project_id" 

grouped = df.groupby("project_id")

mask = grouped["project_id"].transform("size") > 1

df = df.loc[mask].copy()

df.reset_index(drop = True, inplace = True)

df["index"] = df.index

# for each project_id, group the same contractdate together and calculate an average psf for each contract date using the psf column 

df["avg_psf"] = df.groupby(["project_id", "contractdate"])["psf"].transform("mean")

# do the same for area_sqft

df["avg_area_sqft"] = df.groupby(["project_id", "contractdate"])["area_sqft"].transform("mean")

# columns to drop:

# index -> identifier
# area, area_sqft -> redundant (already have avg_area_sqft)
# price, psf -> redundant (already have avg_psf)
# floorange -> landed properties have no floor range
# lease_commence_year, remaining_lease_year and remaining_lease_cat -> some projects have missing information regarding these deatails, hard to impute 

df.drop(columns = ["index", "area", "area_sqft", "price", "psf", "floorrange", "lease_commence_year", "remaining_lease_year", "remaining_lease_cat"], inplace = True)

# drop duplicates

df.drop_duplicates(subset = ["project_id", "contractdate"], inplace = True)

df.reset_index(drop = True, inplace = True)

# generate dependent variable "growth"

df["growth"] = 0

for index, row in df.iterrows():

    current_project_id = row["project_id"]
    current_contractdate = row["contractdate"]
    current_avg_psf = row["avg_psf"]

    max_date = current_contractdate + timedelta(days = 365)
    mask = (df["contractdate"] > current_contractdate) & (df["contractdate"] <= max_date) & (df["project_id"] == current_project_id)
    max_avg_psf = df[mask]["avg_psf"].max()

    if max_avg_psf != 0:

        growth = (max_avg_psf - current_avg_psf) / current_avg_psf * 100
        df.loc[index, "growth"] = growth

# drop all rows where growth = NaN

df = df[df["growth"].notna()].copy()

df.reset_index(drop = True, inplace = True)

# for the column "growth", change to categorical data where 0 = growth < 8.0 and 1 = growth >= 8.0

df["growth"] = df["growth"].apply(lambda x: 0 if x < 8.0 else 1)

df["growth"].value_counts()

# case-control study

def case_control(df):
    
    df_1 = df[df["growth"] == 1].copy()
    df_0 = df[df["growth"] == 0].copy()

    merged_df = pd.DataFrame()

    for index, row in df_1.iterrows():

        current_project_id = row["project_id"]
        current_avg_area_sqft = row["avg_area_sqft"]

        mask = df_0["project_id"] == current_project_id
        df_0_temp = df_0.loc[mask, :].copy()

        df_0_temp["percent_diff"] = abs(df_0_temp["avg_area_sqft"] - current_avg_area_sqft) / current_avg_area_sqft * 100

        df_0_temp = df_0_temp.loc[df_0_temp["percent_diff"] <= 5].copy()

        if not df_0_temp.empty:
            
            merged_df = pd.concat([merged_df, pd.DataFrame([row]), df_0_temp.iloc[:1]], ignore_index = True, sort = False)
            df_0 = df_0.drop(df_0_temp.index[0])

    return merged_df

case_control_df = case_control(df)

# merge case_control_df with df2 by project_id

case_control_df_combined = pd.merge(case_control_df, df2, on = "project_id")

# columns to drop:

# project_id, contractdate -> treat each row as individual property
# project & street -> string descriptions with high cardinality -> redundant
# avg_psf -> do not know the information beforehand
# missing_xy -> redundant column 

case_control_df_combined.drop(columns = ["project_id", "contractdate", "project", "street", "avg_psf", "percent_diff", "missing_xy"], inplace = True)

# feature engineering
# to get shortest distance to an mrt station and the station line

def function1(df, df2):

    for index, row in df.iterrows():

        property_lat = row["latitude"]
        property_long = row["longitude"]

        min_distance = math.inf
        station = None
        mrt_line = None

        for index2, row2 in df2.iterrows():

            mrt_lat = row2["latitude"]
            mrt_long = row2["longitude"]

            # Haversine formula

            lat_diff = math.radians(mrt_lat - property_lat)
            long_diff = math.radians(mrt_long - property_long)
            a = math.sin(lat_diff/2)**2 + math.cos(math.radians(property_lat)) * math.cos(math.radians(mrt_lat)) * math.sin(long_diff/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = 6371 * c  

            if distance < min_distance:

                min_distance = distance
                station = index2
                mrt_line = row2["line"]

        df.loc[index, "shortest_distance_to_mrt"] = min_distance
        df.loc[index, "closest_mrt_line"] = mrt_line

    return df

df_updated1 = function1(case_control_df_combined, df3)

def function2(df, df2, distance_threshold):

    count_list = []

    for index, row in df.iterrows():

        property_lat = row["latitude"]
        property_long = row["longitude"]
        count = 0

        for index2, row2 in df2.iterrows():

            mrt_lat = row2["latitude"]
            mrt_long = row2["longitude"]

            # Haversine formula

            lat_diff = math.radians(mrt_lat - property_lat)
            long_diff = math.radians(mrt_long - property_long)
            a = math.sin(lat_diff/2)**2 + math.cos(math.radians(property_lat)) * math.cos(math.radians(mrt_lat)) * math.sin(long_diff/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = 6371 * c  

            if distance <= distance_threshold:

                count += 1

        count_list.append(count)

    df["mrt_count_within_500m"] = count_list

    return df

df_updated2 = function2(df_updated1, df3, 0.5)  

# drop columns x, y, latitude, longitude

df_updated2.drop(columns = ["x", "y", "latitude", "longitude"], inplace = True)

# factorize categorical columns

# categorical columns - typeofsale, propertytype, typeofarea, growth, district, tenure, marketsegment, zone, cloest_mrt_line

def factor(df):

    nominal_features = df.drop(columns = ["growth"]).select_dtypes(include = ["object"]).columns.tolist()
    nominal_features.extend(["district"])

    df[nominal_features] = df[nominal_features].astype("category")

    return df

df_factored = factor(df_updated2)

df_factored["growth"] = df_factored["growth"].astype("category")

# export to csv

df_factored.to_csv("statistical_testing\case_control.csv", index = False)