from PIL import Image, ExifTags
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd
import os

def main():
    
    big_df = []
    for filename in os.listdir('group_withcity'):
        df = pd.read_csv('group_withcity/'+filename)
        big_df.append(df)
    concat_df = pd.concat(big_df) # has cities with slash
    
    def remove_slash(val):
        ret = val.replace('/','_')
        return ret
    concat_df['city'] = concat_df['city'].apply(remove_slash) # now cities no more slash


    cities_list = concat_df.city.unique()
    # print(cities_list)
    # cities_list_new = []
    # for item in cities_list_old:
    #     item = item.replace('/','_')
    #     cities_list_new.append(item)
    # print(cities_list_new)
    df_collection = {}
    for city in cities_list:
        df_collection[city] = pd.DataFrame()
    
    def city_to_list(x):
        df_collection[x.city] = df_collection[x.city].append(x)


    concat_df.apply(city_to_list,axis=1)
    concat_df['city'] = concat_df['city'].apply(remove_slash)
    # print(concat_df)

    
    for city in cities_list:
        df_collection[city].to_csv('cities/'+city+".csv",encoding='utf-8', index=False)

    
    # print(df_collection['Vancouver'])

if __name__ == '__main__':
    # input = sys.argv[1]
    main()