from PIL import Image, ExifTags
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd

# from pyspark.sql import SparkSession, functions, types, Row
# spark = SparkSession.builder.appName('OSM point of interest extracter').getOrCreate()
# assert spark.version >= '2.4' # make sure we have Spark 2.4+
# spark.sparkContext.setLogLevel('WARN')

def main(input):
    entries = pd.read_csv(input)
    grouped = entries.sort_values('amenity',ascending = False)
    # print(grouped['tags'].iloc[6839].find('wiki'))
    # with_wiki = grouped[grouped['tags'] != 1]
    # print(with_wiki)

    # CREATE THE SEPARATE FILES
    amenities_list = grouped.amenity.unique()
    df_collection = {} # has all the dataframes
    for amenity in amenities_list:
        df_collection[amenity] = pd.DataFrame()
    
    def row_to_collection(x):
        df_collection[x.amenity] = df_collection[x.amenity].append(x)
    
    grouped.apply(row_to_collection,axis=1)

    for amenity in amenities_list:
        df_collection[amenity].to_csv('new/'+amenity+".csv",encoding='utf-8', index=False)
    
    # print(type(df_collection['workshop']['tags'].iloc[1]))

if __name__ == '__main__':
    input = sys.argv[1]
    main(input)