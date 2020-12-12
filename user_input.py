import pandas as pd
def main():
    df = pd.read_csv('categories.csv')
    joined = pd.read_csv('joined.csv')
    print('Please choose what type of amenity you are interested in: ')
    print('0. In general')
    print('1. Financial')
    print('2. Vehicles')
    print('3. Recreation')
    print('4. Food')
    print('5. Work')
    print('6. Pray')
    print('7. Trash')
    print('8. Repair')
    print('9. Work')
    print('10. Clock')
    print('11. Landmark')
    print('12. transport')

    value = input("")
    value = int(value)
    category = "N/A"
    if(value == 1):
        category = "money"
    elif(value == 2):
        category = "vehicles"
    elif(value == 3):
        category = "health"
    elif(value == 4):
        category = "recreation"
    elif(value == 5):
        category = "food"
    elif(value == 6):
        category = "work"
    elif(value == 7):
        category = "pray"
    elif(value == 8):
        category = "trash"
    elif(value == 9):
        category = "repair"
    elif(value == 10):
        category = "clock"
    elif(value == 11):
        category = "landmarks"
    elif(value == 12):
        category = "transports"
    df_category = df[category].dropna()
    print(df_category)
    joined = joined[joined['amenity'].isin(df_category)]
    print(joined)

if __name__ == '__main__':
    main()