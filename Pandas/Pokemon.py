import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

filePath = "pokemon_data.xlsx"

data = pd.read_excel(filePath, header=0)

# get all the columns as list
columns = data.columns.tolist()

# get values of a column
names = data["Name"]

# get values of a column and range of rows
first_ten_names = data["Name"][0:10]

# get values of multiple column
general_info = data[["Name", "HP", "Attack", "Defense", "Speed"]]

# get values of multiple column and range of rows
first_ten_info = data[["Name", "HP", "Attack", "Defense", "Speed"]][0:10]

# get all info of a row
charizard = data.iloc[6]

# get all info of range of rows
charizard_family = data.iloc[4:7]

# get cell value [row, column]
charizard_attack = data.iloc[6, 4]

# get range of rows and range of columns
charizard_basic_info = data.iloc[4:7, 0:6]

# add a column
charizard_basic_info["HP"] = charizard_basic_info["HP"] * 2

# get rows satisfying specific conditions
fire_types = data.loc[(data["Type 1"] == "Fire") & (data["Type 2"] == "Flying")]

# get specific rows satisfying specific conditions
fire_types_info = data.loc[(data["Type 1"] == "Fire") & (data["Type 2"] == "Flying"), ["Name", "HP", "Attack", "Defense"]]
print(fire_types_info)