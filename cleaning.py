import pandas as pd

# Read the CSV file
data = pd.read_csv("data/rumahcom.csv")

# Clean lokasi
## Split the address into parts using comma as the delimiter
data["lokasi"] = data["lokasi"].str.split(",")
## Keep the last two parts of the address and join them back into a single string
data["lokasi"] = data["lokasi"].apply(lambda x: ", ".join(x[-2:]))
## Delete behind the comma from lokasi
data["lokasi"] = data["lokasi"].str.split(",").str[0]
## Remove whitespace from the beginning and end of the string of lokasi
data["lokasi"] = data["lokasi"].str.strip()

# Remove after whitespace from luas_bangunan
data["luas_bangunan"] = data["luas_bangunan"].str.split(" ").str[0]

# Remove after whitespace from luas_tanah
data["luas_tanah"] = data["luas_tanah"].str.split(" ").str[0]

# Remove , from luas_tanah
data["luas_tanah"] = data["luas_tanah"].str.replace(",", "")

# Remove after whitespace from kamar
data["kamar"] = data["kamar"].str.split(" ").str[0]

# Remove after whitespace from kamar_mandi
data["kamar_mandi"] = data["kamar_mandi"].str.split(" ").str[0]

# Remove after whitespace from listrik
data["listrik"] = data["listrik"].str.split(" ").str[0]

# Clean harga
for i, value in enumerate(data["harga"]):
    if isinstance(value, str):
        try:
            data.loc[i, "harga"] = float(value.replace("Rp", "").replace(" ", "").replace("M", "").replace(",", ".")) * 1000000000
        except ValueError:
            pass

for i, value in enumerate(data["harga"]):
    if isinstance(value, str):
        try:
            data.loc[i, "harga"] = float(value.replace("Rp", "").replace(" ", "").replace("jt", "").replace(",", ".")) * 1000000
        except ValueError:
            pass

# Delete row if harga is contain rb
data = data[~data["harga"].astype(str).str.contains("rb")]


# Delete row if is null
data = data[data["harga"].notnull()]
data = data[data["lokasi"].notnull()]
data = data[data["luas_bangunan"].notnull()]
data = data[data["luas_tanah"].notnull()]
data = data[data["kamar"].notnull()]
data = data[data["kamar_mandi"].notnull()]
data = data[data["listrik"].notnull()]
data = data[data["interior"].notnull()]
data = data[data["parkir"].notnull()]
data = data[data["sertifikat"].notnull()]

# Convert data to integer
data["harga"] = data["harga"].astype(int)
data["kamar_mandi"] = data["kamar_mandi"].astype(int)
data["kamar"] = data["kamar"].astype(int)
data["luas_tanah"] = data["luas_tanah"].astype(int)
data["luas_bangunan"] = data["luas_bangunan"].astype(int)
data["parkir"] = data["parkir"].astype(int)
data["listrik"] = data["listrik"].astype(int)


# Delete column web-scraper-order, web-scraper-start-url, title, title-href, nama
data = data.drop(["web-scraper-order", "web-scraper-start-url", "title", "title-href", "nama"], axis=1)

# Save the updated data to a new CSV file
data.to_csv("data/updated_file.csv", index=False)
updated_data = pd.read_csv("data/updated_file.csv")
print(updated_data.head())