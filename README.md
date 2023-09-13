# API-challenge
# In[ ]:


# This version of code will have annotations describing the code for both WeatherPy and VacationPy in greater detail.
# The code is heavily inspired from TA Drew's speedruns.
# You should only view this in the README file.

# WeatherPy

# In[3]:

# Necessary to run one dependency.
pip install citipy


# In[4]:

# Setting up before running the code.
# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress

# Impor the OpenWeatherMap API key
from api_keys import weather_api_key

# Import citipy to determine the cities based on latitude and longitude
from citipy import citipy


# ### Generate the Cities List by Using the `citipy` Library

# In[5]:

# Filling up our data for further analysis. Output lists the # of cities.
# Empty list for holding the latitude and longitude combinations
lat_lngs = []

# Empty list for holding the cities names
cities = []

# Range of latitudes and longitudes
lat_range = (-90, 90)
lng_range = (-180, 180)

# Create a set of random lat and lng combinations
lats = np.random.uniform(lat_range[0], lat_range[1], size=1500)
lngs = np.random.uniform(lng_range[0], lng_range[1], size=1500)
lat_lngs = zip(lats, lngs)

# Identify nearest city for each lat, lng combination
for lat_lng in lat_lngs:
    city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    
    # If the city is unique, then add it to a our cities list
    if city not in cities:
        cities.append(city)

# Print the city count to confirm sufficient count
print(f"Number of cities in the list: {len(cities)}")


# ---

# ## Requirement 1: Create Plots to Showcase the Relationship Between Weather Variables and Latitude
# 
# ### Use the OpenWeatherMap API to retrieve weather data from the cities list generated in the started code

# In[6]:

# Collecting data using openweatherapi, cleaning up process to illustrate progress in user friendly manner.
# Set the API base URL
url = f"http://api.openweathermap.org/data/2.5/weather?units=metric&appid={weather_api_key}"

# Define an empty list to fetch the weather data for each city
city_data = []

# Print to logger
print("Beginning Data Retrieval     ")
print("-----------------------------")

# Create counters
record_count = 1
set_count = 1

# Loop through all the cities in our list to fetch weather data
for i, city in enumerate(cities):
        
    # Group cities in sets of 50 for logging purposes
    if (i % 50 == 0 and i >= 50):
        set_count += 1
        record_count = 0

    # Create endpoint URL with each city
    city_url = f"{url}&q={city}"
    
    # Log the url, record, and set numbers
    print("Processing Record %s of Set %s | %s" % (record_count, set_count, city))

    # Add 1 to the record count
    record_count += 1

    # Run an API request for each of the cities
    try:
        # Parse the JSON and retrieve data
        city_weather = requests.get(city_url).json()

        # Parse out latitude, longitude, max temp, humidity, cloudiness, wind speed, country, and date
        city_lat = city_weather["coord"]["lat"]
        city_lng = city_weather["coord"]["lon"]
        city_max_temp = city_weather["main"]["temp_max"]
        city_humidity = city_weather["main"]["humidity"]
        city_clouds = city_weather["clouds"]["all"]
        city_wind = city_weather["wind"]["speed"]
        city_country = city_weather["sys"]["country"]
        city_date = city_weather["dt"]

        # Append the City information into city_data list
        city_data.append({"City": city, 
                          "Lat": city_lat, 
                          "Lng": city_lng, 
                          "Max Temp": city_max_temp,
                          "Humidity": city_humidity,
                          "Cloudiness": city_clouds,
                          "Wind Speed": city_wind,
                          "Country": city_country,
                          "Date": city_date})

    # If an error is experienced, skip the city
    except:
        print("City not found. Skipping...")
        pass
              
# Indicate that Data Loading is complete 
print("-----------------------------")
print("Data Retrieval Complete      ")
print("-----------------------------")


# In[7]:

# Turning data into pandas dataframe format for easier time to turn data into scatter plot. Also saving in a csv file.
# Convert the cities weather data into a Pandas DataFrame
city_data_df = pd.DataFrame(city_data)

# Show Record Count
city_data_df.count()


# In[8]:


# Display sample data
city_data_df.head()


# In[9]:


# Export the City_Data into a csv
city_data_df.to_csv("output_data/cities.csv", index_label="City_ID")


# In[10]:


# Read saved data
city_data_df = pd.read_csv("output_data/cities.csv", index_col="City_ID")

# Display sample data
city_data_df.head()


# ### Create the Scatter Plots Requested
# 
# #### Latitude Vs. Temperature

# In[11]:
# Starting to make scatterplots for the four relationships.

date = time.strftime("%Y-%m-%d")

# Build scatter plot for latitude vs. temperature
plt.scatter(
    city_data_df["Lat"],
    city_data_df["Max Temp"],
    edgecolor="red",
    linewidths=1,
    marker="o",
    alpha=0.8,
    label="Cities"
)

# Incorporate the other graph properties
plt.title(f"City Max Latitude vs. Temperature ({date})")
plt.xlabel("Latitude")
plt.ylabel("Max Temperature (Celsius)")

# Save the figure
plt.savefig("output_data/Fig1.png")

# Show plot
plt.show()

# First impressions of plot: max temperature higher closer to the Equator (Latitude = 0)


# #### Latitude Vs. Humidity

# In[12]:


date = time.strftime("%Y-%m-%d")

# Build the scatter plots for latitude vs. humidity
plt.scatter(
    city_data_df["Lat"],
    city_data_df["Humidity"],
    edgecolor="green",
    linewidths=1,
    marker="o",
    alpha=0.8,
    label="Cities"
)

# Incorporate the other graph properties
plt.title(f"City Max Latitude vs. Humidity ({date})")
plt.xlabel("Latitude")
plt.ylabel("Humidity")

# Save the figure
plt.savefig("output_data/Fig2.png")

# Show plot
plt.show()

# First impressions of plot: Not much correlation between Latitude and Humidity, spread all over the place


# #### Latitude Vs. Cloudiness

# In[13]:


date = time.strftime("%Y-%m-%d")

# Build the scatter plots for latitude vs. cloudiness
plt.scatter(
    city_data_df["Lat"],
    city_data_df["Cloudiness"],
    edgecolor="yellow",
    linewidths=1,
    marker="o",
    alpha=0.8,
    label="Cities"
)

# Incorporate the other graph properties
plt.title(f"City Max Latitude vs. Cloudiness ({date})")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness")

# Save the figure
plt.savefig("output_data/Fig3.png")

# Show plot
plt.show()

# First impressions of plot: Not much correlation between latitude and cloudiness, scattered around the place.


# #### Latitude vs. Wind Speed Plot

# In[14]:


date = time.strftime("%Y-%m-%d")

# Build the scatter plots for latitude vs. wind speed
plt.scatter(
    city_data_df["Lat"],
    city_data_df["Wind Speed"],
    edgecolor="orange",
    linewidths=1,
    marker="o",
    alpha=0.8,
    label="Cities"
)

# Incorporate the other graph properties
plt.title(f"City Max Latitude vs. Wind Speed ({date})")
plt.xlabel("Latitude")
plt.ylabel("Wind Speed")

# Save the figure
plt.savefig("output_data/Fig4.png")

# Show plot
plt.show()

# First impressions of plot: Not much correlation between Latitude and Wind Speed


# ---
# 
# ## Requirement 2: Compute Linear Regression for Each Relationship
# 

# In[15]:

# Making function for linear regression model creation to streamline process for displaying models on scatterplot.
# Define a function to create Linear Regression plots
def plot_linear_regression(x_value, y_value, title, text_coord):
    
    # Compute linear regression model
    (slope, intercept, rvalue, pvalue, stderr) = linregress(x_value, y_value)
    regress_value = x_value * slope + intercept
    line_eq = f"y = {round(slope,2)}x + {round(intercept,2)}"
    
    plt.scatter(x_value, y_value)
    plt.plot(x_value, regress_value, "r-")
    plt.annotate(line_eq, text_coord, fontsize=15, color="red")
    plt.xlabel("Latitude")
    plt.ylabel(title)
    print(f"The r-value is {rvalue ** 2}")
    plt.show()


# In[16]:


# Create a DataFrame with the Northern Hemisphere data (Latitude >= 0)
northern_hemi_df = city_data_df[city_data_df["Lat"] >= 0]
# Display sample data
northern_hemi_df.head()


# In[17]:


# Create a DataFrame with the Southern Hemisphere data (Latitude < 0)
southern_hemi_df = city_data_df[city_data_df["Lat"] < 0]

# Display sample data
southern_hemi_df.head()


# ###  Temperature vs. Latitude Linear Regression Plot

# In[18]:


# Linear regression on Northern Hemisphere
x_value = northern_hemi_df["Lat"]
y_value = northern_hemi_df["Max Temp"]
plot_linear_regression(x_value, y_value, "Max Temp", (6, 0))

# r-value indicates moderately strong correlation.


# In[19]:


# Linear regression on Southern Hemisphere
x_value = southern_hemi_df["Lat"]
y_value = southern_hemi_df["Max Temp"]
plot_linear_regression(x_value, y_value, "Max Temp", (-25, 10))

# r-value indicates moderately strong correlation.


# **Discussion about the linear relationship:** For both the northern and southern hemisphere, there is a moderately strong linear association between Latitude and Max Temperature. 
# As the Latitude approaches zero, the Max Temperature generally increases.
# Both r-values of different hemispheres support this association (being around if not above r-squared = 0.5).

# ### Humidity vs. Latitude Linear Regression Plot

# In[20]:


# Northern Hemisphere
x_value = northern_hemi_df["Lat"]
y_value = northern_hemi_df["Humidity"]
plot_linear_regression(x_value, y_value, "Humidity", (50, 10))

# r-value indicates very little to no correlation.


# In[21]:


# Southern Hemisphere
x_value = southern_hemi_df["Lat"]
y_value = southern_hemi_df["Humidity"]
plot_linear_regression(x_value, y_value, "Humidity", (-55, 10))

# r-value indicates very little to no correlation.


# **Discussion about the linear relationship:** For both the northern and southern hemisphere, there is little to no linear
# association between Latitude and Humidity. 
# No discernable trend could be observed between Latitude and Humidity.
# Both r-values of different hemispheres support this association (very close to r-squared values of 0).

# ### Cloudiness vs. Latitude Linear Regression Plot

# In[22]:


# Northern Hemisphere
x_value = northern_hemi_df["Lat"]
y_value = northern_hemi_df["Cloudiness"]
plot_linear_regression(x_value, y_value, "Cloudiness", (45, 45))

# r-value indicates very little to no correlation.


# In[23]:


# Southern Hemisphere
x_value = southern_hemi_df["Lat"]
y_value = southern_hemi_df["Cloudiness"]
plot_linear_regression(x_value, y_value, "Cloudiness", (-50, 60))

# r-value indicates very little to no correlation.


# **Discussion about the linear relationship:** For both the northern and southern hemisphere, there is little to no linear
# association between Latitude and Cloudiness. 
# No discernable trend could be observed between Latitude and Cloudiness.
# Both r-values of different hemispheres support this association (very close to r-squared values of 0),
# though it is worth noting a linear trend is slightly more noticeable in the Southern hemisphere compared to the North.

# ### Wind Speed vs. Latitude Linear Regression Plot

# In[24]:


# Northern Hemisphere
x_value = northern_hemi_df["Lat"]
y_value = northern_hemi_df["Wind Speed"]
plot_linear_regression(x_value, y_value, "Wind Speed", (50, 10))

# r-value indicates very little to no correlation.


# In[25]:


# Southern Hemisphere
x_value = southern_hemi_df["Lat"]
y_value = southern_hemi_df["Wind Speed"]
plot_linear_regression(x_value, y_value, "Wind Speed", (-30, 20))

# r-value indicates very little to no correlation.


# **Discussion about the linear relationship:** For both the northern and southern hemisphere, there is little to no linear
# association between Latitude and Wind Speed. 
# No discernable trend could be observed between Latitude and Wind Speed.
# Both r-values of different hemispheres support this association (very close to r-squared values of 0).

# # VacationPy
# ---
# 
# ## Starter Code to Import Libraries and Load the Weather and Coordinates Data

# In[1]:


# Dependencies and Setup
import hvplot.pandas
import pandas as pd
import requests

# Import API key
from api_keys import geoapify_key

# May need to install certain packages: pyproj, cartopy, geoviews onto kernel


# In[2]:


# Load the CSV file created in Part 1 into a Pandas DataFrame
city_data_df = pd.read_csv("output_data/cities.csv")

# Display sample data
city_data_df.head()


# ---
# 
# ### Step 1: Create a map that displays a point for every city in the `city_data_df` DataFrame. The size of the point should be the humidity in each city.

# In[3]:


get_ipython().run_cell_magic('capture', '--no-display', '\n# Configure the map plot\n# YOUR CODE HERE\nmap_plot = city_data_df.hvplot.points(\n    "Lng",\n    "Lat",\n    geo=True,\n    size="Humidity",\n    scale=1,\n    color="City",\n    alpha=0.5,\n    tiles="OSM",\n    frame_width=700,\n    frame_height=500\n)\n# Display the map\nmap_plot\n\n# Map with very pretty colors for each city, a bit complicated to pull up\n')


# ### Step 2: Narrow down the `city_data_df` DataFrame to find your ideal weather condition

# In[4]:


# Narrow down cities that fit criteria and drop any results with null values
narrowed_city_df = city_data_df[
    (city_data_df["Max Temp"] < 27) & (city_data_df["Max Temp"] > 21) \
    & (city_data_df["Wind Speed"] < 4.5) \
    & (city_data_df["Cloudiness"] == 0)
                               ]

# Drop any rows with null values
narrowed_city_df = narrowed_city_df.dropna()

# Display sample data
narrowed_city_df


# ### Step 3: Create a new DataFrame called `hotel_df`.

# In[5]:


# Use the Pandas copy function to create DataFrame called hotel_df to store the city, country, coordinates, and humidity
hotel_df = narrowed_city_df[["City", "Country", "Lat", "Lng", "Humidity"]].copy()

# Add an empty column, "Hotel Name," to the DataFrame so you can store the hotel found using the Geoapify API
hotel_df["Hotel Name"] = ""

# Display sample data
hotel_df


# ### Step 4: For each city, use the Geoapify API to find the first hotel located within 10,000 metres of your coordinates.

# In[7]:


# Set parameters to search for a hotel
radius = 10000
params = {
    "categories": "accommodation.hotel",
    "apiKey": geoapify_key,
    "limit": 20
}

# Print a message to follow up the hotel search
print("Starting hotel search")

# Iterate through the hotel_df DataFrame
for index, row in hotel_df.iterrows():
    # get latitude, longitude from the DataFrame
    latitude = row["Lat"]
    longitude = row["Lng"]
    
    # Add filter and bias parameters with the current city's latitude and longitude to the params dictionary
    params["filter"] = f"circle:{longitude},{latitude},{radius}"
    params["bias"] = f"proximity:{longitude},{latitude}"
    
    # Set base URL
    base_url = "https://api.geoapify.com/v2/places"


    # Make and API request using the params dictionaty
    name_address = requests.get(base_url, params=params)
    
    # Convert the API response to JSON format
    name_address = name_address.json()
    
    # Grab the first hotel from the results and store the name in the hotel_df DataFrame
    try:
        hotel_df.loc[index, "Hotel Name"] = name_address["features"][0]["properties"]["name"]
    except (KeyError, IndexError):
        # If no hotel is found, set the hotel name as "No hotel found".
        hotel_df.loc[index, "Hotel Name"] = "No hotel found"
        
    # Log the search results
    print(f"{hotel_df.loc[index, 'City']} - nearest hotel: {hotel_df.loc[index, 'Hotel Name']}")

# Display sample data
hotel_df

# 14 hotels were found!


# ### Step 5: Add the hotel name and the country as additional information in the hover message for each city in the map.

# In[8]:


get_ipython().run_cell_magic('capture', '--no-display', '\n# Configure the map plot\nmap_plot = hotel_df.hvplot.points(\n    "Lng",\n    "Lat",\n    geo=True,\n    size="Humidity",\n    scale=1,\n    color="City",\n    alpha=0.5,\n    tiles="OSM",\n    frame_width=700,\n    frame_height=500,\n    hover_cols=["Hotel Name", "Country"]\n)\n\n# Display the map\nmap_plot\n\n# The map looks nicer with less cities.\n')
