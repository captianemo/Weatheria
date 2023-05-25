import streamlit as st

# Render sidebar
st.sidebar.title("Weatheria")

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://img.freepik.com/free-vector/gorgeous-clouds-background-with-blue-sky-design_1017-25501.jpg?t=st=1676590909~exp=1676591509~hmac=615fb926eeb1f74cb6256ced16dcb2baad96788cc9ddaef4e213f6095d2fc809");
            background-attachment: fixed;
            background-size: cover;
            opacity: 0.8;
        }}
        .stMetricValue {{
            font-weight: bold;
            }}
        
        
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()




def set_button_styles():
    # CSS styles for Button 1
    button1_style = """
        <style>
        .stButton button:first-child {
            background-color: #4CAF50;  /* Green color */
            border-radius: 8px;  /* Rounded edges */
            color: white;  /* Text color */
        }
        </style>
    """
    st.markdown(button1_style, unsafe_allow_html=True)

set_button_styles()

####################################333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
#333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333



cities = ['Nagpur','Bangalore', 'Mumbai', 'Jaipur', 'Hyderabad']

# Render navigation select box in the sidebar
selected_city = st.sidebar.selectbox("Select a city", cities)

import streamlit as st
from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='94eff4298ef34ce88045602f8912db70')

# Sidebar with retractable section
with st.sidebar.expander("Get Sports News"):
    # Retrieve sports news of India
    top_headlines = newsapi.get_top_headlines(
        category='sports',
        country='in'
    )

    # Display the news articles
    for article in top_headlines['articles']:
        st.markdown(f"### <u>Title:</u>&nbsp;&nbsp;&nbsp;&nbsp;{article['title']}", unsafe_allow_html=True)
        st.markdown(f"**Description:** {article['description']}", unsafe_allow_html=True)
        st.write('-' * 50)

        
######################################################
# Display content based on the selected city
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import requests
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


if selected_city == 'Nagpur':
    st.title("Weatheria")
    st.subheader("Location: Nagpur")


        # Add content for Nagpur here
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import requests
    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_absolute_error

    # Function to retrieve weather data from WeatherAPI.com
    def get_weather_data(api_key, query):
        base_url = "http://api.weatherapi.com/v1/current.json"
        complete_url = base_url + "?key=" + api_key + "&q=" + query
        response = requests.get(complete_url)
        data = response.json()
        return data


    # Function to predict weather using the trained Data of selected date
    def predict_weather(year, month, day):
        user_input = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
        predictions = model.predict(user_input)
        return predictions


    # Load the dataset
    df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\main\nagpur.csv')

    # Prepare the data
    core_weather = df[["maxtempC", "mintempC", "sunHour", "moon_illumination", "DewPointC", "FeelsLikeC", "WindGustKmph",
                    "cloudcover", "humidity", "precipMM", "pressure", "tempC", "visibility", "winddirDegree", "windspeedKmph"]].copy()
    core_weather = core_weather.fillna(method="ffill")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    X = df[['year', 'month', 'day']]
    y = df[['maxtempC', 'mintempC', 'FeelsLikeC', 'WindGustKmph',
            'humidity', 'precipMM', 'pressure', 'tempC']]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model with MultiOutputRegressor
    base_model = AdaBoostRegressor(random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)


    st.write(mae)


    # Function to display the weather information using API data
    def display_api_weather_data(api_data):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        api_humidity = api_data["current"]["humidity"]
        api_pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]
        weather_condition = api_data["current"]["condition"]["text"]

        col1, col2 ,col3= st.columns(3)  # Split the display into two columns

        col1.metric("**Temperature**", f"{temp_c} °C")
        col1.metric("**Precipitation**", f"{precip_mm} mm")
        col2.metric("**Feels Like**", f"{feelslike_c} °C")

        col2.metric("**Condition**", f"{weather_condition}")
        col2.metric("**Humidity**", f"{api_humidity}%")
        col3.metric("**Pressure**", f"{api_pressure} mb")
        col3.metric("**Wind Gust**", f"{wind_gust} kph")
        

        # st.write(f"Max Temperature from API: {max_temp}")
        # st.write(f"Min Temperature from API: {min_temp}")


    # Function to display the predicted weather information using Data of selected date
        
    def display_predicted_weather(predictions):
        temperature = round(predictions[0][7], 2)
        max_temperature = round(predictions[0][0], 2)
        min_temperature = round(predictions[0][1], 2)
        feels_like_temperature = round(predictions[0][2], 2)
        wind_gust = round(predictions[0][3], 2)
        humidity = round(predictions[0][4], 2)
        precipitation = round(predictions[0][5], 2)
        pressure = round(predictions[0][6], 2)

        # Create two columns
        col1, col2, col3,col4 = st.columns(4)

        # Display values in the first column
        col1.metric(label='Temperature', value=temperature)
        col1.metric(label='Max Temperature', value=max_temperature)
        col2.metric(label='Min Temperature', value=min_temperature)
        col2.metric(label='Feels Like Temperature', value=feels_like_temperature)

        # Display values in the second column
        col3.metric(label='Wind Gust (kmph)', value=wind_gust)
        col3.metric(label='Humidity', value=humidity)
        col4.metric(label='Precipitation (mm)', value=precipitation)
        col4.metric(label='Pressure', value=pressure)



    # Function to visualize the comparison between API data and Data of selected date predictions
    # Function to visualize the comparison between API data and Data of selected date predictions
    def visualize_comparison(api_data, predictions):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        humidity = api_data["current"]["humidity"]
        pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]

        diff_temp = abs(round(temp_c, 2) - round(predictions[0][7], 2))
        diff_precip = abs(round(precip_mm, 2) - round(predictions[0][5], 2))
        diff_feelslike = abs(round(feelslike_c, 2) - round(predictions[0][2], 2))
        diff_humidity = abs(round(humidity, 2) - round(predictions[0][4], 2))
        diff_pressure = abs(round(pressure, 2) - round(predictions[0][6], 2))
        diff_wind_gust = abs(round(wind_gust, 2) - round(predictions[0][3], 2))

        fig, ax = plt.subplots(6, figsize=(8, 18))
        fig.suptitle('Comparison of Weather Data')
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

        ax[0].bar(["Current Day's data", selected_day], [round(temp_c, 2), round(predictions[0][7], 2)])
        ax[0].set_ylim([0, 50])
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].text(0, round(temp_c, 2), f'{round(temp_c, 2)}°C')
        ax[0].text(1, round(predictions[0][7], 2), f'{round(predictions[0][7], 2)}°C')
        ax[0].text(0.4, (round(temp_c, 2) + round(predictions[0][7], 2))/2 + 1,
                f'{diff_temp:.2f}°C Difference', color='red')

        ax[1].bar(["Today's Live Weather Data", selected_day], [round(precip_mm, 2), round(predictions[0][5], 2)])
        ax[1].set_ylim([0, 50])
        ax[1].set_ylabel('Precipitation (mm)')
        ax[1].text(0, round(precip_mm, 2), f'{round(precip_mm, 2)} mm')
        ax[1].text(1, round(predictions[0][5], 2), f'{round(predictions[0][5], 2)} mm')
        ax[1].text(0.4, (round(precip_mm, 2) + round(predictions[0][5], 2))/2 + 1,
                f'{diff_precip:.2f} mm Difference', color='red')

        ax[2].bar(["Today's Live Weather Data", selected_day], [round(feelslike_c, 2), round(predictions[0][2], 2)])
        ax[2].set_ylim([0, 50])
        ax[2].set_ylabel('Feels Like Temperature')
        ax[2].text(0, round(feelslike_c, 2), f'{round(feelslike_c, 2)}°C')
        ax[2].text(1, round(predictions[0][2], 2), f'{round(predictions[0][2], 2)}°C')
        ax[2].text(0.4, (round(feelslike_c, 2) + round(predictions[0][2], 2))/2 + 1,
                f'{diff_feelslike:.2f}°C Difference', color='red')

        ax[3].bar(["Today's Live Weather Data", selected_day], [round(humidity, 2), round(predictions[0][4], 2)])
        ax[3].set_ylim([0, 100])
        ax[3].set_ylabel('Humidity (%)')
        ax[3].text(0, round(humidity, 2), f'{round(humidity, 2)}%')
        ax[3].text(1, round(predictions[0][4], 2), f'{round(predictions[0][4], 2)}%')
        ax[3].text(0.4, (round(humidity, 2) + round(predictions[0][4], 2))/2 + 1,
                f'{diff_humidity:.2f}% Difference', color='red')

        ax[4].bar(["Today's Live Weather Data", selected_day], [round(pressure, 2), round(predictions[0][6], 2)])
        ax[4].set_ylim([0, 2000])
        ax[4].set_ylabel('Pressure (mb)')
        ax[4].text(0, round(pressure, 2), f'{round(pressure, 2)} mb')
        ax[4].text(1, round(predictions[0][6], 2), f'{round(predictions[0][6], 2)} mb')
        ax[4].text(0.4, (round(pressure, 2) + round(predictions[0][6], 2))/2 + 100,
                f'{diff_pressure:.2f} mb Difference', color='red')

        ax[5].bar(["Today's Live Weather Data", selected_day], [round(wind_gust, 2), round(predictions[0][3], 2)])
        ax[5].set_ylim([0, 100])
        ax[5].set_ylabel('Wind Gust (kph)')
        ax[5].text(0, round(wind_gust, 2), f'{round(wind_gust, 2)} kph')
        ax[5].text(1, round(predictions[0][3], 2), f'{round(predictions[0][3], 2)} kph')
        ax[5].text(0.4, (round(wind_gust, 2) + round(predictions[0][3], 2))/2 + 5,
                f'{diff_wind_gust:.2f} kph Difference', color='red')

        return fig



    # Get user input for date
    year, month, day = st.date_input('Enter the date:', value=pd.Timestamp.today(
    ), min_value=pd.Timestamp('2009-01-01'), max_value=pd.Timestamp('2023-12-31')).timetuple()[:3]
    

    selected_day = f"{day}-{month}-{year}"
    col1, col2 = st.columns(2)
    
    if col1.button('Current Live weather data'):
        api_key = "23072daca1314bb8961154015232704"
        query = "Nagpur"
        api_data = get_weather_data(api_key, query)
        display_api_weather_data(api_data)


    if col2.button('Predict Weather'):
        predictions = predict_weather(year, month, day)
        display_predicted_weather(predictions)


    with st.expander("Visualise"):
        api_key = "23072daca1314bb8961154015232704"
        query = "Nagpur"
        api_data = get_weather_data(api_key, query)
        predictions = predict_weather(year, month, day)
        fig = visualize_comparison(api_data, predictions)
        st.pyplot(fig)
  
    
    

   
    
    
########################################################################################################   
elif selected_city == 'Bangalore':
    st.title("Weatheria")
    st.subheader("Location: Banglore")
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import requests
    import matplotlib.pyplot as plt
    import streamlit as st

    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_absolute_error

    # Function to retrieve weather data from WeatherAPI.com
    def get_weather_data(api_key, query):
        base_url = "http://api.weatherapi.com/v1/current.json"
        complete_url = base_url + "?key=" + api_key + "&q=" + query
        response = requests.get(complete_url)
        data = response.json()
        return data


    # Function to predict weather using the trained Data of selected date
    def predict_weather(year, month, day):
        user_input = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
        predictions = model.predict(user_input)
        return predictions


    # Load the data
    df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\main\bengaluru.csv')

    # Prepare the data
    core_weather = df[["maxtempC", "mintempC", "sunHour", "moon_illumination", "DewPointC", "FeelsLikeC", "WindGustKmph",
                    "cloudcover", "humidity", "precipMM", "pressure", "tempC", "visibility", "winddirDegree", "windspeedKmph"]].copy()
    core_weather = core_weather.fillna(method="ffill")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    X = df[['year', 'month', 'day']]
    y = df[['maxtempC', 'mintempC', 'FeelsLikeC', 'WindGustKmph',
            'humidity', 'precipMM', 'pressure', 'tempC']]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model with MultiOutputRegressor
    base_model = AdaBoostRegressor(random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)


    st.write(mae)


    # Function to display the weather information using API data
    def display_api_weather_data(api_data):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        api_humidity = api_data["current"]["humidity"]
        api_pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]
        weather_condition = api_data["current"]["condition"]["text"]

        col1, col2 ,col3= st.columns(3)  # Split the display into two columns

        col1.metric("**Temperature**", f"{temp_c} °C")
        col1.metric("**Precipitation**", f"{precip_mm} mm")
        col2.metric("**Feels Like**", f"{feelslike_c} °C")

        col2.metric("**Condition**", f"{weather_condition}")
        col2.metric("**Humidity**", f"{api_humidity}%")
        col3.metric("**Pressure**", f"{api_pressure} mb")
        col3.metric("**Wind Gust**", f"{wind_gust} kph")


    # Function to display the predicted weather information using Data of selected date
        
    def display_predicted_weather(predictions):
        temperature = round(predictions[0][7], 2)
        max_temperature = round(predictions[0][0], 2)
        min_temperature = round(predictions[0][1], 2)
        feels_like_temperature = round(predictions[0][2], 2)
        wind_gust = round(predictions[0][3], 2)
        humidity = round(predictions[0][4], 2)
        precipitation = round(predictions[0][5], 2)
        pressure = round(predictions[0][6], 2)

        # Create two columns
        col1, col2, col3,col4 = st.columns(4)

        # Display values in the first column
        col1.metric(label='Temperature', value=temperature)
        col1.metric(label='Max Temperature', value=max_temperature)
        col2.metric(label='Min Temperature', value=min_temperature)
        col2.metric(label='Feels Like Temperature', value=feels_like_temperature)

        # Display values in the second column
        col3.metric(label='Wind Gust (kmph)', value=wind_gust)
        col3.metric(label='Humidity', value=humidity)
        col4.metric(label='Precipitation (mm)', value=precipitation)
        col4.metric(label='Pressure', value=pressure)



    # Function to visualize the comparison between API data and Data of selected date predictions
    # Function to visualize the comparison between API data and Data of selected date predictions
    def visualize_comparison(api_data, predictions):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        humidity = api_data["current"]["humidity"]
        pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]

        diff_temp = abs(round(temp_c, 2) - round(predictions[0][7], 2))
        diff_precip = abs(round(precip_mm, 2) - round(predictions[0][5], 2))
        diff_feelslike = abs(round(feelslike_c, 2) - round(predictions[0][2], 2))
        diff_humidity = abs(round(humidity, 2) - round(predictions[0][4], 2))
        diff_pressure = abs(round(pressure, 2) - round(predictions[0][6], 2))
        diff_wind_gust = abs(round(wind_gust, 2) - round(predictions[0][3], 2))

        fig, ax = plt.subplots(6, figsize=(8, 18))
        fig.suptitle('Comparison of Weather Data')
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

        ax[0].bar(["Today's Live Weather Data", selected_day], [round(temp_c, 2), round(predictions[0][7], 2)])
        ax[0].set_ylim([0, 50])
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].text(0, round(temp_c, 2), f'{round(temp_c, 2)}°C')
        ax[0].text(1, round(predictions[0][7], 2), f'{round(predictions[0][7], 2)}°C')
        ax[0].text(0.4, (round(temp_c, 2) + round(predictions[0][7], 2))/2 + 1,
                f'{diff_temp:.2f}°C Difference', color='red')

        ax[1].bar(["Today's Live Weather Data", selected_day], [round(precip_mm, 2), round(predictions[0][5], 2)])
        ax[1].set_ylim([0, 50])
        ax[1].set_ylabel('Precipitation (mm)')
        ax[1].text(0, round(precip_mm, 2), f'{round(precip_mm, 2)} mm')
        ax[1].text(1, round(predictions[0][5], 2), f'{round(predictions[0][5], 2)} mm')
        ax[1].text(0.4, (round(precip_mm, 2) + round(predictions[0][5], 2))/2 + 1,
                f'{diff_precip:.2f} mm Difference', color='red')

        ax[2].bar(["Today's Live Weather Data", selected_day], [round(feelslike_c, 2), round(predictions[0][2], 2)])
        ax[2].set_ylim([0, 50])
        ax[2].set_ylabel('Feels Like Temperature')
        ax[2].text(0, round(feelslike_c, 2), f'{round(feelslike_c, 2)}°C')
        ax[2].text(1, round(predictions[0][2], 2), f'{round(predictions[0][2], 2)}°C')
        ax[2].text(0.4, (round(feelslike_c, 2) + round(predictions[0][2], 2))/2 + 1,
                f'{diff_feelslike:.2f}°C Difference', color='red')

        ax[3].bar(["Today's Live Weather Data", selected_day], [round(humidity, 2), round(predictions[0][4], 2)])
        ax[3].set_ylim([0, 100])
        ax[3].set_ylabel('Humidity (%)')
        ax[3].text(0, round(humidity, 2), f'{round(humidity, 2)}%')
        ax[3].text(1, round(predictions[0][4], 2), f'{round(predictions[0][4], 2)}%')
        ax[3].text(0.4, (round(humidity, 2) + round(predictions[0][4], 2))/2 + 1,
                f'{diff_humidity:.2f}% Difference', color='red')

        ax[4].bar(["Today's Live Weather Data", selected_day], [round(pressure, 2), round(predictions[0][6], 2)])
        ax[4].set_ylim([0, 2000])
        ax[4].set_ylabel('Pressure (mb)')
        ax[4].text(0, round(pressure, 2), f'{round(pressure, 2)} mb')
        ax[4].text(1, round(predictions[0][6], 2), f'{round(predictions[0][6], 2)} mb')
        ax[4].text(0.4, (round(pressure, 2) + round(predictions[0][6], 2))/2 + 100,
                f'{diff_pressure:.2f} mb Difference', color='red')

        ax[5].bar(["Today's Live Weather Data", selected_day], [round(wind_gust, 2), round(predictions[0][3], 2)])
        ax[5].set_ylim([0, 100])
        ax[5].set_ylabel('Wind Gust (kph)')
        ax[5].text(0, round(wind_gust, 2), f'{round(wind_gust, 2)} kph')
        ax[5].text(1, round(predictions[0][3], 2), f'{round(predictions[0][3], 2)} kph')
        ax[5].text(0.4, (round(wind_gust, 2) + round(predictions[0][3], 2))/2 + 5,
                f'{diff_wind_gust:.2f} kph Difference', color='red')

        return fig



    # Get user input for date
    year, month, day = st.date_input('Enter the date:', value=pd.Timestamp.today(
    ), min_value=pd.Timestamp('2009-01-01'), max_value=pd.Timestamp('2023-12-31')).timetuple()[:3]


    selected_day = f"{day}-{month}-{year}"
    col1, col2 = st.columns(2)
    
    if col1.button('Current Live weather data'):
        api_key = "23072daca1314bb8961154015232704"
        query = "Bengaluru"
        api_data = get_weather_data(api_key, query)
        display_api_weather_data(api_data)


    if col2.button('Predict Weather'):
        predictions = predict_weather(year, month, day)
        display_predicted_weather(predictions)


    with st.expander("Visualise"):
        api_key = "23072daca1314bb8961154015232704"
        query = "Bengaluru"
        api_data = get_weather_data(api_key, query)
        predictions = predict_weather(year, month, day)
        fig = visualize_comparison(api_data, predictions)
        st.pyplot(fig)

    
###############################################################################################    
#########################################################################################################################
#############################################################################################################################
   
        
elif selected_city == 'Mumbai':
    
    st.title("Weatheria")
    st.subheader("Location: Mumbai, Maharashtra")


 # Function to retrieve weather data from WeatherAPI.com
    def get_weather_data(api_key, query):
        base_url = "http://api.weatherapi.com/v1/current.json"
        complete_url = base_url + "?key=" + api_key + "&q=" + query
        response = requests.get(complete_url)
        data = response.json()
        return data
        
        
  

    # Function to predict weather using the trained Data of selected date
    def predict_weather(year, month, day):
        user_input = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
        predictions = model.predict(user_input)
        return predictions

    # Load the data
    df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\main\bombay.csv')

    # Prepare the data
    core_weather = df[["maxtempC", "mintempC", "sunHour", "moon_illumination", "DewPointC", "FeelsLikeC", "WindGustKmph",
                    "cloudcover", "humidity", "precipMM", "pressure", "tempC", "visibility", "winddirDegree", "windspeedKmph"]].copy()
    core_weather = core_weather.fillna(method="ffill")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    X = df[['year', 'month', 'day']]
    y = df[['maxtempC', 'mintempC', 'FeelsLikeC', 'WindGustKmph',
            'humidity', 'precipMM', 'pressure', 'tempC']]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model with MultiOutputRegressor
    base_model = AdaBoostRegressor(random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)


    st.write(mae)






    def display_api_weather_data(api_data):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        api_humidity = api_data["current"]["humidity"]
        api_pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]
        weather_condition = api_data["current"]["condition"]["text"]

        col1, col2 ,col3= st.columns(3)  # Split the display into two columns

        col1.metric("**Temperature**", f"{temp_c} °C")
        col1.metric("**Precipitation**", f"{precip_mm} mm")
        col2.metric("**Feels Like**", f"{feelslike_c} °C")

        col2.metric("**Condition**", f"{weather_condition}")
        col2.metric("**Humidity**", f"{api_humidity}%")
        col3.metric("**Pressure**", f"{api_pressure} mb")
        col3.metric("**Wind Gust**", f"{wind_gust} kph")

       
            

        # st.write(f"Max Temperature from API: {max_temp}")
        # st.write(f"Min Temperature from API: {min_temp}")


    # Function to display the predicted weather information using Data of selected date
        
    def display_predicted_weather(predictions):
        temperature = round(predictions[0][7], 2)
        max_temperature = round(predictions[0][0], 2)
        min_temperature = round(predictions[0][1], 2)
        feels_like_temperature = round(predictions[0][2], 2)
        wind_gust = round(predictions[0][3], 2)
        humidity = round(predictions[0][4], 2)
        precipitation = round(predictions[0][5], 2)
        pressure = round(predictions[0][6], 2)

        # Create two columns
        col1, col2, col3,col4 = st.columns(4)

        # Display values in the first column
        col1.metric(label='Temperature', value=temperature)
        col1.metric(label='Max Temperature', value=max_temperature)
        col2.metric(label='Min Temperature', value=min_temperature)
        col2.metric(label='Feels Like Temperature', value=feels_like_temperature)

        # Display values in the second column
        col3.metric(label='Wind Gust (kmph)', value=wind_gust)
        col3.metric(label='Humidity', value=humidity)
        col4.metric(label='Precipitation (mm)', value=precipitation)
        col4.metric(label='Pressure', value=pressure)



    # Function to visualize the comparison between API data and Data of selected date predictions
    # Function to visualize the comparison between API data and Data of selected date predictions
    def visualize_comparison(api_data, predictions):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        humidity = api_data["current"]["humidity"]
        pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]

        diff_temp = abs(round(temp_c, 2) - round(predictions[0][7], 2))
        diff_precip = abs(round(precip_mm, 2) - round(predictions[0][5], 2))
        diff_feelslike = abs(round(feelslike_c, 2) - round(predictions[0][2], 2))
        diff_humidity = abs(round(humidity, 2) - round(predictions[0][4], 2))
        diff_pressure = abs(round(pressure, 2) - round(predictions[0][6], 2))
        diff_wind_gust = abs(round(wind_gust, 2) - round(predictions[0][3], 2))

        fig, ax = plt.subplots(6, figsize=(8, 18))
        fig.suptitle('Comparison of Weather Data')
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

        ax[0].bar(["Today's Live Weather Data", selected_day], [round(temp_c, 2), round(predictions[0][7], 2)])
        ax[0].set_ylim([0, 50])
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].text(0, round(temp_c, 2), f'{round(temp_c, 2)}°C')
        ax[0].text(1, round(predictions[0][7], 2), f'{round(predictions[0][7], 2)}°C')
        ax[0].text(0.4, (round(temp_c, 2) + round(predictions[0][7], 2))/2 + 1,
                f'{diff_temp:.2f}°C Difference', color='red')

        ax[1].bar(["Today's Live Weather Data", selected_day], [round(precip_mm, 2), round(predictions[0][5], 2)])
        ax[1].set_ylim([0, 50])
        ax[1].set_ylabel('Precipitation (mm)')
        ax[1].text(0, round(precip_mm, 2), f'{round(precip_mm, 2)} mm')
        ax[1].text(1, round(predictions[0][5], 2), f'{round(predictions[0][5], 2)} mm')
        ax[1].text(0.4, (round(precip_mm, 2) + round(predictions[0][5], 2))/2 + 1,
                f'{diff_precip:.2f} mm Difference', color='red')

        ax[2].bar(["Today's Live Weather Data", selected_day], [round(feelslike_c, 2), round(predictions[0][2], 2)])
        ax[2].set_ylim([0, 50])
        ax[2].set_ylabel('Feels Like Temperature')
        ax[2].text(0, round(feelslike_c, 2), f'{round(feelslike_c, 2)}°C')
        ax[2].text(1, round(predictions[0][2], 2), f'{round(predictions[0][2], 2)}°C')
        ax[2].text(0.4, (round(feelslike_c, 2) + round(predictions[0][2], 2))/2 + 1,
                f'{diff_feelslike:.2f}°C Difference', color='red')

        ax[3].bar(["Today's Live Weather Data", selected_day], [round(humidity, 2), round(predictions[0][4], 2)])
        ax[3].set_ylim([0, 100])
        ax[3].set_ylabel('Humidity (%)')
        ax[3].text(0, round(humidity, 2), f'{round(humidity, 2)}%')
        ax[3].text(1, round(predictions[0][4], 2), f'{round(predictions[0][4], 2)}%')
        ax[3].text(0.4, (round(humidity, 2) + round(predictions[0][4], 2))/2 + 1,
                f'{diff_humidity:.2f}% Difference', color='red')

        ax[4].bar(["Today's Live Weather Data", selected_day], [round(pressure, 2), round(predictions[0][6], 2)])
        ax[4].set_ylim([0, 2000])
        ax[4].set_ylabel('Pressure (mb)')
        ax[4].text(0, round(pressure, 2), f'{round(pressure, 2)} mb')
        ax[4].text(1, round(predictions[0][6], 2), f'{round(predictions[0][6], 2)} mb')
        ax[4].text(0.4, (round(pressure, 2) + round(predictions[0][6], 2))/2 + 100,
                f'{diff_pressure:.2f} mb Difference', color='red')

        ax[5].bar(["Today's Live Weather Data", selected_day], [round(wind_gust, 2), round(predictions[0][3], 2)])
        ax[5].set_ylim([0, 100])
        ax[5].set_ylabel('Wind Gust (kph)')
        ax[5].text(0, round(wind_gust, 2), f'{round(wind_gust, 2)} kph')
        ax[5].text(1, round(predictions[0][3], 2), f'{round(predictions[0][3], 2)} kph')
        ax[5].text(0.4, (round(wind_gust, 2) + round(predictions[0][3], 2))/2 + 5,
                f'{diff_wind_gust:.2f} kph Difference', color='red')

        return fig



    # Get user input for date
    year, month, day = st.date_input('Enter the date:', value=pd.Timestamp.today(
    ), min_value=pd.Timestamp('2009-01-01'), max_value=pd.Timestamp('2023-12-31')).timetuple()[:3]
    selected_day = f"{day}-{month}-{year}"
    
    
    col1, col2 = st.columns(2)
    
    if col1.button('Current Live weather data'):
        api_key = "23072daca1314bb8961154015232704"
        query = "Wankhede Stadium, Mumbai"
        api_data = get_weather_data(api_key, query)
        display_api_weather_data(api_data)


    if col2.button('Predict Weather'):
        predictions = predict_weather(year, month, day)
        display_predicted_weather(predictions)


    with st.expander("Visualise"):
        api_key = "23072daca1314bb8961154015232704"
        query = "Wankhede Stadium, Mumbai"
        api_data = get_weather_data(api_key, query)
        predictions = predict_weather(year, month, day)
        fig = visualize_comparison(api_data, predictions)
        st.pyplot(fig)
        
        
        
        
##############################################################################################################################3###################33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
###########################################################################################################################################################################################################
#########################################################################################################################
#############################################################################################################################
   
        
elif selected_city == 'Jaipur':
    
    st.title("Weatheria")
    st.subheader("Location: Jaipur, Rajasthan")


 # Function to retrieve weather data from WeatherAPI.com
    def get_weather_data(api_key, query):
        base_url = "http://api.weatherapi.com/v1/current.json"
        complete_url = base_url + "?key=" + api_key + "&q=" + query
        response = requests.get(complete_url)
        data = response.json()
        return data
        
        
  

    # Function to predict weather using the trained Data of selected date
    def predict_weather(year, month, day):
        user_input = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
        predictions = model.predict(user_input)
        return predictions

    # Load the data
    df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\main\pune.csv')

    # Prepare the data
    core_weather = df[["maxtempC", "mintempC", "sunHour", "moon_illumination", "DewPointC", "FeelsLikeC", "WindGustKmph",
                    "cloudcover", "humidity", "precipMM", "pressure", "tempC", "visibility", "winddirDegree", "windspeedKmph"]].copy()
    core_weather = core_weather.fillna(method="ffill")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    X = df[['year', 'month', 'day']]
    y = df[['maxtempC', 'mintempC', 'FeelsLikeC', 'WindGustKmph',
            'humidity', 'precipMM', 'pressure', 'tempC']]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model with MultiOutputRegressor
    base_model = AdaBoostRegressor(random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)


    st.write(mae)






    def display_api_weather_data(api_data):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        api_humidity = api_data["current"]["humidity"]
        api_pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]
        weather_condition = api_data["current"]["condition"]["text"]

        col1, col2 ,col3= st.columns(3)  # Split the display into two columns

        col1.metric("**Temperature**", f"{temp_c} °C")
        col1.metric("**Precipitation**", f"{precip_mm} mm")
        col2.metric("**Feels Like**", f"{feelslike_c} °C")

        col2.metric("**Condition**", f"{weather_condition}")
        col2.metric("**Humidity**", f"{api_humidity}%")
        col3.metric("**Pressure**", f"{api_pressure} mb")
        col3.metric("**Wind Gust**", f"{wind_gust} kph")

       
            

        # st.write(f"Max Temperature from API: {max_temp}")
        # st.write(f"Min Temperature from API: {min_temp}")


    # Function to display the predicted weather information using Data of selected date
        
    def display_predicted_weather(predictions):
        temperature = round(predictions[0][7], 2)
        max_temperature = round(predictions[0][0], 2)
        min_temperature = round(predictions[0][1], 2)
        feels_like_temperature = round(predictions[0][2], 2)
        wind_gust = round(predictions[0][3], 2)
        humidity = round(predictions[0][4], 2)
        precipitation = round(predictions[0][5], 2)
        pressure = round(predictions[0][6], 2)

        # Create two columns
        col1, col2, col3,col4 = st.columns(4)

        # Display values in the first column
        col1.metric(label='Temperature', value=temperature)
        col1.metric(label='Max Temperature', value=max_temperature)
        col2.metric(label='Min Temperature', value=min_temperature)
        col2.metric(label='Feels Like Temperature', value=feels_like_temperature)

        # Display values in the second column
        col3.metric(label='Wind Gust (kmph)', value=wind_gust)
        col3.metric(label='Humidity', value=humidity)
        col4.metric(label='Precipitation (mm)', value=precipitation)
        col4.metric(label='Pressure', value=pressure)



    # Function to visualize the comparison between API data and Data of selected date predictions
    # Function to visualize the comparison between API data and Data of selected date predictions
    def visualize_comparison(api_data, predictions):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        humidity = api_data["current"]["humidity"]
        pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]

        diff_temp = abs(round(temp_c, 2) - round(predictions[0][7], 2))
        diff_precip = abs(round(precip_mm, 2) - round(predictions[0][5], 2))
        diff_feelslike = abs(round(feelslike_c, 2) - round(predictions[0][2], 2))
        diff_humidity = abs(round(humidity, 2) - round(predictions[0][4], 2))
        diff_pressure = abs(round(pressure, 2) - round(predictions[0][6], 2))
        diff_wind_gust = abs(round(wind_gust, 2) - round(predictions[0][3], 2))

        fig, ax = plt.subplots(6, figsize=(8, 18))
        fig.suptitle('Comparison of Weather Data')
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

        ax[0].bar(["Today's Live Weather Data", selected_day], [round(temp_c, 2), round(predictions[0][7], 2)])
        ax[0].set_ylim([0, 50])
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].text(0, round(temp_c, 2), f'{round(temp_c, 2)}°C')
        ax[0].text(1, round(predictions[0][7], 2), f'{round(predictions[0][7], 2)}°C')
        ax[0].text(0.4, (round(temp_c, 2) + round(predictions[0][7], 2))/2 + 1,
                f'{diff_temp:.2f}°C Difference', color='red')

        ax[1].bar(["Today's Live Weather Data", selected_day], [round(precip_mm, 2), round(predictions[0][5], 2)])
        ax[1].set_ylim([0, 50])
        ax[1].set_ylabel('Precipitation (mm)')
        ax[1].text(0, round(precip_mm, 2), f'{round(precip_mm, 2)} mm')
        ax[1].text(1, round(predictions[0][5], 2), f'{round(predictions[0][5], 2)} mm')
        ax[1].text(0.4, (round(precip_mm, 2) + round(predictions[0][5], 2))/2 + 1,
                f'{diff_precip:.2f} mm Difference', color='red')

        ax[2].bar(["Today's Live Weather Data", selected_day], [round(feelslike_c, 2), round(predictions[0][2], 2)])
        ax[2].set_ylim([0, 50])
        ax[2].set_ylabel('Feels Like Temperature')
        ax[2].text(0, round(feelslike_c, 2), f'{round(feelslike_c, 2)}°C')
        ax[2].text(1, round(predictions[0][2], 2), f'{round(predictions[0][2], 2)}°C')
        ax[2].text(0.4, (round(feelslike_c, 2) + round(predictions[0][2], 2))/2 + 1,
                f'{diff_feelslike:.2f}°C Difference', color='red')

        ax[3].bar(["Today's Live Weather Data", selected_day], [round(humidity, 2), round(predictions[0][4], 2)])
        ax[3].set_ylim([0, 100])
        ax[3].set_ylabel('Humidity (%)')
        ax[3].text(0, round(humidity, 2), f'{round(humidity, 2)}%')
        ax[3].text(1, round(predictions[0][4], 2), f'{round(predictions[0][4], 2)}%')
        ax[3].text(0.4, (round(humidity, 2) + round(predictions[0][4], 2))/2 + 1,
                f'{diff_humidity:.2f}% Difference', color='red')

        ax[4].bar(["Today's Live Weather Data", selected_day], [round(pressure, 2), round(predictions[0][6], 2)])
        ax[4].set_ylim([0, 2000])
        ax[4].set_ylabel('Pressure (mb)')
        ax[4].text(0, round(pressure, 2), f'{round(pressure, 2)} mb')
        ax[4].text(1, round(predictions[0][6], 2), f'{round(predictions[0][6], 2)} mb')
        ax[4].text(0.4, (round(pressure, 2) + round(predictions[0][6], 2))/2 + 100,
                f'{diff_pressure:.2f} mb Difference', color='red')

        ax[5].bar(["Today's Live Weather Data", selected_day], [round(wind_gust, 2), round(predictions[0][3], 2)])
        ax[5].set_ylim([0, 100])
        ax[5].set_ylabel('Wind Gust (kph)')
        ax[5].text(0, round(wind_gust, 2), f'{round(wind_gust, 2)} kph')
        ax[5].text(1, round(predictions[0][3], 2), f'{round(predictions[0][3], 2)} kph')
        ax[5].text(0.4, (round(wind_gust, 2) + round(predictions[0][3], 2))/2 + 5,
                f'{diff_wind_gust:.2f} kph Difference', color='red')

        return fig



    # Get user input for date
    year, month, day = st.date_input('Enter the date:', value=pd.Timestamp.today(
    ), min_value=pd.Timestamp('2009-01-01'), max_value=pd.Timestamp('2023-12-31')).timetuple()[:3]
    selected_day = f"{day}-{month}-{year}"
    
    
    col1, col2 = st.columns(2)
    
    if col1.button('Current Live weather data'):
        api_key = "23072daca1314bb8961154015232704"
        query = "Jaipur, Rajasthan"
        api_data = get_weather_data(api_key, query)
        display_api_weather_data(api_data)


    if col2.button('Predict Weather'):
        predictions = predict_weather(year, month, day)
        display_predicted_weather(predictions)


    with st.expander("Visualise"):
        api_key = "23072daca1314bb8961154015232704"
        query = "Jaipur, Rajasthan"
        api_data = get_weather_data(api_key, query)
        predictions = predict_weather(year, month, day)
        fig = visualize_comparison(api_data, predictions)
        st.pyplot(fig)
        
                      


##############################################################################################################################3###################33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
###########################################################################################################################################################################################################
#########################################################################################################################
#############################################################################################################################
   
        
elif selected_city == 'Hyderabad':
    
    st.title("Weatheria")
    st.subheader("Location: Hyderabad, Telangana")


 # Function to retrieve weather data from WeatherAPI.com
    def get_weather_data(api_key, query):
        base_url = "http://api.weatherapi.com/v1/current.json"
        complete_url = base_url + "?key=" + api_key + "&q=" + query
        response = requests.get(complete_url)
        data = response.json()
        return data
        
        
  

    # Function to predict weather using the trained Data of selected date
    def predict_weather(year, month, day):
        user_input = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
        predictions = model.predict(user_input)
        return predictions

    # Load the data
    df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\main\pune.csv')

    # Prepare the data
    core_weather = df[["maxtempC", "mintempC", "sunHour", "moon_illumination", "DewPointC", "FeelsLikeC", "WindGustKmph",
                    "cloudcover", "humidity", "precipMM", "pressure", "tempC", "visibility", "winddirDegree", "windspeedKmph"]].copy()
    core_weather = core_weather.fillna(method="ffill")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    X = df[['year', 'month', 'day']]
    y = df[['maxtempC', 'mintempC', 'FeelsLikeC', 'WindGustKmph',
            'humidity', 'precipMM', 'pressure', 'tempC']]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model with MultiOutputRegressor
    base_model = AdaBoostRegressor(random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)


    st.write(mae)






    def display_api_weather_data(api_data):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        api_humidity = api_data["current"]["humidity"]
        api_pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]
        weather_condition = api_data["current"]["condition"]["text"]

        col1, col2 ,col3= st.columns(3)  # Split the display into two columns

        col1.metric("**Temperature**", f"{temp_c} °C")
        col1.metric("**Precipitation**", f"{precip_mm} mm")
        col2.metric("**Feels Like**", f"{feelslike_c} °C")

        col2.metric("**Condition**", f"{weather_condition}")
        col2.metric("**Humidity**", f"{api_humidity}%")
        col3.metric("**Pressure**", f"{api_pressure} mb")
        col3.metric("**Wind Gust**", f"{wind_gust} kph")

       
            

        # st.write(f"Max Temperature from API: {max_temp}")
        # st.write(f"Min Temperature from API: {min_temp}")


    # Function to display the predicted weather information using Data of selected date
        
    def display_predicted_weather(predictions):
        temperature = round(predictions[0][7], 2)
        max_temperature = round(predictions[0][0], 2)
        min_temperature = round(predictions[0][1], 2)
        feels_like_temperature = round(predictions[0][2], 2)
        wind_gust = round(predictions[0][3], 2)
        humidity = round(predictions[0][4], 2)
        precipitation = round(predictions[0][5], 2)
        pressure = round(predictions[0][6], 2)

        # Create two columns
        col1, col2, col3,col4 = st.columns(4)

        # Display values in the first column
        col1.metric(label='Temperature', value=temperature)
        col1.metric(label='Max Temperature', value=max_temperature)
        col2.metric(label='Min Temperature', value=min_temperature)
        col2.metric(label='Feels Like Temperature', value=feels_like_temperature)

        # Display values in the second column
        col3.metric(label='Wind Gust (kmph)', value=wind_gust)
        col3.metric(label='Humidity', value=humidity)
        col4.metric(label='Precipitation (mm)', value=precipitation)
        col4.metric(label='Pressure', value=pressure)



    # Function to visualize the comparison between API data and Data of selected date predictions
    # Function to visualize the comparison between API data and Data of selected date predictions
    def visualize_comparison(api_data, predictions):
        temp_c = api_data["current"]["temp_c"]
        precip_mm = api_data["current"]["precip_mm"]
        feelslike_c = api_data["current"]["feelslike_c"]
        humidity = api_data["current"]["humidity"]
        pressure = api_data["current"]["pressure_mb"]
        wind_gust = api_data["current"]["wind_kph"]

        diff_temp = abs(round(temp_c, 2) - round(predictions[0][7], 2))
        diff_precip = abs(round(precip_mm, 2) - round(predictions[0][5], 2))
        diff_feelslike = abs(round(feelslike_c, 2) - round(predictions[0][2], 2))
        diff_humidity = abs(round(humidity, 2) - round(predictions[0][4], 2))
        diff_pressure = abs(round(pressure, 2) - round(predictions[0][6], 2))
        diff_wind_gust = abs(round(wind_gust, 2) - round(predictions[0][3], 2))

        fig, ax = plt.subplots(6, figsize=(8, 18))
        fig.suptitle('Comparison of Weather Data')
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

        ax[0].bar(["Today's Live Weather Data", selected_day], [round(temp_c, 2), round(predictions[0][7], 2)])
        ax[0].set_ylim([0, 50])
        ax[0].set_ylabel('Temperature (°C)')
        ax[0].text(0, round(temp_c, 2), f'{round(temp_c, 2)}°C')
        ax[0].text(1, round(predictions[0][7], 2), f'{round(predictions[0][7], 2)}°C')
        ax[0].text(0.4, (round(temp_c, 2) + round(predictions[0][7], 2))/2 + 1,
                f'{diff_temp:.2f}°C Difference', color='red')

        ax[1].bar(["Today's Live Weather Data", selected_day], [round(precip_mm, 2), round(predictions[0][5], 2)])
        ax[1].set_ylim([0, 50])
        ax[1].set_ylabel('Precipitation (mm)')
        ax[1].text(0, round(precip_mm, 2), f'{round(precip_mm, 2)} mm')
        ax[1].text(1, round(predictions[0][5], 2), f'{round(predictions[0][5], 2)} mm')
        ax[1].text(0.4, (round(precip_mm, 2) + round(predictions[0][5], 2))/2 + 1,
                f'{diff_precip:.2f} mm Difference', color='red')

        ax[2].bar(["Today's Live Weather Data", selected_day], [round(feelslike_c, 2), round(predictions[0][2], 2)])
        ax[2].set_ylim([0, 50])
        ax[2].set_ylabel('Feels Like Temperature')
        ax[2].text(0, round(feelslike_c, 2), f'{round(feelslike_c, 2)}°C')
        ax[2].text(1, round(predictions[0][2], 2), f'{round(predictions[0][2], 2)}°C')
        ax[2].text(0.4, (round(feelslike_c, 2) + round(predictions[0][2], 2))/2 + 1,
                f'{diff_feelslike:.2f}°C Difference', color='red')

        ax[3].bar(["Today's Live Weather Data", selected_day], [round(humidity, 2), round(predictions[0][4], 2)])
        ax[3].set_ylim([0, 100])
        ax[3].set_ylabel('Humidity (%)')
        ax[3].text(0, round(humidity, 2), f'{round(humidity, 2)}%')
        ax[3].text(1, round(predictions[0][4], 2), f'{round(predictions[0][4], 2)}%')
        ax[3].text(0.4, (round(humidity, 2) + round(predictions[0][4], 2))/2 + 1,
                f'{diff_humidity:.2f}% Difference', color='red')

        ax[4].bar(["Today's Live Weather Data", selected_day], [round(pressure, 2), round(predictions[0][6], 2)])
        ax[4].set_ylim([0, 2000])
        ax[4].set_ylabel('Pressure (mb)')
        ax[4].text(0, round(pressure, 2), f'{round(pressure, 2)} mb')
        ax[4].text(1, round(predictions[0][6], 2), f'{round(predictions[0][6], 2)} mb')
        ax[4].text(0.4, (round(pressure, 2) + round(predictions[0][6], 2))/2 + 100,
                f'{diff_pressure:.2f} mb Difference', color='red')

        ax[5].bar(["Today's Live Weather Data", selected_day], [round(wind_gust, 2), round(predictions[0][3], 2)])
        ax[5].set_ylim([0, 100])
        ax[5].set_ylabel('Wind Gust (kph)')
        ax[5].text(0, round(wind_gust, 2), f'{round(wind_gust, 2)} kph')
        ax[5].text(1, round(predictions[0][3], 2), f'{round(predictions[0][3], 2)} kph')
        ax[5].text(0.4, (round(wind_gust, 2) + round(predictions[0][3], 2))/2 + 5,
                f'{diff_wind_gust:.2f} kph Difference', color='red')

        return fig



    # Get user input for date
    year, month, day = st.date_input('Enter the date:', value=pd.Timestamp.today(
    ), min_value=pd.Timestamp('2009-01-01'), max_value=pd.Timestamp('2023-12-31')).timetuple()[:3]
    selected_day = f"{day}-{month}-{year}"
    col1, col2 = st.columns(2)
    
    if col1.button('Current Live weather data'):
        api_key = "23072daca1314bb8961154015232704"
        query = "Hyderabad, Telangana"
        api_data = get_weather_data(api_key, query)
        display_api_weather_data(api_data)


    if col2.button('Predict Weather'):
        predictions = predict_weather(year, month, day)
        display_predicted_weather(predictions)


    with st.expander("Visualise"):
        api_key = "23072daca1314bb8961154015232704"
        query = "Hyderabad, Telangana"
        api_data = get_weather_data(api_key, query)
        predictions = predict_weather(year, month, day)
        fig = visualize_comparison(api_data, predictions)
        st.pyplot(fig)
        
                      
