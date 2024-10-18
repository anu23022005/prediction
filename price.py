import streamlit as st
import google.generativeai as genai
import os
import json
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random

# Set the API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC7DS7QJYaCtv2LdxgObkqY9f9QNuf5Yls"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def simulate_historical_data(current_price, months=12):
    """Simulate historical price data with some volatility."""
    historical_prices = [current_price]
    for _ in range(months - 1):
        change = random.uniform(-0.03, 0.03)  # -3% to 3% monthly change
        new_price = historical_prices[-1] * (1 + change)
        historical_prices.append(round(new_price, 2))
    return historical_prices[::-1]  # Reverse to have most recent last

def extract_predictions(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if 'predictions' in data:
                return data['predictions']
        except json.JSONDecodeError:
            pass
    
    pattern = r'month[:\s]+(\d+).*?price[:\s]+(-?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    if matches:
        return [{'month': int(month), 'price': float(price)} for month, price in matches]
    
    return None

def predict_house_price(house_details, historical_prices, months):
    prompt = f"""
    Based on the following house details and historical price data:
    {house_details}
    
    Historical prices (most recent 12 months):
    {', '.join([f'Rs{price:,.2f}' for price in historical_prices])}
    
    Predict the house price for the next {months} months (one prediction per month).
    Consider market trends, historical data, seasonal variations, economic factors, and property features.
    Return only a JSON string with format:
    {{"predictions": [
        {{"month": 1, "price": 100000.00}},
        {{"month": 2, "price": 98500.00}},
        ...
    ]}}
    """
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    
    predictions = extract_predictions(response.text)
    if not predictions:
        st.error("Failed to parse the AI response. Please try again.")
        return None
    
    return predictions

def create_line_chart(historical_prices, predictions):
    historical_months = range(-len(historical_prices) + 1, 1)
    predicted_months = [p['month'] for p in predictions]
    
    plt.figure(figsize=(12, 6))
    plt.plot(historical_months, historical_prices, marker='o', label='Historical')
    plt.plot(predicted_months, [p['price'] for p in predictions], marker='o', label='Predicted')
    plt.axvline(x=0, color='r', linestyle='--', label='Current')
    plt.title('Historical and Predicted House Prices')
    plt.xlabel('Months (Negative = Past, Positive = Future)')
    plt.ylabel('Price (Rs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs{x:,.0f}'))
    
    return plt

def main():
    st.title("House Price Prediction")

    # Input fields
    location = st.text_input("Location (City, State)")
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=2)
    square_feet = st.number_input("Square Feet", min_value=100, value=1500)
    year_built = st.number_input("Year Built", min_value=1800, max_value=datetime.now().year, value=2000)
    lot_size = st.number_input("Lot Size (sqft)", min_value=100, value=5000)
    current_price = st.number_input("Current Price (Rs)", min_value=1000, value=300000)
    months = st.number_input("Number of months to predict", min_value=1, max_value=120, value=12)

    if st.button("Predict Future Prices"):
        historical_prices = simulate_historical_data(current_price)
        
        house_details = f"""
        Location: {location}
        Bedrooms: {bedrooms}
        Bathrooms: {bathrooms}
        Square Feet: {square_feet}
        Year Built: {year_built}
        Lot Size: {lot_size} sqft
        Current Price: Rs{current_price:,}
        """
        
        with st.spinner("Predicting prices..."):
            predictions = predict_house_price(house_details, historical_prices, months)
        
        if predictions:
            st.subheader("Price Predictions:")
            for pred in predictions:
                future_date = datetime.now() + timedelta(days=30*pred['month'])
                price_change = pred['price'] - current_price
                st.markdown(f"Month {pred['month']} ({future_date.strftime('%B %Y')}): Rs{pred['price']:,.2f} "
                          f"Rs{price_change:+,.2f}")
            
            chart = create_line_chart(historical_prices, predictions)
            st.pyplot(chart)

            # Analysis of prediction trends
            increases = sum(1 for i in range(len(predictions)-1) if predictions[i+1]['price'] > predictions[i]['price'])
            decreases = sum(1 for i in range(len(predictions)-1) if predictions[i+1]['price'] < predictions[i]['price'])
            st.write(f"Prediction analysis: {increases} increases, {decreases} decreases")

    # Sidebar instructions
    st.sidebar.title("How to Use")
    st.sidebar.write("""
    1. Enter the details of the house, including its location.
    2. Specify the number of months for prediction.
    3. Click "Predict Future Prices".
    4. View the price predictions, chart, and trend analysis.
    """)

if __name__ == "__main__":
    main()
