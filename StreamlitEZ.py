import streamlit as st
import pandas as pd
import numpy as np

# Title of the app
st.title('My First Streamlit App')

# Generate some data
data = pd.DataFrame(
    np.random.randn(100, 2),
    columns=['x', 'y']
)

# Display the data as a line chart
st.line_chart(data)
