import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from funcs import *
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.decomposition import TruncatedSVD

# %matplotlib inline
plt.style.use("ggplot")
                                                                

def main():
        
    st.sidebar.markdown('<h1 style="color: red;">Ecommerce Product Recommendation System</h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 style="color: red;">Lecturer: Đỗ Duy Thanh</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 style="color: red;"> Group Member:</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 > Nguyễn Thị Bích Dao-21521925 </h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 > Lê Thị Kiều-21522268 </h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 > Nguyễn Thanh Tri-21521569</h1>', unsafe_allow_html=True)
    

    page_options = ["Data Overview And Visualize Data",
                    "Customer Recommendations",]
    

    
    choice = st.sidebar.radio("Includes", page_options)
    articles_df = pd.read_csv('articles.csv')
    
    models = ['Similar items based on image embeddings',  
              'Similar items based discriptive features', 
              'Similar items based on embeddings from TensorFlow Recommendrs model']
    
    model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'Features embeddings are calculated by one-hot encoding the descriptive features provided by H&M',
                  'TFRS model performes a collaborative filtering based ranking using a neural network']



#########################################################################################
#########################################################################################
    if choice == "Customer Recommendations":
        #data customers_rcmnds.csv
        customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
        customers = customers_rcmnds.customer.unique()        
        
        get_item = st.sidebar.button('Result')
        if get_item:
            st.sidebar.write('#### Customer history')

            rand_customer = np.random.choice(customers)
            customer_data = customers_rcmnds[customers_rcmnds.customer == rand_customer]
            customer_history = np.array(eval(customer_data.history.iloc[0]))

            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(customer_data)
            
            scores = get_rcmnds_scores(customer_data)
            features = get_rcmnds_features(articles_df, combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            images = get_rcmnds_images(combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)

            splits = [customer_history[i:i+3] for i in range(0, len(customer_history), 3)]
                            
            for split in splits:
                with st.sidebar.container():
                    cols = st.columns(3)
                    for item, col in zip(split, cols):
                        col.image(get_item_image(str(item), 100))
                    

            with st.container():            
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded=True)
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
            
            
#########################################################################################  
#########################################################################################
#########################################################################################
    if choice == "Data Overview And Visualize Data":
            customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')

           
        #data articles
            customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
            st.markdown('<h2 style="color: red;">Data overview for the customers_rcmnds.csv.</h2>', unsafe_allow_html=True)
            # Data Visualization: Bar chart
            st.header('Bar Chart - Frequency of Categories')
            selected_column = st.selectbox('Select a column:', customers_rcmnds.columns)
            customers_rcmnds[selected_column] = customers_rcmnds[selected_column].astype('category')
            if customers_rcmnds[selected_column].dtype == 'category':
                value_counts = customers_rcmnds[selected_column].value_counts()
                fig_bar = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': selected_column, 'y': 'Frequency'})
                st.plotly_chart(fig_bar)
            else:
                st.write('Selected column is not categorical. Please choose a categorical column for the bar chart.')
            # Total number of rows and columns
            st.header('Data Shape')
            st.write('Number of rows:', customers_rcmnds.shape[0])
            st.write('Number of columns:', customers_rcmnds.shape[1])

            # Show the raw data
            st.header('Sample Data')
            st.write(customers_rcmnds.head())

            # Summary statistics
            st.header('Summary Statistics')
            st.write(customers_rcmnds.describe())

            
            
            st.stop()


if __name__ == '__main__':
    main()