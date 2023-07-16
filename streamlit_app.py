import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from funcs import *
import streamlit.components.v1 as components
import plotly.express as px
import matplotlib.pyplot as plt 

def main():


    sidebar_header =  'H&M Fashion Recommendation System'
    
    page_options = ["Data Overview",
                    "Customer Recommendations",
                    "Product Captioning"]
    
    st.sidebar.info(sidebar_header)


    
    page_selection = st.sidebar.radio("Try", page_options)
    articles_df = pd.read_csv('articles.csv')
    
    models = ['Similar items based on image embeddings', 
              'Similar items based on text embeddings', 
              'Similar items based discriptive features', 
              'Similar items based on embeddings from TensorFlow Recommendrs model',
              'Similar items based on a combination of all embeddings']
    
    model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub',
                  'Features embeddings are calculated by one-hot encoding the descriptive features provided by H&M',
                  'TFRS model performes a collaborative filtering based ranking using a neural network', 
                  'A concatenation of all embeddings above is used to find similar items']

#########################################################################################
#########################################################################################
    if page_selection == "Data Overview":
        customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')

        # Hiển thị thông tin cơ bản về dữ liệu customers_rcmnds
        st.title('Data Overview - Customer Recommendations')
        st.markdown('''Develop product recommendations based on data from previous transactions of H&M Groups with 53 online markets and approximately 4,850 stores., as well as from customer and product meta data. 
            But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, product recommendations are key. More importantly, helping customers make the right choices also has a positive implications for sustainability, as it reduces returns, and thereby minimizes emissions from transportation.
            The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images.This is the data overview for the CSV file: customers_rcmnds.csv.''')

        # Thêm biểu đồ tròn
        
        customer_counts = customers_rcmnds['customer'].value_counts()
        fig_pie = px.pie(labels=customer_counts.index, values=customer_counts.values, title="Customer Purchase Count")
        st.plotly_chart(fig_pie)
       
# Đường dẫn tới file CSV

    # Tiêu đề và mô tả chung
        st.write('Total number of unique customers:', len(customers_rcmnds.customer.unique()))

        # Hiển thị số lượng hàng và cột của DataFrame
        st.header('Data Shape')
        st.write('Number of rows:', customers_rcmnds.shape[0])
        st.write('Number of columns:', customers_rcmnds.shape[1])

        # Hiển thị dữ liệu mẫu
        st.header('Sample Data')
        st.write(customers_rcmnds.head())

        # Kết thúc ứng dụng
        st.stop()


                              
#########################################################################################
#########################################################################################
    if page_selection == "Customer Recommendations":
        
        customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
        customers = customers_rcmnds.customer.unique()        
        
        get_item = st.sidebar.button('Get Random Customer')
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


if __name__ == '__main__':
    main()
