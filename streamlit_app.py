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
        
    sidebar_header = '''Ecommerce Product Recommendation System'''  
    
    page_options = ["Data Overview And Visualize Data",
                    "Find similar items",
                    "Customer Recommendations",]
    
    st.sidebar.title(sidebar_header)
    
    # def custom_header(text):
    #     header_html = "<h1 style='text-align: center; color: white; '>{}</h1>".format(text)
    #     st.markdown(header_html, unsafe_allow_html=True)

    #     # Sử dụng hàm custom_header() để tạo header tùy chỉnh
    #     sidebar_header = custom_header("Ecommerce Product Recommendation System")
    #     st.sidebar.info(sidebar_header)



    
    choice = st.sidebar.radio("Content", page_options)
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

    if choice == "Find similar items":

        articles_rcmnds = pd.read_csv('results/articles_rcmnds.csv')
        # amazon_ratings = amazon_ratings.dropna()
        # amazon_ratings.head()
        # amazon_ratings.shape
    
        articles = articles_rcmnds.article_id.unique()
        get_item = st.sidebar.button('Result')
        
        if get_item:
            
            rand_article = np.random.choice(articles)
            article_data = articles_rcmnds[articles_rcmnds.article_id == rand_article]
            rand_article_desc = articles_df[articles_df.article_id == rand_article].detail_desc.iloc[0]
            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(article_data)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            scores = get_rcmnds_scores(article_data)
            features = get_rcmnds_features(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            images = get_rcmnds_images(image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            st.sidebar.image(get_item_image(str(rand_article), width=200, height=300))
            st.sidebar.write('Article description')
            st.sidebar.caption(rand_article_desc)

            with st.container():     
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded = model == 'Similar items based on image embeddings' or model == 'Similar items based on text embeddings')
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
                                if model == 'Similar items based on text embeddings':
                                   st.caption(detail_desc)
            # popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
            # most_popular = popular_products.sort_values('Rating', ascending=False)
            # most_popular.head(10) 

            # most_popular.head(30).plot(kind = "bar")

                                    
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

            # Hiển thị thông tin cơ bản về dữ liệu customers_rcmnds
            st.title('Data Overview And Visualize Data - Customer Recommendations')
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

        #data articles
            articles = pd.read_csv('results/articles_rcmnds.csv')
            st.markdown('This is the data overview for the CSV file.')
            # Total number of rows and columns
            st.header('Data Shape')
            st.write('Number of rows:', articles.shape[0])
            st.write('Number of columns:', articles.shape[1])

            # Show the raw data
            st.header('Sample Data')
            st.write(articles.head())

            # Summary statistics
            st.header('Summary Statistics')
            st.write(articles.describe())

            # Data Visualization: Bar chart
            st.header('Bar Chart - Frequency of Categories')
            selected_column = st.selectbox('Select a column:', articles.columns)
            articles[selected_column] = articles[selected_column].astype('category')
            if articles[selected_column].dtype == 'category':
                value_counts = articles[selected_column].value_counts()
                fig_bar = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': selected_column, 'y': 'Frequency'})
                st.plotly_chart(fig_bar)
            else:
                st.write('Selected column is not categorical. Please choose a categorical column for the bar chart.')
            st.stop()


if __name__ == '__main__':
    main()