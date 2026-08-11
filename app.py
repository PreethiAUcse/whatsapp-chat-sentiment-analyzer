import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('vader_lexicon')


st.set_page_config(page_title="WhatsApp Chat and Sentiment Analyzer", layout="wide")

# Sidebar
st.sidebar.title(' WhatsApp Chat and Sentiment Analyzer')
st.sidebar.markdown("Upload your exported chat file below to begin analysis.")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=[".txt"])

# Main logic
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # User Selection
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Analyze chat for:", user_list)

    if st.sidebar.button("Show Analysis"):

        # Top Stats
        st.header(' Top Statistics')
        num_messages, words, num_media_messages, links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", num_messages)
        with col2:
            st.metric("Total Words", words)
        with col3:
            st.metric("Media Shared", num_media_messages)
        with col4:
            st.metric("Links Shared", links)

        # Most Busy Users
        if selected_user == "Overall":
            st.header(" Most Active Users")
            x, new_df = helper.most_busy_user(df)
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation=90)
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Word Cloud
        st.header('☁️ Word Cloud')
        df_wc = helper.create_word_cloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        ax.axis('off')
        st.pyplot(fig)

        # Emoji Analysis
        st.header(' Emoji Usage')
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(emoji_df)
        with col2:
            if not emoji_df.empty:
                fig, ax = plt.subplots()
                ax.pie(emoji_df['count'].head(5), labels=emoji_df['emoji'].head(5), autopct="%0.2f%%")
                st.pyplot(fig)

        # Monthly Timeline
        st.header("📆 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green', marker='o')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.header("🗓️ Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='blue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.header(' Activity Map')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Most Active Days')
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            sns.barplot(x=busy_day.index, y=busy_day.values, ax=ax, palette="viridis")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.subheader('Most Active Months')
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            sns.barplot(x=busy_month.index, y=busy_month.values, ax=ax, palette="rocket")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        user_heatmap = helper.activity_heat_map(selected_user, df)
        st.header('Heat Map')
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Sentiment Analysis
        st.header("Sentiment Analysis")
        total_positive, total_negative, total_neutral = helper.sentiment_analysis(selected_user, df)

        total_msgs = total_positive + total_negative + total_neutral
        if total_msgs == 0:
            st.info("No valid messages for sentiment analysis.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", f"{(total_positive / total_msgs) * 100:.2f}%")
            with col2:
                st.metric("Negative", f"{(total_negative / total_msgs) * 100:.2f}%")
            with col3:
                st.metric("Neutral", f"{(total_neutral / total_msgs) * 100:.2f}%")

            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            ax.pie([total_positive, total_negative, total_neutral],
                   labels=['Positive', 'Negative', 'Neutral'],
                   autopct='%0.2f%%',
                   colors=['lightgreen', 'salmon', 'lightgray'])
            st.pyplot(fig)






