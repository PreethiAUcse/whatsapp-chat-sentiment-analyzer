import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_user(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index':'Name', 'user':'Percent'})
    return x, df

def create_word_cloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out unwanted messages
    filtered_df = df[
        (~df['message'].str.contains('<Media omitted>', na=False)) &
        (df['message'].str.lower() != 'null') &
        (df['user'] != 'group_notification') &
        (df['message'].str.strip() != '')
    ]

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black')
    df_wc = wc.generate(filtered_df['message'].str.cat(sep=" "))
    return df_wc


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heat_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out media messages
    df = df[~df['message'].str.contains('<Media omitted>', na=False)]

    sentiments = SentimentIntensityAnalyzer()

    total_positive = 0
    total_negative = 0
    total_neutral = 0
    total_messages = 0

    for msg in df['message']:
        score = sentiments.polarity_scores(msg)
        total_messages += 1
        if score['compound'] >= 0.05:
            total_positive += 1
        elif score['compound'] <= -0.05:
            total_negative += 1
        else:
            total_neutral += 1

    return total_positive, total_negative, total_neutral


