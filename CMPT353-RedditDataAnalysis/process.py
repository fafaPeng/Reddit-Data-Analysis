import sys
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat

import nltk

USERID = 'cla429'  # REPLACE 'youruserid' WITH YOUR ACTUAL USERID

# nltk.data.path.append('/home/cla429/nltk_data')

nltk.data.path.append(f'/home/{USERID}/nltk_data')



# Initialize AVDER model
sia = SentimentIntensityAnalyzer()



@F.pandas_udf(returnType=FloatType())
def compute_sentiment(text_series: pd.Series) -> pd.Series:
    return text_series.apply(lambda text: sia.polarity_scores(text)["compound"])



@F.pandas_udf(returnType=FloatType())
def compute_readability(text_series: pd.Series) -> pd.Series:
    return text_series.apply(lambda text: textstat.flesch_kincaid_grade(text))


def main(in_directory, out_directory):

    spark = SparkSession.builder.appName("Reddit Analysis").getOrCreate()


    reddit_data = spark.read.json(in_directory)

    # Get the sentiment score
    reddit_data = reddit_data.withColumn(
        "sentiment_score", compute_sentiment(reddit_data["body"])
    )

    # Get the readability score
    reddit_data = reddit_data.withColumn(
        "readability_score", compute_readability(reddit_data["body"])
    )


    high_threshold = 0.9
    low_threshold = 0.1


    subreddit_window = Window.partitionBy('subreddit').orderBy('score')


    reddit_data_with_percentile = reddit_data.withColumn(
        "percentile",
        F.percent_rank().over(subreddit_window)
    )

    reddit_data_with_quality = reddit_data_with_percentile.withColumn(
        "quality",
        F.when(F.col("percentile") >= high_threshold, "good")
        .when(F.col("percentile") <= low_threshold, "bad")
        .otherwise(F.when(F.col("score") < -5, "bad").otherwise("normal")),
    )



    reddit_data = reddit_data.dropDuplicates()

    reddit_data_with_quality.write.json(out_directory, mode='overwrite')


if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
