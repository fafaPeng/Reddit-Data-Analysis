import sys
from pyspark.sql import SparkSession, functions as F, types

spark = SparkSession.builder.appName('Reddit data processing').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

def main(in_directory, out_directory):

    reddit_data = spark.read.json(in_directory)


    reddit_data_filtered = reddit_data.filter(
        (reddit_data['author'] != '[deleted]') &
        (reddit_data['body'] != '[deleted]') &
        (reddit_data['edited'] != 'true')
    )


    reddit_data_with_timestamp = reddit_data_filtered.withColumn(
        'timestamp',
        F.from_unixtime(reddit_data_filtered['created_utc']).cast(types.TimestampType())
    )


    reddit_data_with_daytype = reddit_data_with_timestamp.withColumn(
        'daytype',
        F.when(
            F.date_format(reddit_data_with_timestamp['timestamp'], 'E').isin(['Sat', 'Sun']),
            'weekend'
        ).otherwise('weekday')
    )


    reddit_data_with_day_of_week = reddit_data_with_daytype.withColumn(
        'day_of_week',
        F.when(F.date_format(reddit_data_with_daytype['timestamp'], 'E') == 'Mon', 1)
        .when(F.date_format(reddit_data_with_daytype['timestamp'], 'E') == 'Tue', 2)
        .when(F.date_format(reddit_data_with_daytype['timestamp'], 'E') == 'Wed', 3)
        .when(F.date_format(reddit_data_with_daytype['timestamp'], 'E') == 'Thu', 4)
        .when(F.date_format(reddit_data_with_daytype['timestamp'], 'E') == 'Fri', 5)
        .when(F.date_format(reddit_data_with_daytype['timestamp'], 'E') == 'Sat', 6)
        .otherwise(7)
    )


    reddit_data_selected = reddit_data_with_day_of_week.select(
        reddit_data_with_day_of_week['author'],
        reddit_data_with_day_of_week['body'],
        reddit_data_with_day_of_week['timestamp'],
        reddit_data_with_day_of_week['score'],
        reddit_data_with_day_of_week['subreddit'],
        reddit_data_with_day_of_week['ups'],
        reddit_data_with_day_of_week['daytype'],
        reddit_data_with_day_of_week['day_of_week']
        # reddit_data_with_day_of_week['downs']
    )


    reddit_data_selected.write.json(out_directory, compression='uncompressed', mode='overwrite')

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
