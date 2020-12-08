# Creating a SparkSession
# We've already created a SparkSession for you called spark, but what if you're not sure there already is one?
# Creating multiple SparkSessions and SparkContexts can cause issues, so it's best practice to use the SparkSession.builder.getOrCreate() method.
# This returns an existing SparkSession if there's already one in the environment, or creates a new one if necessary!

# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)


# Once you've created a SparkSession, you can start poking around to see what data is in your cluster!
# Your SparkSession has an attribute called catalog which lists all the data inside the cluster.
# This attribute has a few methods for extracting different pieces of information.
# One of the most useful is the .listTables() method,
# which returns the names of all the tables in your cluster as a list.

# Print the tables in the catalog
print(spark.catalog.listTables())

# Don't change this query
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()

# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())

# Put some Spark in your data
# In the last exercise, you saw how to move data from Spark to pandas.
# However, maybe you want to go the other direction, and put a pandas DataFrame into a Spark cluster!
# The SparkSession class has a method for this as well.
#
# The .createDataFrame() method takes a pandas DataFrame and returns a Spark DataFrame.
#
# The output of this method is stored locally, not in the SparkSession catalog.
# This means that you can use all the Spark DataFrame methods on it, but you can't access the data in other contexts.
#
# For example, a SQL query (using the .sql() method) that references your DataFrame will throw an error.
# To access the data in this way, you have to save it as a temporary table.
#
# You can do this using the .createTempView() Spark DataFrame method,
# which takes as its only argument the name of the temporary table you'd like to register.
# This method registers the DataFrame as a table in the catalog, but as this table is temporary,
# it can only be accessed from the specific SparkSession used to create the Spark DataFrame.
#
# There is also the method .createOrReplaceTempView().
# This safely creates a new temporary table if nothing was there before, or updates an existing table if one was already defined.
# You'll use this method to avoid running into problems with duplicate tables.

# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())

# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path,header = True)


# Notice that in the first case, we pass a string to .filter(). In SQL,
# we would write this filtering task as SELECT * FROM flights WHERE air_time > 120.
# Spark's .filter() can accept any expression that could go in the WHEREclause of a SQL query

# Filter flights by passing a string
long_flights1 = flights.filter("distance > 1000")

# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000)

# Select the first set of columns
selected1 = flights.select('tailnum','origin','dest')

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Select the columns "origin", "dest", "tailnum", and avg_speed (without quotes!).

# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")

# This creates a GroupedData object (so you can use the .min() method),
# then finds the minimum value in col, and returns it as a DataFrame.

# Find the shortest flight from PDX in terms of distance
flights.filter("origin == 'PDX'").groupBy().min('distance').show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == 'SEA').groupBy().max('air_time').show()

# Grouping and Aggregating II
# In addition to the GroupedData methods you've already seen, there is also the .agg() method.
# This method lets you pass an aggregate column expression
# that uses any of the aggregate functions from the pyspark.sql.functions submodule.

# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month","dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()

# Joining

# Rename the faa column
airports = airports.withColumnRenamed('faa','dest')

# Join the DataFrames
flights_with_airports = flights.join(airports, on = 'dest', how = 'leftouter')

# Machine Learning Pipelines
# In the next two chapters you'll step through every stage of the machine learning pipeline,
# from data intake to model evaluation. Let's get to it!
#
# At the core of the pyspark.ml module are the Transformer and Estimator classes.
# Almost every other class in the module behaves similarly to these two basic classes.
#
# Transformer classes have a .transform() method that takes a DataFrame and returns a new DataFrame;
# usually the original one with a new column appended.
# For example, you might use the class Bucketizer to create discrete bins
# from a continuous feature or the class PCA to reduce the dimensionality
# of your dataset using principal component analysis.
#
# Estimator classes all implement a .fit() method.
# These methods also take a DataFrame, but instead of returning another DataFrame they return a model object.
# This can be something like a StringIndexerModel for including categorical data saved as strings in your models,
# or a RandomForestModel that uses the random forest algorithm for classification or regression.

# Fortunately, PySpark has functions for handling this built into the pyspark.ml.features submodule.
#
# The first step to encoding your categorical feature is to create a StringIndexer.
# Members of this class are Estimators that take a DataFrame with a column of strings and map each unique string to a number.
# Then, the Estimator returns a Transformer that takes a DataFrame, attaches the mapping to it as metadata,
# and returns a new DataFrame with a numeric column corresponding to the string column.
#
# The second step is to encode this numeric column as a one-hot vector using a OneHotEncoder.
# This works exactly the same way as the StringIndexer by creating an Estimator and then a Transformer.
# The end result is a column that encodes your categorical feature as a vector that's suitable for machine learning routines!
#
# This may seem complicated, but don't worry! All you have to remember is that
# you need to create a StringIndexer and a OneHotEncoder, and the Pipeline will take care of the rest.

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol = 'carrier', outputCol = 'carrier_index')

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol = 'carrier_index', outputCol = 'carrier_fact')

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=['month', 'air_time', 'carrier_fact', 'dest_fact', 'plane_age'], outputCol='features')

# Create the pipeline
# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([0.6,0.4])

# Select all the unique council voters
voter_df = df.select(df["VOTER NAME"]).distinct()

# Count the rows in voter_df
print("\nThere are %d rows in the voter_df DataFrame.\n" % voter_df.count())

# Add a ROW_ID
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the rows with 10 highest IDs in the set
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)

To check the number of partitions, use the method .rdd.getNumPartitions() on a DataFrame.

print("\nThere are %d partitions in the voter_df DataFrame.\n" % voter_df.rdd.getNumPartitions())

# Removing commented lines
# Import the file to a DataFrame and perform a row count
annotations_df = spark.read.csv('annotations.csv.gz', sep='|')
full_count = annotations_df.count()

# Count the number of rows beginning with '#'
comment_count = annotations_df.filter(col('_c0').startswith('#')).count()

# Import the file to a new DataFrame, without commented rows
no_comments_df = spark.read.csv('annotations.csv.gz', sep='|', comment='#')

# Count the new DataFrame and verify the difference is as expected
no_comments_count = no_comments_df.count()
print("Full count: %d\nComment count: %d\nRemaining count: %d" % (full_count, comment_count, no_comments_count))

# Removing invalid rows
# Split _c0 on the tab character and store the list in a variable
tmp_fields = F.split(annotations_df['_c0'], '\t')

# Create the colcount column on the DataFrame
annotations_df = annotations_df.withColumn('colcount', F.size(tmp_fields))

# Remove any rows containing fewer than 5 fields
annotations_df_filtered = annotations_df.filter(~ (annotations_df.colcount<5))

# Count the number of rows
final_count = annotations_df_filtered.count()
print("Initial count: %d\nFinal count: %d" % (initial_count, final_count))
