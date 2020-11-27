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

# Show the data
airports.show()
