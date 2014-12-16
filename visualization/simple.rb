require 'redis'

require_relative '../lib/core/Utilities'

$redis = Redis.new(:host => $currentHost)


anArray = [1, 2, 3, 4, 5]
x = Marshal.dump(anArray)
redis.set("myKey", x)
y = redis.get("myKey")
maybeAnArray = Marshal.load(y)
logger.puts "maybeAnArray=\t#{maybeAnArray.class}"
logger.puts "array= #{maybeAnArray}"
logger.puts "*****************************"


arrayOfKeys = redis.keys("Net*")
logger.puts arrayOfKeys

logger.puts "*****************************"

logger.puts redis.get("myKey").class

logger.puts "*****************************"

#logger.puts redis.smembers("NetInputs:epochs:primary_key")

logger.puts "type=\t#{redis.type("NetInputs:values:4")}"

logger.puts "get field names and values for the key ('NetInputs:values:4')=\t#{redis.hgetall("NetInputs:values:4")}"

## logger.puts redis.flushdb
