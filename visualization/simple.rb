require 'redis'

require_relative '../lib/core/Utilities'

$redis = Redis.new(:host => $currentHost)


anArray = [1, 2, 3, 4, 5]
x = Marshal.dump(anArray)
redis.set("myKey", x)
y = redis.get("myKey")
maybeAnArray = Marshal.load(y)
puts "maybeAnArray=\t#{maybeAnArray.class}"
puts "array= #{maybeAnArray}"
puts "*****************************"


arrayOfKeys = redis.keys("Net*")
puts arrayOfKeys

puts "*****************************"

puts redis.get("myKey").class

puts "*****************************"

#puts redis.smembers("NetInputs:epochs:primary_key")

puts "type=\t#{redis.type("NetInputs:values:4")}"

puts "get field names and values for the key ('NetInputs:values:4')=\t#{redis.hgetall("NetInputs:values:4")}"

## puts redis.flushdb
