require_relative 'Utilities'

require 'redis'
require 'yaml'

# Notes: PutLoveInYourHeart (:host => "192.168.1.128", :port => 8765)
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.127", :port => 8765) Wired
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.131", :port => 8765) Wireless


redis = Redis.new # (:host => "192.168.1.131", :port => 8765)
def pushData(key, data, redis)
  $redis.rpush(key,YAML.dump(data))
end

def retrieveAllData(key, redis)
  veryLargeInteger = 999999999999
  aListOfData = $redis.lrange(key, 0, veryLargeInteger)
  aListOfData.collect {|aDataItem| YAML.load( aDataItem )}
end


anArray = [1,2]
key = "myData"
data = anArray
pushData(key, data, redis)
data << 5
pushData(key, data, redis)


puts "\n\nLet's see..."
puts "Call to retrieveAllData #{retrieveAllData(key, redis)}"



#puts "try=\t#{YAML.load(YAML.dump(anArray))}"


puts "\n\n********** LIST **********"
anArray = [1,2,3]
x1 = YAML.dump(anArray)
puts "x1=\t#{x1}"

anArray2 = [2,3,4]
x2 = YAML.dump(anArray2)
redis.lpush("dataList",x1)
redis.lpush("dataList",x2)
#valuesReturned = redis.ltrim("dataList",0,1)
puts "valueReturned=\t#{YAML.load(redis.lpop("dataList"))}"
puts "valueReturned=\t#{YAML.load(redis.lpop("dataList"))}"
#puts "valuesReturned[0]=\t#{valuesReturned[0].class}"

puts "*****************************"


def storeData(key, data, redis)
  redis.lpush(key,YAML.dump(data))
end

def retrieveData(key, redis)
  data = YAML.load(redis.lpop(key))
end









puts "*****************************"

anArray = [1,2,3]
x = Marshal.dump(anArray)
redis.set("myKey", x)
y = redis.get("myKey")
maybeAnArray = Marshal.load(y)
puts "maybeAnArray=\t#{maybeAnArray.class}"
puts "*****************************"






anArray = [1,2,3]
x = Marshal.dump(anArray)
redis.set("myKey", x)
y = redis.get("myKey")
maybeAnArray = Marshal.load(y)
puts "maybeAnArray=\t#{maybeAnArray.class}"
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
