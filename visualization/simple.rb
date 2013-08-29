require 'redis'

# Notes: PutLoveInYourHeart (:host => "192.168.1.128", :port => 8765)
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.127", :port => 8765) Wired
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.131", :port => 8765) Wireless


redis = Redis.new # (:host => "192.168.1.131", :port => 8765)

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
