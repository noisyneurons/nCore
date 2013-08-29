require 'redis'

# Notes: PutLoveInYourHeart (:host => "192.168.1.128", :port => 8765)
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.127", :port => 8765) Wired
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.131", :port => 8765) Wireless

redis = Redis.new # (:host => "192.168.1.131", :port => 8765)

arrayOfKeys = redis.keys("NeuronData*")
puts arrayOfKeys


puts "##########################################################################################################################################"
arrayOfKeys = redis.keys("ND*")
puts arrayOfKeys

puts " &&&&&&&&&&&&&&&&&&&&&&&&&&& "
puts redis.get("ND144.36")

#arrayOfKeys = redis.keys("*")
#puts arrayOfKeys

puts "\n  Current Experiment Number=\t #{redis.get("experimentNumber")}"



