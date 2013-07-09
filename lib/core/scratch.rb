require_relative 'Utilities'
require 'relix'
require 'yaml'

$redis = Redis.new
# $redis.flushdb


ary = $redis.keys("*")
ary.each { |item| p item }

p "############ckjlkjfdlks"

p $redis.type("FlockData:values:0.356")

#p $redis.zrevrangebyscore("FlockData:values:0.356","-inf","+inf")

