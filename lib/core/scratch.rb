require_relative 'Utilities'
require 'relix'
require 'yaml'

$redis = Redis.new
#$redis.flushdb


ary = $redis.keys("*")
ary.each { |item| p item }

# $redis.shutdown




