require_relative 'Utilities'
require 'relix'
require 'yaml'

theComputersName = ENV["COMPUTERNAME"]
puts "theComputersName = #{theComputersName}"
puts "Socket.gethostname = #{Socket.gethostname}"

#$redis = Redis.new
##$redis.flushdb
#
#
#ary = $redis.keys("*")
#ary.each { |item| p item }
#
## $redis.shutdown
#
#
#

