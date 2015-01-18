require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require 'distribution'

puts Distribution::Normal.cdf(1.96)

randomNumberGenerator = Distribution::Normal.rng
ary = []
100.times do
  v = randomNumberGenerator.call
  ary << v
  print "#{v}\t"
end
puts

puts ary.mean
puts ary.standardError

ary = []
10000.times do
  v = randomNumberGenerator.call
  ary << v
  #dprint "#{v}\t"
end
puts

puts ary.mean
puts ary.standardError




