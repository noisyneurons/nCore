require_relative 'Utilities'
require_relative 'DataSet'
require_relative 'NeuralIOFunctions'

include NonMonotonicIOFunction


#k1 = Random.new(2)
#
#anorm1 = NormalDistribution.new(0.0,1.0)
#5.times do
#  puts "#{Kernel.rand}"
#end
#
#
#k2 = Random.new(2)
#
#anorm2 = NormalDistribution.new(0.0,1.0)
#
#5.times do
#  puts "#{k2.rand}"
#end


#def genTestingArray(offset, increment, gridDivisions)
#  testXs = (0..gridDivisions.to_i).collect do |aValue|
#    x = offset + (aValue * increment)
#    y = ioFunction(x)
#    [x, y]
#  end
#end
#
#def findMax
#  gridDivisions = 20.0
#  offset = 0.0
#  increment = 5.0 / gridDivisions
#  maxXY = nil
#  10.times do
#    testXs = genTestingArray(offset, increment, gridDivisions)
#    maxXY = testXs.max { |a, b| a[1] <=> b[1] }
#    offset = maxXY[0] - increment
#    increment = 2.0 * increment / gridDivisions
#  end
#  maxXY
#end

#testXs = genTestingArray(offset, increment)
#maxXY = testXs.max {|a,b| a[1] <=> b[1] }
#offset = maxXY[0] - increment
#increment = increment / 100.0
#testXs = genTestingArray(offset, increment)
#maxXY = testXs.max {|a,b| a[1] <=> b[1] }
#end

srand(3)
10.times do
  puts (Kernel.rand - 0.5)
end

puts
srand(3)
10.times do
  puts (rand - 0.5)
end




