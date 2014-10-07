require_relative 'Utilities'
require_relative 'DataSet'


k1 = Random.new(2)

anorm1 = NormalDistribution.new(0.0,1.0)
5.times do
  puts "#{Kernel.rand}"
end


k2 = Random.new(2)

anorm2 = NormalDistribution.new(0.0,1.0)

5.times do
  puts "#{k2.rand}"
end



