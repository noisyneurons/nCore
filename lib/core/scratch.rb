require_relative 'Utilities'

def dominantExamplesForCluster
  (0..numExamples).find_all { |i| membershipWeightForEachExample[i] >= 0.5 }
end


numExamples = 3
membershipWeightForEachExample = [0.55, 0.6, 0.0]

puts (0...numExamples).to_a.find_all { |i| membershipWeightForEachExample[i] >= 0.5 }

puts (0...numExamples).find_all { |i| membershipWeightForEachExample[i] >= 0.5 }

puts (0...numExamples).to_a

puts [0, 1, 2]


(1..3).each { |ii| puts ii }