require_relative '../lib/core/DataSet'


def createSimplest2GaussianClustersForTrainingSet
  examples = []
  numberOfClasses = 2
  numberOfExamplesInEachClass = (numberOfExamples=10) / numberOfClasses

  negativeNormalCluster = NormalDistribution.new(-1.0, 0.3)
  positiveNormalCluster = NormalDistribution.new(1.0, 0.3)

  arrayOfClassIndexAndClassRNGs = []

  classIndexAndClassRNG = [(classIndex = 0), (classRNG = negativeNormalCluster)]
  arrayOfClassIndexAndClassRNGs << classIndexAndClassRNG
  classIndexAndClassRNG = [(classIndex = 1), (classRNG = positiveNormalCluster)]
  arrayOfClassIndexAndClassRNGs << classIndexAndClassRNG

  exampleNumber = 0
  arrayOfClassIndexAndClassRNGs.each do |classIndexAndClassRNG|
    classIndex = classIndexAndClassRNG[0]
    desiredOutputs = [classIndex.to_f]
    numberOfExamplesInEachClass.times do |classExNumb|

      randomNumberGenerator = classIndexAndClassRNG[1]
      x = randomNumberGenerator.rng
      aPoint = [x]

      examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => classIndex}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end


norm = NormalDistribution.new(10, 0.2)
cdf = norm.cdf(10.0)
pdf = norm.pdf(10.0)
puts cdf
puts pdf

loopcnt = 0

puts "generating random numbers"
while (loopcnt < 10)
  puts norm.rng(n=1)
  loopcnt += 1
end


puts createSimplest2GaussianClustersForTrainingSet
