require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require 'distribution'



#puts Distribution::Normal.cdf(1.96)
#puts norm_cdf(1.96)

class GaussModel
  attr_reader :mean, :std, :prior, :bayesNumerator
  include Distribution
  include Distribution::Shorthand

  def initialize(mean, std, prior, numberOfExamples)
    @numberOfExamples = numberOfExamples
    @mean = mean
    @std = std
    @prior = prior
    @bayesNumerator = nil
    @examplesProbability = nil

    @n = nil
    @newMean = nil
    @m2 = nil
    @delta = nil
    self.reInitialize
  end

  def reInitialize
    @n = 0.0
    @newMean = 0.0
    @m2 = 0.0
  end

  def calculateBayesNumerator(neuronsInputOrOutput)

    @bayesNumerator = @prior * norm_pdf( (neuronsInputOrOutput- @mean) / @std )
    #puts "\nmean= #{mean}"
    #puts "neuronsInputOrOutput= #{neuronsInputOrOutput}"
    #puts "bayesNumerator= #{@bayesNumerator}"
  end

  def estimateProbabilityOfExample(bayesDenominator)
    @examplesProbability = @bayesNumerator / bayesDenominator
    #puts "\nmean= #{mean}"
    #puts "bayesDenominator= #{bayesDenominator}"
    #puts "@examplesProbability= #{@examplesProbability}"
  end

  def prepForRecalculatingModelsParams(x)
    puts "\nmean= #{mean}"
    @n += @examplesProbability
    puts "n= #{@n}"
    @delta = (x - @newMean) * @examplesProbability
    puts "newMean1= #{@newMean}"
    @newMean = @newMean + (@delta / @n)  if(@n > 0.0)
    puts "newMean2= #{@newMean}"
    deltaPrime = (x - @newMean) * @examplesProbability
    @m2 = @m2 + (@delta * deltaPrime)
    puts "m2= #{@m2}"
  end

  def recalculateModelsParamsAtEndOfEpoch
    @mean = @newMean

    variance = @m2/(@n - 1)
    newStd = Math.sqrt(variance)
    @std = newStd

    @prior = @n / @numberOfExamples
  end

  def to_s
    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
  end
end


#mean = 0.0
#std = 1.0
#prior = 1.0
#numberOfExamples = 10
#
#bMod =  GaussModel.new(mean, std, prior, numberOfExamples)

class ExampleDistributionModel
  attr_reader :models

  def initialize(args)
    @args = args
    @numberOfExamples = args[:numberOfExamples]
    @mean = [1.0, -1.0]
    @std = [1.0, 1.0]
    @prior = [0.5, 0.5]
    @models = []
  end

  # use this method at beginning of EACH META-EPOCH
  def createInitialModels
    numberOfModels = @mean.length
    puts "number of models= #{numberOfModels}"
    numberOfModels.times { |i| @models << GaussModel.new(@mean[i], @std[i], @prior[i], @numberOfExamples) }
  end

  # use this method at beginning of EACH EPOCH  -- YES
  def startNextIterationToImproveModel
    @models.each { |model| model.reInitialize }
  end

  # use this method for EACH EXAMPLE   -- YES
  def useExampleToImproveDistributionModel(netInputOrOutput)
    @models.each { |model| model.calculateBayesNumerator(netInputOrOutput) }
    bayesDenominator = @models.inject(0.0) { |sum, model| sum + model.bayesNumerator }
    @models.each do |model|
      model.estimateProbabilityOfExample(bayesDenominator)
      model.prepForRecalculatingModelsParams(netInputOrOutput)
    end
  end

  # use this method at the END of EACH EPOCH -- YES
  def calculateModelParams
    @models.each { |model| model.recalculateModelsParamsAtEndOfEpoch }
  end

  def calcError(netInput)
  end

  def to_s
    models.inject("") { |concat, model| concat + model.to_s }
  end
end

include Distribution
include Distribution::Shorthand

## Just creating synthetic data
positiveDist = norm_rng(mean = 0.5, sigma = 0.05)
negativeDist = norm_rng(mean = -1.5, sigma = 0.05)
numDataExamples = 6
args = {}
args[:numberOfExamples] = numDataExamples
syntheticData = []

(numDataExamples/2).times { syntheticData << positiveDist.call }
(numDataExamples/2).times { syntheticData << negativeDist.call }
syntheticData.shuffle!
puts syntheticData

distributionModel = ExampleDistributionModel.new(args)
distributionModel.createInitialModels      # meta-epoch level
puts distributionModel.to_s
puts

def doAnEpoch(distributionModel, syntheticData)
  distributionModel.startNextIterationToImproveModel
  syntheticData.each do |exampleValue|
    distributionModel.useExampleToImproveDistributionModel(exampleValue)
  end
  distributionModel.calculateModelParams
end



30.times do |i|
  doAnEpoch(distributionModel, syntheticData)
   puts distributionModel.to_s
  puts
end
# distributionModel.startNextIterationToImproveModel


#distributionModel.useExampleToImproveDistributionModel(x)