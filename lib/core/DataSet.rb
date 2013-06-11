### VERSION "nCore"
## ../nCore/lib/core/DataSet.rb

require_relative 'Utilities'

module ExampleDistribution

  def distributeDataToInputAndOutputNeurons(examples, arrayOfArraysOfInputAndOutputNeurons)
    arrayOfArraysOfInputAndOutputNeurons.each { |anArray| distributeDataToAnArrayOfNeurons(anArray, examples) }
  end


  def distributeDataToAnArrayOfNeurons(anArray, examples)
    anArray.each_with_index { |aNeuron, arrayIndexToExampleData| distributeSelectedDataToNeuron(aNeuron, examples, aNeuron.keyToExampleData, arrayIndexToExampleData) }
  end

  private

  def distributeSelectedDataToNeuron(theNeuron, examples, keyToExampleData, arrayIndexToExampleData)
    theNeuron.arrayOfSelectedData = examples.collect { |anExample| anExample[keyToExampleData][arrayIndexToExampleData] }
  end
end

