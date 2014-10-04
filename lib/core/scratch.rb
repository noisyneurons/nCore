require_relative 'Utilities'
require_relative 'DataSet'




def layerDetermination(learningLayers, allNeuronLayers)
  learningLayer = learningLayers[0]
  propagatingLayers = []
  controllingLayer = nil

  allNeuronLayers.each do |aLayer|
    propagatingLayers << aLayer
    break if aLayer == learningLayer
    controllingLayer = aLayer
  end

  controllingLayers = [controllingLayer]
  return [propagatingLayers, controllingLayers]
end


learningLayers = [[2]]
allNeuronLayers = [[0], [1], [2], [3]]

learningLayers = [[3]]
allNeuronLayers = [[0], [1], [2], [3]]



propagatingLayers, controllingLayers = layerDetermination(learningLayers, allNeuronLayers)

puts "propagatingLayers = #{propagatingLayers}\n\n"

puts "controllingLayers = #{controllingLayers}"