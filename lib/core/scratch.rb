require_relative 'Utilities'
require_relative 'DataSet'






learningLayers = [[2]]
allNeuronLayers = [[0], [1], [2], [3]]

learningLayers = [[3]]
allNeuronLayers = [[], [], [], [3]]

puts allNeuronLayers.flatten.empty?

#propagatingLayers, controllingLayers = layerDetermination(learningLayers, allNeuronLayers)
#
#puts "propagatingLayers = #{propagatingLayers}\n\n"
#
#puts "controllingLayers = #{controllingLayers}"